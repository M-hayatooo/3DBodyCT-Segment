"""Patient-grouped five-fold cross-validation for 34 CT scans from 15 identifiable patients.

For Fat, the default label values are 1 for subcutaneous adipose tissue (SAT)
and 2 for visceral adipose tissue (VAT). Use --class_names to change the names
in label-value order (1, 2, ...). Examples::

    python 5fold_test.py --split_only
    python 5fold_test.py --dry_run --allow_label_crop
    python 5fold_test.py --allow_label_crop
    python 5fold_test.py --summarize_only --output_dir weights/NEW_RUN

Only CT scans named PID_YYYY or PID_YYYYMMDD are included. Anonymized images
without an identifiable patient ID are excluded from both training and
evaluation. Longitudinal images from the same patient are assigned to the same
fold. By default, each fold has 3 test patients and 12 training patients.
A new UNet is trained for 3000 epochs with batch size 4, and the final epoch is
evaluated. The test set is not used for epoch selection or early stopping.
As in the original code, one slice is randomly sampled from each training case
per epoch. As in train.py, all three slices in the input stack must be labeled.

Preprocessing uses utils.load. The default, matching the paper and original
train.py, is a 256 x 256 grid at 1.6 mm spacing (409.6 mm square). Specify
--image_size 320 to extend the field of view to 512 mm square at the same
spacing. To reproduce the original 256-grid settings while permitting labels
outside the field of view, explicitly specify --allow_label_crop.

By default, evaluation includes only annotated slices from the S1 centroid
slice, inclusive, to the T12 centroid slice, exclusive. TotalSegmentator label
values 26 and 32 identify S1 and T12, respectively. Training uses the full
annotated range, as in the original code. Use --evaluation_region all_annotated
to evaluate all annotated slices. Background-only slices are treated as
unannotated; partially annotated regions of interest are not supported.
Examination-level metrics are averaged within each patient, and patient-level
values are weighted equally in the overall summary. The 95% confidence intervals
use patient-level bootstrap sampling. Examination-level summaries are also saved
as secondary results in a separate CSV file.

To specify inputs in a CSV file, use --manifest CSV. The columns are case_id,
image_path, label_path, and group_id. If group_id is omitted, the patient ID is
extracted from the filename; if supplied, it must match that ID. Relative paths
are resolved from the CSV directory.

Use --cache_source to reuse preprocessing caches from an existing experiment
only when the inputs and preprocessing settings match. Existing model weights
are not reused. --split_only performs patient selection and splitting only;
--dry_run also validates the cache and evaluation region. Folds can be run
separately, for example with --folds 1, and completed folds are reused.
Interrupted folds are trained again from the beginning. Use a different
--output_dir if settings or input files change.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import nibabel as nib
import numpy as np
import torch
from monai.losses import DiceCELoss
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import BodyDataset
from utils.functions import seed_worker
from utils.load import DEFAULT_IN_PLANE_SPACING, conform_image
from utils.network import UNet


NUM_CLASSES = {"Fat": 3, "Whole": 3, "Muscle": 7, "Muscle2": 11}
METRICS = ("dice", "iou", "precision", "recall", "area_bias_cm2", "area_mae_cm2")
PID_PATTERN = re.compile(r"(?P<pid>[0-9]+)_(?P<date>[0-9]{4}|[0-9]{8})\Z")


def patient_id_from_case_id(case_id):
    """Only the documented numeric PID/date convention is accepted."""
    match = PID_PATTERN.fullmatch(case_id)
    if not match:
        return None
    date = match["date"]
    try:
        datetime.strptime(date, "%Y" if len(date) == 4 else "%Y%m%d")
    except ValueError:
        return None
    return match["pid"]


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def create_parser(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", default=datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d"))
    parser.add_argument("--mode", choices=tuple(NUM_CLASSES), default="Fat")
    parser.add_argument("--data_root", type=Path, help="Parent directory containing voxels/, labels/ and total/; default: /media/toru/DATA/BodyCT/MODE")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--expected_cases", type=int, default=34, help="Expected included CT examinations after PID filtering")
    parser.add_argument("--expected_patients", type=int, default=15)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", help="1-based fold numbers; default: all folds")
    parser.add_argument("--class_names", nargs="+", help="Foreground names in label-value order: 1, 2, ...")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--in_plane_spacing", type=float, default=DEFAULT_IN_PLANE_SPACING)
    parser.add_argument(
        "--image_size",
        type=int,
        choices=(256, 320),
        default=256,
        help="Input grid size: 256 (default; 409.6 mm at 1.6 mm spacing) or 320 (512 mm)",
    )
    parser.add_argument("--intensity_aug_prob", type=float, default=0.5)
    parser.add_argument("--intensity_shift_hu", type=float, default=20.0)
    parser.add_argument("--intensity_scale_min", type=float, default=0.9)
    parser.add_argument("--intensity_scale_max", type=float, default=1.1)
    parser.add_argument("--gaussian_noise_prob", type=float, default=0.5)
    parser.add_argument("--gaussian_noise_std_min", type=float, default=0.02)
    parser.add_argument("--gaussian_noise_std_max", type=float, default=0.12)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--qc_dice_threshold", type=float, default=0.8, help="Flag cases for visual review; does not exclude them")
    parser.add_argument("--allow_label_crop", action="store_true", help="Explicitly allow foreground outside the preprocessing field of view")
    parser.add_argument("--total_dir", type=Path, help="TotalSegmentator labels; default: DATA_ROOT/total")
    parser.add_argument("--evaluation_region", choices=("s1_t12", "all_annotated"), default="s1_t12")
    parser.add_argument("--cache_source", type=Path, help="Existing experiment directory containing run_config.json and cache/")
    parser.add_argument("--split_only", action="store_true", help="Write inclusion audit and patient folds without building image caches or training")
    parser.add_argument("--dry_run", action="store_true", help="Validate images, build cache and splits, without training")
    parser.add_argument("--summarize_only", action="store_true", help="Aggregate completed folds without loading source images")
    opt = parser.parse_args(argv)
    if opt.output_dir is None:
        opt.output_dir = Path(f"weights/{opt.day}_{opt.mode}_{opt.expected_patients}patient_{opt.n_splits}fold_batch{opt.batch_size}_{opt.image_size}px")
    if opt.data_root is None:
        data_mode = "Muscle_2" if opt.mode == "Muscle2" else opt.mode
        opt.data_root = Path("/media/toru/DATA/BodyCT/") / data_mode
    if opt.total_dir is None:
        opt.total_dir = opt.data_root / "total"
    if opt.evaluation_region == "s1_t12" and opt.mode != "Fat":
        parser.error("S1-T12 evaluation in this script is intended for Fat; use --evaluation_region all_annotated for other modes")
    if opt.class_names is None:
        opt.class_names = ["SAT", "VAT"] if opt.mode == "Fat" else [f"label_{i}" for i in range(1, NUM_CLASSES[opt.mode])]
    if len(opt.class_names) != NUM_CLASSES[opt.mode] - 1 or len(set(opt.class_names)) != len(opt.class_names):
        parser.error("--class_names must contain one unique name for each foreground class")
    for name in ("expected_cases", "expected_patients", "epochs", "batch_size", "eval_batch_size", "bootstrap_samples"):
        if getattr(opt, name) < 1:
            parser.error(f"--{name} must be positive")
    if opt.n_splits < 2 or opt.num_workers < 0 or opt.lr <= 0 or opt.in_plane_spacing <= 0 or opt.weight_decay < 0:
        parser.error("Invalid split count, workers, learning rate, spacing or weight decay")
    if opt.n_splits > opt.expected_patients:
        parser.error("--n_splits cannot exceed --expected_patients")
    if opt.summarize_only and (opt.split_only or opt.dry_run):
        parser.error("--summarize_only cannot be combined with --split_only or --dry_run")
    if not 0 <= opt.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    for name in ("intensity_aug_prob", "gaussian_noise_prob", "qc_dice_threshold"):
        if not 0 <= getattr(opt, name) <= 1:
            parser.error(f"--{name} must be between 0 and 1")
    if not 0 < opt.intensity_scale_min <= opt.intensity_scale_max or not 0 <= opt.gaussian_noise_std_min <= opt.gaussian_noise_std_max or opt.intensity_shift_hu < 0:
        parser.error("Invalid augmentation range")
    if opt.folds and (len(set(opt.folds)) != len(opt.folds) or any(f < 1 or f > opt.n_splits for f in opt.folds)):
        parser.error("--folds must contain unique numbers between 1 and n_splits")
    return opt


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def discover_cases(opt, excluded_cases=None):
    cases = []
    excluded_cases = [] if excluded_cases is None else excluded_cases

    def include_case(case_id, image_path, label_path, supplied_group=None):
        pid = patient_id_from_case_id(case_id)
        if pid is None:
            excluded_cases.append({"case_id": case_id, "image_path": str(image_path),
                                   "label_path": str(label_path), "reason": "patient_id_not_identifiable_from_filename"})
            return False
        if supplied_group and supplied_group != pid:
            raise ValueError(f"{case_id}: group_id {supplied_group!r} must equal filename PID {pid!r}")
        cases.append({"case_id": case_id, "group_id": pid,
                      "image_path": str(image_path), "label_path": str(label_path)})
        return True

    if opt.manifest:
        manifest = opt.manifest.resolve()
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"case_id", "image_path", "label_path"}.issubset(reader.fieldnames or []):
                raise ValueError("Manifest requires case_id,image_path,label_path columns")
            for row in reader:
                paths = {}
                for key in ("image_path", "label_path"):
                    path = Path(row[key])
                    paths[key] = (manifest.parent / path).resolve()
                case_id = row["case_id"].strip()
                filename_id = paths["image_path"].name.removesuffix("_voxel.nii.gz")
                if case_id != filename_id:
                    raise ValueError(f"Manifest case_id {case_id!r} does not match CT filename {filename_id!r}")
                include_case(case_id, paths["image_path"], paths["label_path"], (row.get("group_id") or "").strip())
    else:
        root = opt.data_root.expanduser().resolve()
        image_root = root / "voxels"
        label_root = root / "labels"
        for directory in (image_root, label_root):
            if not directory.is_dir():
                raise FileNotFoundError(f"Data directory not found: {directory}. Check --data_root.")
        suffix = "_label.nii.gz" if opt.mode == "Muscle2" else f"_{opt.mode}.nii.gz"
        # Match by path and filename, never by the order of two glob results.
        labels = set(label_root.rglob(f"*{suffix}"))
        used = set()
        for image in sorted(image_root.rglob("*_voxel.nii.gz")):
            label = (label_root / image.relative_to(image_root)).with_name(image.name.removesuffix("_voxel.nii.gz") + suffix)
            case_id = image.name.removesuffix("_voxel.nii.gz")
            if not include_case(case_id, image, label):
                continue
            if not label.is_file():
                raise FileNotFoundError(f"Missing manual label for {image}: expected {label}")
            used.add(label)
        orphan_labels = [path for path in labels - used
                         if patient_id_from_case_id(path.name.removesuffix(suffix)) is not None]
        if orphan_labels:
            raise ValueError(f"Manual labels with identifiable PID but without matching images: {sorted(map(str, orphan_labels))}")
    cases.sort(key=lambda case: case["case_id"])
    if len(cases) != opt.expected_cases:
        raise ValueError(f"Expected {opt.expected_cases} included CT examinations, found {len(cases)} after PID filtering; excluded {len(excluded_cases)}.")
    n_patients = len({case["group_id"] for case in cases})
    if n_patients != opt.expected_patients:
        raise ValueError(f"Expected {opt.expected_patients} patients, found {n_patients} after PID filtering")
    for key in ("case_id", "image_path", "label_path"):
        values = [case[key] for case in cases]
        if not all(values) or len(set(values)) != len(values):
            raise ValueError(f"Empty or duplicate {key}; use a manifest with unique case IDs")
    if any(not case["group_id"] for case in cases):
        raise ValueError("group_id cannot be empty")
    for case in cases:
        for key in ("image_path", "label_path"):
            info = Path(case[key]).stat()
            case[key + "_size"] = info.st_size
            case[key + "_mtime_ns"] = info.st_mtime_ns
    return cases


def make_splits(cases, n_splits, seed):
    groups = sorted({case["group_id"] for case in cases})
    if len(groups) < n_splits:
        raise ValueError("Number of independent patient groups must be >= n_splits")
    shuffled = list(np.random.default_rng(seed).permutation(groups))
    # Balance patient counts first (3 per fold for 15 patients), then CT counts.
    # The seeded permutation breaks ties without splitting any patient's images.
    counts = Counter(case["group_id"] for case in cases)
    shuffled.sort(key=lambda group: -counts[group])
    capacities = [len(chunk) for chunk in np.array_split(groups, n_splits)]
    test_groups, sizes = [set() for _ in range(n_splits)], [0] * n_splits
    for group in shuffled:
        eligible = [fold for fold in range(n_splits) if len(test_groups[fold]) < capacities[fold]]
        fold = min(eligible, key=lambda index: (sizes[index], len(test_groups[index]), index))
        test_groups[fold].add(group)
        sizes[fold] += counts[group]
    splits = []
    for number, groups_in_fold in enumerate(test_groups, 1):
        train = [i for i, case in enumerate(cases) if case["group_id"] not in groups_in_fold]
        test = [i for i, case in enumerate(cases) if case["group_id"] in groups_in_fold]
        splits.append({"fold": number, "train_indices": train, "test_indices": test,
                       "train_case_ids": [cases[i]["case_id"] for i in train],
                       "test_case_ids": [cases[i]["case_id"] for i in test],
                       "train_group_ids": sorted({cases[i]["group_id"] for i in train}),
                       "test_group_ids": sorted(groups_in_fold)})
    validate_splits(cases, splits)
    return splits


def validate_splits(cases, splits):
    """Fail before training if even one patient occurs in train and test."""
    all_ids = {case["case_id"] for case in cases}
    by_id = {case["case_id"]: case["group_id"] for case in cases}
    held_out = []
    held_out_groups = []
    for split in splits:
        train, test = set(split["train_case_ids"]), set(split["test_case_ids"])
        if not test or not train or train & test or train | test != all_ids:
            raise ValueError(f"Fold {split['fold']}: invalid examination partition")
        train_groups = {by_id[case_id] for case_id in train}
        test_groups = {by_id[case_id] for case_id in test}
        if train_groups & test_groups:
            raise ValueError(f"Fold {split['fold']}: patient leakage: {sorted(train_groups & test_groups)}")
        if train_groups != set(split["train_group_ids"]) or test_groups != set(split["test_group_ids"]):
            raise ValueError(f"Fold {split['fold']}: inconsistent patient list")
        held_out.extend(split["test_case_ids"])
        held_out_groups.extend(split["test_group_ids"])
    if len(held_out) != len(cases) or set(held_out) != all_ids:
        raise ValueError("Each examination must be held out exactly once")
    if len(held_out_groups) != len(set(by_id.values())) or set(held_out_groups) != set(by_id.values()):
        raise ValueError("Each patient must belong to exactly one test fold")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def foreground_outside_fov(labels, native_affine, target_affine, shape):
    transform = np.linalg.inv(target_affine) @ native_affine
    count = 0
    for z in np.flatnonzero(np.any(labels > 0, axis=(0, 1))):
        xy = np.argwhere(labels[:, :, z] > 0)
        xyz = np.column_stack((xy, np.full(len(xy), z)))
        mapped = xyz @ transform[:3, :3].T + transform[:3, 3]
        count += int(np.count_nonzero(np.any((mapped < -0.5) | (mapped > np.asarray(shape) - 0.5), axis=1)))
    return count


def prepare_case(case, cache_dir, opt):
    """Keep annotated centers plus CT context on disk; training still uses all levels."""
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists() and (cache_dir / "images.npy").exists() and (cache_dir / "labels.npy").exists():
        metadata = read_json(metadata_path)
        if (metadata.get("case_id") == case["case_id"] and metadata.get("group_id") == case["group_id"]
                and metadata.get("image_size") == opt.image_size and "training_rows" in metadata
                and np.allclose(metadata["spacing_mm"][:2], opt.in_plane_spacing)):
            return metadata
    image = nib.squeeze_image(nib.as_closest_canonical(nib.load(case["image_path"])))
    label = nib.squeeze_image(nib.as_closest_canonical(nib.load(case["label_path"])))
    if len(image.shape) != 3 or image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-4, rtol=0):
        raise ValueError(f"{case['case_id']}: CT and manual label must share the same 3D shape and affine")
    native = label.get_fdata(dtype=np.float32)
    if not np.all(np.isfinite(native)) or not np.array_equal(native, np.rint(native)) or native.min() < 0 or native.max() >= NUM_CLASSES[opt.mode]:
        raise ValueError(f"{case['case_id']}: labels must be integers in [0, {NUM_CLASSES[opt.mode] - 1}]")
    conformed = conform_image(image, opt.in_plane_spacing, opt.image_size)
    outside = foreground_outside_fov(native, label.affine, conformed.affine, conformed.shape)
    if outside and not opt.allow_label_crop:
        raise ValueError(f"{case['case_id']}: {outside} foreground pixels outside the {opt.image_size}x{opt.image_size} preprocessing field of view. Increase --image_size to preserve spacing, increase --in_plane_spacing, or explicitly use --allow_label_crop.")
    target_image = conform_image(label, opt.in_plane_spacing, opt.image_size, is_label=True)
    targets = np.asarray(target_image.dataobj, dtype=np.int16).transpose(2, 1, 0)
    indices = np.flatnonzero(np.any(targets > 0, axis=(1, 2)))
    if not len(indices):
        raise ValueError(f"{case['case_id']}: no annotated foreground slice survives preprocessing")
    if not set(np.unique(native)).issubset(set(np.unique(targets)) | {0}):
        raise ValueError(f"{case['case_id']}: a foreground class disappears during resampling")
    training_indices = BodyDataset.get_slice_indices(targets)
    if not len(training_indices):
        raise ValueError(f"{case['case_id']}: no 3 consecutive labeled slices for training under the supplied train.py protocol")
    training_rows = np.searchsorted(indices, training_indices).tolist()
    # Fixed HU normalization, with the same windows as BodyDataset.
    low, high = (-2000.0, 150.0) if opt.mode == "Whole" else (-500.0, 500.0)
    axial = conformed.get_fdata(dtype=np.float32).transpose(2, 1, 0)
    if not np.all(np.isfinite(axial)):
        raise ValueError(f"{case['case_id']}: non-finite CT intensities")
    axial = (2.0 * (np.clip(axial, low, high) - low) / (high - low) - 1.0).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stacks = np.lib.format.open_memmap(cache_dir / "images.npy", mode="w+", dtype=np.float32,
                                       shape=(len(indices), 3, opt.image_size, opt.image_size))
    for row, center in enumerate(indices):
        for channel, z in enumerate((center - 1, center, center + 1)):
            stacks[row, channel] = axial[z] if 0 <= z < len(axial) else -1.0
    stacks.flush()
    del stacks
    np.save(cache_dir / "labels.npy", targets[indices])
    metadata = {"case_id": case["case_id"], "group_id": case["group_id"],
                "cache_dir": str(cache_dir.resolve()), "slice_indices": indices.tolist(),
                "n_native_labeled_slices": int(np.count_nonzero(np.any(native > 0, axis=(0, 1)))),
                "n_labeled_slices": len(indices), "foreground_pixels_outside_fov": outside,
                "training_rows": training_rows, "n_training_slices": len(training_rows),
                "image_size": opt.image_size,
                "class_values": np.unique(targets).tolist(), "spacing_mm": list(map(float, conformed.header.get_zooms()))}
    write_json(metadata_path, metadata)
    return metadata


def load_reusable_cache(cases, source, opt):
    """Reuse only preprocessing arrays, after checking source files and grid settings."""
    if source is None:
        return {}
    source = source.expanduser().resolve()
    run = read_json(source / "run_config.json")
    source_options = run["options"]
    for key in ("mode", "image_size", "in_plane_spacing", "allow_label_crop"):
        if source_options.get(key) != getattr(opt, key):
            raise ValueError(f"Cache source {source}: preprocessing setting {key} differs")
    source_cases = {case["case_id"]: case for case in run["cases"]}
    metadata_index = {}
    for path in sorted((source / "cache").glob("case_*/metadata.json")):
        metadata = read_json(path)
        if metadata["case_id"] in metadata_index:
            raise ValueError(f"Duplicate cache case_id in {source}: {metadata['case_id']}")
        metadata_index[metadata["case_id"]] = (path.parent, metadata)
    reused = {}
    keys = ("image_path", "label_path", "image_path_size", "image_path_mtime_ns", "label_path_size", "label_path_mtime_ns")
    for case in cases:
        case_id = case["case_id"]
        original = source_cases.get(case_id)
        if original is None or any(case[key] != original.get(key) for key in keys):
            raise ValueError(f"{case_id}: cache source CT/label paths or file fingerprints differ")
        if case_id not in metadata_index:
            raise FileNotFoundError(f"{case_id}: cache metadata missing in {source}")
        root, metadata = metadata_index[case_id]
        if (metadata.get("image_size") != opt.image_size or "training_rows" not in metadata
                or not np.allclose(metadata["spacing_mm"][:2], opt.in_plane_spacing)):
            raise ValueError(f"{case_id}: cache metadata grid/training protocol differs")
        images = np.load(root / "images.npy", mmap_mode="r")
        labels = np.load(root / "labels.npy", mmap_mode="r")
        n_slices = len(metadata["slice_indices"])
        if images.shape != (n_slices, 3, opt.image_size, opt.image_size) or labels.shape != (n_slices, opt.image_size, opt.image_size):
            raise ValueError(f"{case_id}: cached array shapes are invalid")
        # Original metadata may have used the examination ID as group_id.
        # Correct it in memory without changing the original experiment.
        reused[case_id] = {**metadata, "group_id": case["group_id"], "cache_dir": str(root)}
    print(f"Reusing preprocessing arrays for {len(reused)} examinations from {source}; no model weights reused.")
    return reused


def get_evaluation_window(case, metadata, opt):
    """Determine model-independent annotated rows in the requested axial region."""
    indices = np.asarray(metadata["slice_indices"], dtype=int)
    z_s1 = z_t12 = None
    total_info = {}
    if opt.evaluation_region == "s1_t12":
        path = opt.total_dir.expanduser().resolve() / f"{case['case_id']}_label.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"{case['case_id']}: TotalSegmentator label missing: {path}")
        total = nib.squeeze_image(nib.as_closest_canonical(nib.load(path)))
        image = nib.squeeze_image(nib.as_closest_canonical(nib.load(case["image_path"])))
        if len(total.shape) != 3 or total.shape != image.shape or not np.allclose(total.affine, image.affine, atol=1e-4, rtol=0):
            raise ValueError(f"{case['case_id']}: TotalSegmentator labels and CT must share canonical shape and affine")
        data = np.asarray(total.dataobj)
        centers = []
        for label_value in (26, 32):
            mask = data == label_value
            if not np.any(mask):
                raise ValueError(f"{case['case_id']}: vertebral label {label_value} missing")
            # Matches FatCSV.py: integer truncation of the center of mass.
            z_coordinates, counts = np.unique(np.nonzero(mask)[2], return_counts=True)
            centers.append(int(np.average(z_coordinates, weights=counts)))
        z_s1, z_t12 = centers
        if not z_s1 < z_t12 or not len(indices) or indices.min() < 0 or indices.max() >= total.shape[2]:
            raise ValueError(f"{case['case_id']}: invalid vertebral boundaries or slice indices")
        rows = np.flatnonzero((indices >= z_s1) & (indices < z_t12)).tolist()
        info = path.stat()
        total_info = {"total_path": str(path), "total_path_size": info.st_size, "total_path_mtime_ns": info.st_mtime_ns}
    else:
        rows = list(range(len(indices)))
    if not rows:
        raise ValueError(f"{case['case_id']}: no annotated slices in {opt.evaluation_region}")
    return {"case_id": case["case_id"], "group_id": case["group_id"],
            "evaluation_region": opt.evaluation_region, "z_s1": z_s1, "z_t12": z_t12,
            "n_annotated_slices": len(indices), "n_window_slices": len(rows),
            "evaluation_rows": rows, "evaluation_slice_indices": indices[rows].tolist(), **total_info}


class FoldTrainingDataset(BodyDataset):
    """One randomly selected labeled center per patient image and epoch."""

    def __init__(self, cases, opt):
        self.cases = cases
        # Reuse the existing augmentation implementation, without loading volumes.
        self.mode = opt.mode
        self.is_train = True
        self.gaussian_noise_prob = opt.gaussian_noise_prob
        self.gaussian_noise_std_range = (opt.gaussian_noise_std_min, opt.gaussian_noise_std_max)
        self.intensity_aug_prob = opt.intensity_aug_prob
        self.intensity_shift_hu = opt.intensity_shift_hu
        self.intensity_scale_range = (opt.intensity_scale_min, opt.intensity_scale_max)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        root = Path(case["cache_dir"])
        row = int(np.random.choice(case["training_rows"]))
        image = np.load(root / "images.npy", mmap_mode="r")[row].transpose(1, 2, 0).copy()
        label = np.load(root / "labels.npy", mmap_mode="r")[row].copy()
        image = self.add_gaussian_noise(self.add_intensity_augmentation(image))
        if random.random() < 0.8:
            image, label = self.random_stride(image, label)
        if random.random() < 0.8:
            image, label = self.random_rotate_2d(image, label)
        return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32), np.ascontiguousarray(label[None], dtype=np.int64)


def train_fold(cases, fold_dir, fold, opt, device):
    seed_everything(opt.seed + fold)
    generator = torch.Generator().manual_seed(opt.seed + fold)
    loader = DataLoader(FoldTrainingDataset(cases, opt), batch_size=opt.batch_size, shuffle=True,
                        num_workers=opt.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=opt.num_workers > 0, worker_init_fn=seed_worker, generator=generator)
    # No pretrained weights: all five models are initialized independently.
    model = UNet(3, NUM_CLASSES[opt.mode]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * 0.1)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True).to(device)
    amp = opt.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp)
    fold_dir.mkdir(parents=True, exist_ok=True)
    with (fold_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "learning_rate"])
        writer.writeheader()
        progress = tqdm(range(1, opt.epochs + 1), desc=f"Fold {fold}: training")
        for epoch in progress:
            model.train()
            loss_sum, count = 0.0, 0
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device.type, enabled=amp):
                    loss = criterion(model(images), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Fold {fold}, epoch {epoch}: non-finite training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(images)
                count += len(images)
            writer.writerow({"epoch": epoch, "train_loss": loss_sum / count, "learning_rate": optimizer.param_groups[0]["lr"]})
            stream.flush()
            scheduler.step()
            progress.set_postfix(loss=f"{loss_sum / count:.4f}")
    model_dir = fold_dir / "model"
    model_dir.mkdir(exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, model_dir / "UNet.pth")
    return model


def overlap_metrics(tp, predicted, reference):
    """Both-empty overlap is undefined; missed reference / false positive is zero."""
    return {"dice": 2.0 * tp / (predicted + reference) if predicted + reference else None,
            "iou": tp / (predicted + reference - tp) if predicted + reference else None,
            "precision": tp / predicted if predicted else (0.0 if reference else None),
            "recall": tp / reference if reference else None}


@torch.inference_mode()
def evaluate_fold(model, cases, fold, opt, device):
    model.eval()
    case_rows, slice_rows = [], []
    for case in tqdm(cases, desc=f"Fold {fold}: held-out evaluation"):
        root = Path(case["cache_dir"])
        images = np.load(root / "images.npy", mmap_mode="r")
        labels = np.load(root / "labels.npy", mmap_mode="r")
        evaluation_rows = case["evaluation_rows"]
        counts = {value: np.zeros(3, dtype=np.int64) for value in range(1, NUM_CLASSES[opt.mode])}
        area_errors = {value: [] for value in counts}
        pixel_area = case["spacing_mm"][0] * case["spacing_mm"][1] / 100.0
        for start in range(0, len(evaluation_rows), opt.eval_batch_size):
            selected = evaluation_rows[start:start + opt.eval_batch_size]
            batch = torch.from_numpy(np.array(images[selected])).to(device)
            with autocast(device.type, enabled=opt.use_amp and device.type == "cuda"):
                logits = model(batch)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"{case['case_id']}: non-finite prediction")
            predictions = logits.argmax(dim=1).cpu().numpy()
            for offset, prediction in enumerate(predictions):
                position = selected[offset]
                target = labels[position]
                for value, name in enumerate(opt.class_names, 1):
                    pred_mask, ref_mask = prediction == value, target == value
                    tp = int(np.count_nonzero(pred_mask & ref_mask))
                    predicted, reference = int(pred_mask.sum()), int(ref_mask.sum())
                    counts[value] += (tp, predicted, reference)
                    bias = (predicted - reference) * pixel_area
                    area_errors[value].append(bias)
                    slice_rows.append({"fold": fold, "case_id": case["case_id"], "group_id": case["group_id"],
                        "slice_index": case["slice_indices"][position], "class_value": value, "class_name": name,
                        "tp_pixels": tp, "pred_pixels": predicted, "ref_pixels": reference,
                        **overlap_metrics(tp, predicted, reference), "pred_area_cm2": predicted * pixel_area,
                        "ref_area_cm2": reference * pixel_area, "area_bias_cm2": bias, "area_mae_cm2": abs(bias)})
        for value, name in enumerate(opt.class_names, 1):
            tp, predicted, reference = map(int, counts[value])
            metrics = overlap_metrics(tp, predicted, reference)
            case_rows.append({"fold": fold, "case_id": case["case_id"], "group_id": case["group_id"],
                "class_value": value, "class_name": name, "n_labeled_slices": len(evaluation_rows),
                "evaluation_region": opt.evaluation_region, "z_s1": case["z_s1"], "z_t12": case["z_t12"],
                "tp_pixels": tp, "pred_pixels": predicted, "ref_pixels": reference, **metrics,
                "pred_mean_area_cm2": predicted * pixel_area / len(evaluation_rows),
                "ref_mean_area_cm2": reference * pixel_area / len(evaluation_rows),
                "area_bias_cm2": float(np.mean(area_errors[value])), "area_mae_cm2": float(np.mean(np.abs(area_errors[value]))),
                "qc_low_dice": metrics["dice"] is not None and metrics["dice"] < opt.qc_dice_threshold,
                "qc_foreground_cropped": case["foreground_pixels_outside_fov"] > 0})
    return case_rows, slice_rows


def summarize_rows(rows, names, seed, bootstrap_samples):
    """Equal row weighting; percentile bootstrap resamples patient groups.

    Use examination rows for the secondary summary and patient means for the primary.
    """
    summary = []
    for value, name in enumerate(names, 1):
        class_rows = [row for row in rows if row["class_value"] == value]
        for metric in METRICS:
            valid = [row for row in class_rows if row[metric] is not None]
            values = np.array([row[metric] for row in valid], dtype=float)
            stats = {"class_value": value, "class_name": name, "metric": metric, "n_cases": len(values),
                     "n_groups": len({row["group_id"] for row in valid}), "mean": None, "std": None,
                     "median": None, "q25": None, "q75": None, "min": None, "max": None,
                     "ci95_low": None, "ci95_high": None}
            if len(values):
                stats.update(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else None,
                             median=float(np.median(values)), q25=float(np.quantile(values, 0.25)),
                             q75=float(np.quantile(values, 0.75)), min=float(values.min()), max=float(values.max()))
                groups = sorted({row["group_id"] for row in valid})
                if len(groups) > 1:
                    sums = np.array([sum(row[metric] for row in valid if row["group_id"] == group) for group in groups])
                    sizes = np.array([sum(row["group_id"] == group for row in valid) for group in groups])
                    rng = np.random.default_rng(seed + value)
                    means = []
                    for start in range(0, bootstrap_samples, 1000):
                        sampled = rng.integers(0, len(groups), size=(min(1000, bootstrap_samples - start), len(groups)))
                        means.extend((sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)).tolist())
                    stats["ci95_low"], stats["ci95_high"] = map(float, np.quantile(means, [0.025, 0.975]))
            summary.append(stats)
    return summary


def make_patient_rows(case_rows):
    """Mean of examination-level scores within a patient, never a pixel-pooled score."""
    grouped = {}
    for row in case_rows:
        grouped.setdefault((row["group_id"], row["class_value"]), []).append(row)
    patient_rows = []
    for (group_id, value), rows in sorted(grouped.items()):
        if len({row["fold"] for row in rows}) != 1:
            raise ValueError(f"Patient {group_id} appears in multiple held-out folds")
        if len({row["case_id"] for row in rows}) != len(rows):
            raise ValueError(f"Patient {group_id}: duplicate examination/class metrics")
        result = {"fold": rows[0]["fold"], "group_id": group_id,
                  "class_value": value, "class_name": rows[0]["class_name"],
                  "n_examinations": len(rows)}
        for metric in METRICS:
            valid = [row[metric] for row in rows if row[metric] is not None and np.isfinite(row[metric])]
            result[metric] = float(np.mean(valid)) if valid else None
            result[f"n_defined_{metric}"] = len(valid)
        patient_rows.append(result)
    return patient_rows


def summarize_patients(case_rows, names, seed, bootstrap_samples):
    patient_rows = make_patient_rows(case_rows)
    summary = summarize_rows(patient_rows, names, seed, bootstrap_samples)
    for row in summary:
        row["n_patients"] = row.pop("n_cases")
        row["unit"] = "patient_mean_of_examination_metrics"
    return patient_rows, summary


def validate_fold_result(result, run, split):
    """Validate cached OOF results before reuse or reporting."""
    case_index = {case["case_id"]: case for case in run["cases"]}
    expected = {(case_id, value) for case_id in split["test_case_ids"]
                for value in range(1, len(run["options"]["class_names"]) + 1)}
    found = {(row["case_id"], row["class_value"]) for row in result["case_metrics"]}
    if result["run_id"] != run["run_id"] or found != expected or len(found) != len(result["case_metrics"]):
        raise ValueError(f"Fold {split['fold']}: inconsistent or duplicate held-out case results")
    windows = {row["case_id"]: row for row in run["evaluation_windows"]}
    for row in result["case_metrics"]:
        window = windows[row["case_id"]]
        if (row["group_id"] != case_index[row["case_id"]]["group_id"] or row["fold"] != split["fold"]
                or row.get("evaluation_region") != run["options"]["evaluation_region"]
                or row["n_labeled_slices"] != window["n_window_slices"]
                or row.get("z_s1") != window["z_s1"] or row.get("z_t12") != window["z_t12"]):
            raise ValueError(f"Fold {split['fold']}: patient ID, fold or region mismatch for {row['case_id']}")
    expected_slices = {(case_id, z, value) for case_id, value in expected
                       for z in windows[case_id]["evaluation_slice_indices"]}
    actual_slices = {(row["case_id"], row["slice_index"], row["class_value"]) for row in result["slice_metrics"]}
    if actual_slices != expected_slices or len(actual_slices) != len(result["slice_metrics"]):
        raise ValueError(f"Fold {split['fold']}: missing, extra or duplicate evaluated slices")
    if any(row["group_id"] != case_index[row["case_id"]]["group_id"] or row["fold"] != split["fold"]
           for row in result["slice_metrics"]):
        raise ValueError(f"Fold {split['fold']}: incorrect patient or fold in slice results")


def aggregate_results(output_dir):
    run = read_json(output_dir / "run_config.json")
    opt = run["options"]
    if run.get("validation_protocol") != "filename_pid_patient_grouped_v1":
        raise ValueError("This script can only summarize its own patient-grouped runs, not the old 71-examination runs")
    validate_splits(run["cases"], run["splits"])
    case_rows, slice_rows, fold_summary, completed = [], [], [], []
    for split in run["splits"]:
        path = output_dir / f"fold_{split['fold']:02d}" / "results.json"
        if not path.exists():
            continue
        result = read_json(path)
        validate_fold_result(result, run, split)
        completed.append(split["fold"])
        case_rows.extend(result["case_metrics"])
        slice_rows.extend(result["slice_metrics"])
        _, stats = summarize_patients(result["case_metrics"], opt["class_names"], opt["seed"], opt["bootstrap_samples"])
        fold_summary.extend({"fold": split["fold"], "n_train_cases": len(split["train_indices"]),
                             "n_test_cases": len(split["test_indices"]),
                             "n_train_patients": len(split["train_group_ids"]),
                             "n_test_patients": len(split["test_group_ids"]), **row} for row in stats)
    if not case_rows:
        print("No completed folds to summarize.")
        return
    unique_cases = {row["case_id"] for row in case_rows}
    unique_patients = {row["group_id"] for row in case_rows}
    complete = (len(completed) == opt["n_splits"] and len(unique_cases) == opt["expected_cases"]
                and len(unique_patients) == opt["expected_patients"])
    patient_rows, summary = summarize_patients(case_rows, opt["class_names"], opt["seed"], opt["bootstrap_samples"])
    case_summary = summarize_rows(case_rows, opt["class_names"], opt["seed"], opt["bootstrap_samples"])
    write_csv(output_dir / "oof_case_metrics.csv", case_rows)
    write_csv(output_dir / "oof_slice_metrics.csv", slice_rows)
    write_csv(output_dir / "oof_patient_metrics.csv", patient_rows)
    write_csv(output_dir / "fold_summary.csv", fold_summary)
    write_csv(output_dir / "overall_summary.csv", summary)
    write_csv(output_dir / "case_summary.csv", case_summary)
    write_json(output_dir / "summary.json", {"complete": complete, "completed_folds": completed,
        "n_evaluated_cases": len(unique_cases), "n_expected_cases": opt["expected_cases"],
        "n_evaluated_patients": len(unique_patients), "n_expected_patients": opt["expected_patients"],
        "evaluation_region": opt["evaluation_region"], "statistics": summary, "secondary_case_statistics": case_summary})
    report = ["BodyCT segmentation: patient-grouped out-of-fold internal validation",
              f"Status: {'COMPLETE' if complete else 'INCOMPLETE - preliminary results only'}",
              f"Cases evaluated: {len(unique_cases)}/{opt['expected_cases']}; completed folds: {completed}",
              f"Patients evaluated: {len(unique_patients)}/{opt['expected_patients']}",
              f"Examinations excluded for unidentifiable PID: {len(run['excluded_cases'])}",
              "All examinations of one patient were assigned to the same fold; no patient occurred in both training and testing.",
              f"Split seed: {opt['seed']}; epochs fixed in advance: {opt['epochs']}",
              "Each fold used a newly initialized 2D UNet with three adjacent CT slices as input.",
              ("Training sampled one center per case and epoch from triplets with labels on all three slices, matching train.py."
               if run.get("training_protocol") == "three_labeled_slices"
               else "Training sampled one foreground-annotated center per case and epoch."),
              "Optimization used AdamW, DiceCELoss (including background), and cosine learning-rate annealing.",
              "Only the training cases were used for optimization; the final epoch was evaluated.",
              ("Evaluation used foreground-annotated slices from the S1 centroid (inclusive) to the T12 centroid (exclusive),"
               if opt["evaluation_region"] == "s1_t12" else "Evaluation used all foreground-annotated slices,"),
              "without augmentation on the resampled",
              f"{opt.get('image_size', 256)}x{opt.get('image_size', 256)} grid (in-plane spacing {opt['in_plane_spacing']} mm). All-background slices",
              "were considered unannotated. Reference labels must cover the entire foreground on each evaluated slice.",
              "Case Dice pools TP/predicted/reference pixel counts across that case's evaluated slices.",
              "Primary patient metrics are within-patient arithmetic means of examination-level metrics (not pixel-pooled scores).",
              "Primary overall statistics weight patients equally; standard deviation uses ddof=1.",
              "Examination-weighted results are secondary and are saved separately in case_summary.csv.",
              f"95% CIs use {opt['bootstrap_samples']} percentile bootstrap samples of patient groups.",
              "These are descriptive intervals conditional on the fixed CV split and fitted models;",
              "they do not quantify training/split variability. Pixel-pooled Dice is reported separately.",
              "Both-empty overlap scores are undefined and excluded (blank in CSV); false positives or missed labels score 0.",
              "Area bias/MAE are within-case means of slice area differences in cm2 on the resampled grid; they are not volumes.",
              "Label names: " + ", ".join(f"{value}={name}" for value, name in enumerate(opt["class_names"], 1)), ""]
    for value, name in enumerate(opt["class_names"], 1):
        report.append(name + ":")
        for stat in (row for row in summary if row["class_value"] == value):
            if stat["mean"] is None:
                report.append(f"  {stat['metric']}: undefined (n=0)")
                continue
            sd = f"{stat['std']:.4f}" if stat["std"] is not None else "undefined"
            ci = f"{stat['ci95_low']:.4f}-{stat['ci95_high']:.4f}" if stat["ci95_low"] is not None else "undefined"
            report.append(f"  {stat['metric']}: {stat['mean']:.4f} +/- {sd}; 95% CI {ci}; patients={stat['n_patients']}")
        rows = [row for row in case_rows if row["class_value"] == value]
        pooled = overlap_metrics(sum(row["tp_pixels"] for row in rows), sum(row["pred_pixels"] for row in rows), sum(row["ref_pixels"] for row in rows))["dice"]
        report.append(f"  pixel-pooled Dice (secondary): {pooled}")
    flagged = {row["case_id"] for row in case_rows if row["qc_low_dice"] or row["qc_foreground_cropped"]}
    report.extend(["", f"Cases flagged for visual review: {len(flagged)} (Dice threshold {opt['qc_dice_threshold']}); none excluded.",
                   "Flagged case IDs: " + ", ".join(sorted(flagged)),
                   "This experiment assesses internal validation against the supplied manual labels.",
                   "It does not assess interobserver agreement or external validation.",
                   "Visual quality control and annotation procedures must be described based on what was actually performed."])
    (output_dir / "reviewer_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Aggregated {len(unique_cases)} examinations / {len(unique_patients)} patients, {len(completed)}/{opt['n_splits']} folds: {output_dir / 'reviewer_report.txt'}")


def main(argv=None):
    opt = create_parser(argv)
    output_dir = opt.output_dir.resolve()
    if opt.summarize_only:
        aggregate_results(output_dir)
        return
    start = time.time()
    excluded_cases = []
    cases = discover_cases(opt, excluded_cases)
    splits = make_splits(cases, opt.n_splits, opt.seed)
    for split in splits:
        print(f"Fold {split['fold']}: train={len(split['train_group_ids'])} patients / {len(split['train_indices'])} examinations, "
              f"test={len(split['test_group_ids'])} patients / {len(split['test_indices'])} examinations")
    print(f"Included {len(cases)} examinations / {opt.expected_patients} patients; excluded {len(excluded_cases)} examinations with unidentifiable PID.")
    assignment = {i: split["fold"] for split in splits for i in split["test_indices"]}
    assignment_rows = [{**case, "test_fold": assignment[i]} for i, case in enumerate(cases)]
    if opt.split_only:
        config_path = output_dir / "run_config.json"
        if config_path.exists():
            existing = read_json(config_path)
            if existing.get("cases") != cases or existing.get("splits") != splits:
                raise ValueError("Output directory contains a different experiment; use a new --output_dir")
        write_csv(output_dir / "included_cases.csv", cases)
        write_csv(output_dir / "excluded_cases.csv", excluded_cases)
        write_csv(output_dir / "fold_assignments.csv", assignment_rows)
        write_json(output_dir / "split_preview.json", {"seed": opt.seed, "n_cases": len(cases),
                   "n_patients": opt.expected_patients, "splits": splits})
        print(f"Patient split checks passed: {output_dir}. No image cache built and no training performed.")
        return

    # Runtime controls do not alter the split or experiment settings.
    excluded = {"day", "data_root", "manifest", "output_dir", "folds", "dry_run", "split_only",
                "summarize_only", "num_workers", "eval_batch_size", "cache_source"}
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(opt).items() if key not in excluded}
    base = Path(__file__).resolve().parent
    sources = [Path(__file__), base / "utils/load.py", base / "utils/dataset.py", base / "utils/network.py", base / "utils/functions.py"]
    code_hash = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        existing = read_json(config_path)
        expected_fields = {"options": settings, "cases": cases, "splits": splits, "code_sha256": code_hash,
                           "validation_protocol": "filename_pid_patient_grouped_v1", "excluded_cases": excluded_cases}
        if any(existing.get(key) != value for key, value in expected_fields.items()):
            raise ValueError("Output directory contains different settings, inputs or code; use a new --output_dir. No cache or results changed.")
    reused = load_reusable_cache(cases, opt.cache_source, opt)
    prepared = [reused[case["case_id"]] if case["case_id"] in reused else
                prepare_case(case, output_dir / "cache" / f"case_{i:03d}", opt)
                for i, case in enumerate(tqdm(cases, desc="Validating and caching included examinations"))]
    evaluation_windows = [get_evaluation_window(case, metadata, opt)
                          for case, metadata in tqdm(list(zip(cases, prepared)), desc="Validating evaluation region")]
    prepared = [{**metadata, **window} for metadata, window in zip(prepared, evaluation_windows)]
    signature = {"options": settings, "cases": cases, "splits": splits, "code_sha256": code_hash,
                 "training_protocol": "three_labeled_slices", "validation_protocol": "filename_pid_patient_grouped_v1",
                 "excluded_cases": excluded_cases, "evaluation_windows": evaluation_windows,
                 "primary_statistics": "equal_patient_weight_mean_of_examination_metrics"}
    run_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if config_path.exists():
        if read_json(config_path)["run_id"] != run_id:
            raise ValueError("Settings, input files or code changed. Use a new --output_dir to avoid mixing experiments.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_path, {**signature, "run_id": run_id, "versions": {
            "torch": torch.__version__, "numpy": np.__version__, "nibabel": nib.__version__,
            "monai": version("monai"), "scipy": version("scipy")}})
    write_csv(output_dir / "included_cases.csv", cases)
    write_csv(output_dir / "excluded_cases.csv", excluded_cases)
    write_csv(output_dir / "fold_assignments.csv", assignment_rows)
    write_csv(output_dir / "data_qc.csv", [{key: value for key, value in case.items()
              if key not in {"cache_dir", "training_rows", "evaluation_rows", "evaluation_slice_indices"}} for case in prepared])
    write_csv(output_dir / "windows.csv", [{"fold": assignment[i], **{key: value for key, value in window.items()
              if key not in {"evaluation_rows", "evaluation_slice_indices"}}} for i, window in enumerate(evaluation_windows)])
    if opt.dry_run:
        print(f"Data and {opt.evaluation_region} checks passed for {len(cases)} examinations / {opt.expected_patients} patients. "
              f"Splits and QC saved to {output_dir}. No training performed.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if opt.device == "auto" else torch.device(opt.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selected = set(opt.folds or range(1, opt.n_splits + 1))
    for split in splits:
        fold = split["fold"]
        if fold not in selected:
            continue
        fold_dir = output_dir / f"fold_{fold:02d}"
        if (fold_dir / "results.json").exists():
            validate_fold_result(read_json(fold_dir / "results.json"), read_json(config_path), split)
            print(f"Fold {fold}: reusing completed results")
            continue
        train_cases = [prepared[i] for i in split["train_indices"]]
        test_cases = [prepared[i] for i in split["test_indices"]]
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_json(fold_dir / "fold_config.json", {**split, "run_id": run_id, "seed": opt.seed + fold,
                   "device": str(device), "amp_enabled": bool(opt.use_amp and device.type == "cuda")})
        model = train_fold(train_cases, fold_dir, fold, opt, device)
        case_rows, slice_rows = evaluate_fold(model, test_cases, fold, opt, device)
        write_csv(fold_dir / "case_metrics.csv", case_rows)
        write_csv(fold_dir / "slice_metrics.csv", slice_rows)
        write_json(fold_dir / "results.json", {"run_id": run_id, "case_metrics": case_rows, "slice_metrics": slice_rows})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        aggregate_results(output_dir)
    aggregate_results(output_dir)
    print(f"Elapsed: {(time.time() - start) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
