"""Segment an unseen CT volume and optionally evaluate a complete reference mask."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from tqdm import tqdm

from utils.load import DEFAULT_IN_PLANE_SPACING, conform_image
from utils.network import UNet


NUM_CLASSES = {"Fat": 3, "Whole": 3, "Muscle": 7, "Muscle2": 11}


def create_parser(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input CT NIfTI file (HU)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained UNet state dictionary")
    parser.add_argument("--output", type=Path, required=True, help="Prediction NIfTI file on the resampled grid")
    parser.add_argument("--mode", choices=tuple(NUM_CLASSES), default="Fat")
    parser.add_argument("--in_plane_spacing", type=float, default=DEFAULT_IN_PLANE_SPACING)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--label", type=Path, help="Optional complete reference mask for per-class Dice")
    options = parser.parse_args(argv)
    if options.in_plane_spacing <= 0:
        parser.error("--in_plane_spacing must be positive")
    for name in ("image", "checkpoint", "label"):
        path = getattr(options, name)
        if path is not None and not path.is_file():
            parser.error(f"--{name}: file not found: {path}")
    if options.output.resolve() in {
        path.resolve() for path in (options.image, options.checkpoint, options.label)
        if path is not None
    }:
        parser.error("--output must differ from the input files")
    return options


def load_volume(path):
    image = nib.squeeze_image(nib.as_closest_canonical(nib.load(path)))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D volume: {path}, shape={image.shape}")
    return image


def prepare_image(path, mode, spacing):
    image = conform_image(load_volume(path), in_plane_spacing=spacing, image_size=256)
    volume = image.get_fdata(dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("CT contains non-finite intensities")
    low, high = (-2000, 150) if mode == "Whole" else (-500, 500)
    volume = 2.0 * (np.clip(volume, low, high) - low) / (high - low) - 1.0
    return image, volume.transpose(2, 1, 0)


def predict(model, axial, device):
    prediction = np.zeros(axial.shape, dtype=np.uint8)
    padding = np.full(axial.shape[1:], -1.0, dtype=np.float32)
    with torch.inference_mode():
        for z in tqdm(range(len(axial)), desc="Segmenting"):
            stack = np.stack([
                axial[index] if 0 <= index < len(axial) else padding
                for index in (z - 1, z, z + 1)
            ])
            inputs = torch.from_numpy(stack[None]).to(device)
            prediction[z] = model(inputs).argmax(dim=1)[0].cpu().numpy()
    return prediction.transpose(2, 1, 0)


def evaluate(prediction, image, label_path, mode):
    reference = load_volume(label_path)
    values = reference.get_fdata()
    if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
            or values.min() < 0 or values.max() >= NUM_CLASSES[mode]):
        raise ValueError("Reference mask must contain valid integer class values")
    # Metrics cover only the resampled prediction field of view.
    reference = resample_from_to(reference, image, order=0)
    truth = np.asarray(reference.dataobj, dtype=np.int16)
    for value in range(1, NUM_CLASSES[mode]):
        expected = truth == value
        actual = prediction == value
        denominator = expected.sum() + actual.sum()
        dice = (2.0 * np.count_nonzero(expected & actual) / denominator
                if denominator else float("nan"))
        name = ("SAT", "VAT")[value - 1] if mode == "Fat" else f"label_{value}"
        print(f"{name} Dice: {dice:.4f}")


def main(argv=None):
    options = create_parser(argv)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if options.device == "auto" else torch.device(options.device)
    image, axial = prepare_image(options.image, options.mode, options.in_plane_spacing)
    model = UNet(3, NUM_CLASSES[options.mode]).to(device)
    state = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    prediction = predict(model, axial, device)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prediction, image.affine, header), options.output)
    print(f"Saved: {options.output}")
    if options.label is not None:
        evaluate(prediction, image, options.label, options.mode)


if __name__ == "__main__":
    main()
