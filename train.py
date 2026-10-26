import argparse
import glob
import json
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
from monai.losses import DiceCELoss
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils.dataset import BodyDataset
from utils.functions import fix_seed, seed_worker
from utils.load import load_images
from utils.network import UNet
from utils.trainer import trainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def create_parser():
    current_date = datetime.now().strftime("%Y%m%d")
    now_time = datetime.now().strftime("%H:%M")
    print(f"{current_date} {now_time} ...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=str, default=current_date)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=("Muscle", "Muscle2", "Fat", "Whole"),
        default="Fat",
    )
    parser.add_argument("--in_plane_spacing", type=float, default=1.6)
    parser.add_argument("--intensity_aug_prob", type=float, default=0.5)
    parser.add_argument("--intensity_shift_hu", type=float, default=20.0)
    parser.add_argument("--intensity_scale_min", type=float, default=0.9)
    parser.add_argument("--intensity_scale_max", type=float, default=1.1)
    parser.add_argument("--gaussian_noise_prob", type=float, default=0.5)
    parser.add_argument("--gaussian_noise_std_min", type=float, default=0.02)
    parser.add_argument("--gaussian_noise_std_max", type=float, default=0.12)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    print("BodyCT.  Training start:", end=" ")
    start_time = time.time()
    opt = create_parser()
    fix_seed(opt.seed)

    data_mode = "Muscle_2" if opt.mode == "Muscle2" else opt.mode
    label_pattern = "*_label.nii.gz" if opt.mode == "Muscle2" else f"*_{opt.mode}.nii.gz"
    data_root = f"/home/toru/BodyDATA/{data_mode}"
    found_vpathes = sorted(glob.glob(f"{data_root}/**/*_voxel.nii.gz", recursive=True))  # [:10]
    found_lpathes = sorted(glob.glob(f"{data_root}/**/{label_pattern}", recursive=True))  # [:10]

    pairs = []
    missing_labels = []
    label_suffix = "_label.nii.gz" if opt.mode == "Muscle2" else f"_{opt.mode}.nii.gz"
    for vpath in found_vpathes:
        lpath = vpath.replace("/voxels/", "/labels/").replace("_voxel.nii.gz", label_suffix)
        if os.path.exists(lpath):
            pairs.append((vpath, lpath))
        else:
            missing_labels.append(vpath)

    paired_label_set = {lpath for _, lpath in pairs}
    extra_labels = [lpath for lpath in found_lpathes if lpath not in paired_label_set]
    if missing_labels or extra_labels:
        print(
            "Skipped unmatched files: "
            f"voxels_without_label={len(missing_labels)}, labels_without_voxel={len(extra_labels)}"
        )

    vpathes = [vpath for vpath, _ in pairs]
    lpathes = [lpath for _, lpath in pairs]
    print(f"found voxels: {len(found_vpathes)}")
    print(f"found labels: {len(found_lpathes)}")
    print(f"matched pairs: {len(pairs)}")
    if len(vpathes) == 0 or len(lpathes) == 0:
        raise RuntimeError(f"No training case was found: mode={opt.mode}, data_root={data_root}")
    if len(vpathes) != len(lpathes):
        raise RuntimeError(f"Number of voxels and labels does not match: {len(vpathes)} vs {len(lpathes)}")
    print(f"train cases: {len(lpathes)}")

    BASE = f"weights/{opt.day}_{opt.mode}"
    PATH_MODEL = os.path.join(BASE, "model")
    os.makedirs(PATH_MODEL, exist_ok=True)
    with open(os.path.join(BASE, f"params_{opt.mode}.json"), mode="w") as f:
        json.dump(opt.__dict__, f, indent=4)

    voxels, labels = load_images(
        vpathes,
        lpathes,
        in_plane_spacing=opt.in_plane_spacing,
    )
    print(voxels[0].max(), voxels[0].min())
    print(labels[0].max(), labels[0].min())

    dataset = BodyDataset(
        voxels,
        labels,
        opt.mode,
        is_train=True,
        gaussian_noise_prob=opt.gaussian_noise_prob,
        gaussian_noise_std_range=(opt.gaussian_noise_std_min, opt.gaussian_noise_std_max),
        intensity_aug_prob=opt.intensity_aug_prob,
        intensity_shift_hu=opt.intensity_shift_hu,
        intensity_scale_range=(opt.intensity_scale_min, opt.intensity_scale_max),
    )
    if len(dataset) == 0:
        raise RuntimeError("No 3-slice stack with labels on all three slices was found.")

    num_workers = 4
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.use_amp = bool(opt.use_amp and device.type == "cuda")

    if opt.mode == "Muscle":
        num_classes = 7
    elif opt.mode == "Muscle2":
        num_classes = 11
    elif opt.mode == "Whole" or opt.mode == "Fat":
        num_classes = 3
    unet = UNet(3, num_classes)
    unet.to(device)

    optimizer = optim.AdamW(
        unet.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * 0.1)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True).to(device)
    scaler = GradScaler("cuda", enabled=opt.use_amp)
    trainer(
        opt,
        BASE,
        PATH_MODEL,
        dataloader,
        unet,
        optimizer,
        criterion,
        scheduler,
        scaler,
        device,
    )

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600
    print(f"処理にかかった時間: {elapsed_time:.1f} hours")
    return


if __name__ == "__main__":
    main()
    print("Finish training.")
