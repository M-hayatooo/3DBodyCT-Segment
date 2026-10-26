# BodyCT Deep — 3DBodyCT-Segment

CT body composition segmentation using PyTorch. A 2D U-Net takes three adjacent axial CT slices as input and predicts the segmentation of the center slice (a 2.5D approach).

This guide describes how to train a model on your own data and apply the trained model to unseen CT volumes. Datasets and pretrained weights are not included.

## Setup

Run the following commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy scipy nibabel monai tqdm
```

Install a PyTorch build appropriate for your CPU or CUDA environment. The training code requires the `torch.amp.GradScaler` and `torch.amp.autocast` APIs. Dependency versions are not pinned in this repository.

Training automatically uses CUDA when available and otherwise uses CPU. Mixed precision is enabled only on CUDA.

## Training data

Prepare CT images and manual segmentation masks as NIfTI files. For the default `Fat` task, use the following layout:

```text
/path/to/Fat/
├── voxels/
│   ├── case001_voxel.nii.gz
│   └── case002_voxel.nii.gz
└── labels/
    ├── case001_Fat.nii.gz
    └── case002_Fat.nii.gz
```

Images and masks must have matching case IDs and corresponding geometry. Nested directories are supported when their relative paths match between `voxels/` and `labels/`. CT intensities should be in Hounsfield units (HU), and masks must contain integer class values.

The default fat labels are:

| Value | Class |
| --- | --- |
| 0 | Background |
| 1 | Subcutaneous adipose tissue (SAT) |
| 2 | Visceral adipose tissue (VAT) |

Supported training modes are:

| Mode | Number of classes, including background | Label filename |
| --- | --- | --- |
| `Fat` | 3 | `<case_id>_Fat.nii.gz` |
| `Whole` | 3 | `<case_id>_Whole.nii.gz` |
| `Muscle` | 7 | `<case_id>_Muscle.nii.gz` |
| `Muscle2` | 11 | `<case_id>_label.nii.gz` |

For modes other than `Fat`, define a consistent class mapping in your annotations. Class values must range from zero to the number of classes minus one.

Keep test patients separate from training patients, including any repeat examinations of the same patient. `train.py` uses every matched examination it finds for training; it does not create a test split.

## Training

Set the `data_root` assignment in `train.py` to your dataset directory (for example, `/path/to/Fat`) before running training. The current code uses `/home/toru/BodyDATA/<mode>`, with `Muscle_2` as the directory name for `Muscle2`.

Train a fat segmentation model:

```bash
python train.py \
  --mode Fat \
  --day my_run \
  --epochs 1000 \
  --batch_size 4 \
  --lr 0.0001 \
  --in_plane_spacing 1.6 \
  --seed 42
```

Use `python train.py --help` to inspect all options. `--day` is the output run identifier and can be a date or a custom name. To disable mixed precision, add `--use_amp false`.

Training converts images to canonical orientation and resamples them to a 256 × 256 grid with the chosen in-plane spacing while retaining the source slice spacing. At 1.6 mm, the in-plane field of view is 409.6 × 409.6 mm. Inspect your data to ensure this field of view covers the target anatomy: preprocessing can crop foreground outside the grid.

For `Fat`, `Muscle`, and `Muscle2`, intensities are clipped to −500 to 500 HU and normalized to [−1, 1]. `Whole` uses −2000 to 150 HU. Images use linear interpolation; masks use nearest-neighbor interpolation.

Each epoch samples one three-slice stack per eligible training examination. All three slices must contain foreground annotations; examinations without such a stack are skipped. Training applies spatial and intensity augmentation and uses AdamW, cosine learning-rate scheduling, and a combined Dice/cross-entropy loss.

For the example above, outputs are saved under:

```text
weights/my_run_Fat/
├── params_Fat.json
├── loss_Fat.txt
└── model/
    ├── UNet_epoch1000.pth
    └── UNet.pth
```

`UNet.pth` contains the final model state dictionary. Additional checkpoints are saved every 1,000 epochs. The training script starts a new model each time; it does not resume from an existing checkpoint.

## Testing on an unseen CT volume

Use `test.py` to segment a CT volume with a trained model. Manual labels are not required for inference.

```bash
python test.py \
  --image /path/to/test_case_voxel.nii.gz \
  --checkpoint weights/my_run_Fat/model/UNet.pth \
  --output outputs/test_case_prediction.nii.gz \
  --mode Fat \
  --in_plane_spacing 1.6
```

The script applies the same resampling, normalization, and slice ordering as training. Use the same `--mode` and `--in_plane_spacing` as your training run. CUDA is selected automatically when available; use `--device cpu` or `--device cuda:0` to choose explicitly. Run `python test.py --help` for all options.

The output uses the **resampled 256 × 256 grid**, rather than the original CT grid. For `Fat`, values are 0 (background), 1 (SAT), and 2 (VAT). If an original-grid mask is needed, resample the prediction to the original CT geometry using nearest-neighbor interpolation.

## Evaluating predictions

Add `--label` to compute and print per-class Dice against a complete manual reference mask:

```bash
python test.py \
  --image /path/to/test_case_voxel.nii.gz \
  --checkpoint weights/my_run_Fat/model/UNet.pth \
  --output outputs/test_case_prediction.nii.gz \
  --mode Fat \
  --in_plane_spacing 1.6 \
  --label /path/to/test_case_Fat.nii.gz
```

The reference mask is resampled to the prediction grid with nearest-neighbor interpolation. Metrics cover only that field of view; anatomy cropped during preprocessing is not evaluated. A class absent from both masks is reported as `NaN`.

Use independently annotated test patients that were not used for training or parameter selection. The evaluation assumes complete reference annotations over the prediction field of view. Partially annotated volumes require a separate evaluation restricted to annotated slices or regions. Inspect predictions alongside CT images and report per-class results across the held-out test set.

## Local files

The `.gitignore` excludes datasets, medical images, model checkpoints, generated outputs, Python caches, local environments, credentials, and machine-specific settings. Notebooks are excluded by default because saved outputs may contain patient information or embedded images.
