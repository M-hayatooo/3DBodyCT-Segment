from functools import partial

import nibabel as nib
import numpy as np
from nibabel import processing
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)
DEFAULT_IN_PLANE_SPACING = 1.6
DEFAULT_IMAGE_SIZE = 256


def conform_image(
    image,
    in_plane_spacing=DEFAULT_IN_PLANE_SPACING,
    image_size=DEFAULT_IMAGE_SIZE,
    is_label=False,
):
    """Shared CT/label resampling for training and cross-validation."""
    if image_size <= 0 or image_size % 16:
        raise ValueError("image_size must be a positive multiple of 16 for UNet")
    if in_plane_spacing <= 0:
        raise ValueError("in_plane_spacing must be positive")
    return processing.conform(
        image,
        out_shape=(image_size, image_size, image.shape[2]),
        voxel_size=(in_plane_spacing, in_plane_spacing, image.header.get_zooms()[2]),
        order=0 if is_label else 1,
        cval=0 if is_label else -2000,
    )


def load_images(
    vpathes,
    lpathes,
    in_plane_spacing=DEFAULT_IN_PLANE_SPACING,
    image_size=DEFAULT_IMAGE_SIZE,
):
    voxels = []
    labels = []
    for vpath, lpath in tqdm(zip(vpathes, lpathes)):
        vd = nib.squeeze_image(nib.as_closest_canonical(nib.load(vpath)))
        ld = nib.squeeze_image(nib.as_closest_canonical(nib.load(lpath)))

        vd = conform_image(vd, in_plane_spacing, image_size)
        ld = conform_image(ld, in_plane_spacing, image_size, is_label=True)
        voxels.append(vd.get_fdata().astype("float32"))
        labels.append(ld.get_fdata().astype("int16"))
    return voxels, labels
