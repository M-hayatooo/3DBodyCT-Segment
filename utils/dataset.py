import random

import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
from tqdm import tqdm


class BodyDataset(Dataset):
    def __init__(
        self,
        voxels,
        labels,
        mode,
        is_train=True,
        gaussian_noise_prob=0.0,
        gaussian_noise_std_range=(0.02, 0.08),
        intensity_aug_prob=0.0,
        intensity_shift_hu=20.0,
        intensity_scale_range=(0.9, 1.1),
    ):
        self.mode = mode
        self.is_train = is_train
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std_range = gaussian_noise_std_range
        self.intensity_aug_prob = intensity_aug_prob
        self.intensity_shift_hu = intensity_shift_hu
        self.intensity_scale_range = intensity_scale_range
        self.labels = []
        self.voxels = []
        self.slice_indices = []
        skipped_volumes = 0
        for voxel, label in tqdm(zip(voxels, labels), total=len(voxels)):
            label = label.transpose(2, 1, 0).astype(np.int16, copy=False)
            slice_indices = self.get_slice_indices(label)
            if slice_indices.size == 0:
                skipped_volumes += 1
                continue

            voxel = self.preprocessing(voxel)
            self.voxels.append(voxel.transpose(2, 1, 0))
            self.labels.append(label)
            self.slice_indices.append(slice_indices)

        eligible = sum(len(indices) for indices in self.slice_indices)
        print(f"eligible 3-slice stacks: {eligible}")
        print(f"skipped volumes without 3 consecutive labeled slices: {skipped_volumes}")

    def __len__(self):
        return len(self.voxels)

    def _normalize(self, voxel, low, high):
        if low is None or high is None:
            low = float(np.min(voxel))
            high = float(np.max(voxel))
        if high <= low:
            return np.zeros_like(voxel, dtype=np.float32)
        voxel = 2.0 * (voxel - low) / (high - low) - 1.0
        return voxel.astype(np.float32, copy=False)

    def _get_intensity_range(self):
        if self.mode in ("Fat", "Muscle", "Muscle2"):
            return -500, 500
        if self.mode == "Born":
            return 100, None
        if self.mode == "Whole":
            return -2000, 150
        return None, None

    def preprocessing(self, voxel):
        low, high = self._get_intensity_range()
        if low is not None:
            if high is None:
                high = float(np.max(voxel))
            voxel = np.clip(voxel, low, high)

        voxel = self._normalize(voxel, low, high)
        return voxel

    def add_intensity_augmentation(self, image):
        if not self.is_train or random.random() >= self.intensity_aug_prob:
            return image

        scale_min, scale_max = self.intensity_scale_range
        scale = np.random.uniform(scale_min, scale_max)
        low, high = self._get_intensity_range()
        if low is None or high is None:
            shift = 0.0
        else:
            shift_hu = np.random.uniform(-self.intensity_shift_hu, self.intensity_shift_hu)
            shift = 2.0 * shift_hu / (high - low)
        return np.clip(image * scale + shift, -1, 1)

    def add_gaussian_noise(self, image):
        if not self.is_train or random.random() >= self.gaussian_noise_prob:
            return image

        std_min, std_max = self.gaussian_noise_std_range
        noise_std = np.random.uniform(std_min, std_max)
        noise = np.random.normal(0, noise_std, image.shape).astype(np.float32, copy=False)
        return np.clip(image + noise, -1, 1)

    def random_stride(self, image, label, slide=10):
        target_h, target_w = label.shape
        image = np.pad(
            image,
            [(slide, slide), (slide, slide), (0, 0)],
            "constant",
            constant_values=image.min(),
        )
        label = np.pad(label, [(slide, slide), (slide, slide)], "constant", constant_values=0)
        h, w, _ = image.shape
        x_l = np.random.randint(0, w - target_w)
        x_r = x_l + target_w
        y_l = np.random.randint(0, h - target_h)
        y_r = y_l + target_h
        image = image[y_l:y_r, x_l:x_r, :]
        label = label[y_l:y_r, x_l:x_r]
        return image, label

    def random_rotate_2d(self, image, label, angle_range=10):
        angle_deg = np.random.randint(-angle_range, angle_range)
        image = ndimage.rotate(image, angle_deg, mode="nearest", reshape=False, order=1, axes=(0, 1))
        label = ndimage.rotate(label, angle_deg, mode="nearest", reshape=False, order=0)
        return image, label

    @staticmethod
    def get_slice_indices(label):
        num_slices = label.shape[0]
        if num_slices == 0:
            return np.empty(0, dtype=np.int32)

        flat = label.reshape(num_slices, -1)
        has_label = flat.max(axis=1) > 0

        # Use a center position only when all three input slices contain labels.
        eligible = np.zeros(num_slices, dtype=bool)
        if num_slices >= 3:
            eligible[1:-1] = has_label[:-2] & has_label[1:-1] & has_label[2:]
        return np.flatnonzero(eligible).astype(np.int32, copy=False)

    def select_slice_index(self, slice_indices):
        if self.is_train:
            return int(np.random.choice(slice_indices))
        return int(slice_indices[len(slice_indices) // 2])

    def get_image_stack(self, voxel, index):
        slices = []
        for slice_index in (index - 1, index, index + 1):
            if 0 <= slice_index < voxel.shape[0]:
                slices.append(voxel[slice_index])
            else:
                slices.append(np.full(voxel.shape[1:], -1, dtype=voxel.dtype))
        return np.stack(slices).transpose(1, 2, 0)

    def __getitem__(self, idx):
        voxel = self.voxels[idx]
        label = self.labels[idx]
        slice_indices = self.slice_indices[idx]

        index = self.select_slice_index(slice_indices)
        image = self.get_image_stack(voxel, index)
        image = self.add_intensity_augmentation(image)
        image = self.add_gaussian_noise(image)
        slabel = label[index]

        if self.is_train and random.random() < 0.8:
            image, slabel = self.random_stride(image, slabel)
        if self.is_train and random.random() < 0.8:
            image, slabel = self.random_rotate_2d(image, slabel)

        image = image.astype(np.float32, copy=False)
        slabel = slabel.astype(np.int64, copy=False)
        return image.transpose(2, 0, 1), slabel[None]
