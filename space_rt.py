import nibabel as nib
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.nn.functional import pad

import config
from config import BATCH_SIZE
import random
from data_augmentation import DefaultDataAugmentation


def pad_image(output_image_torch):
    output_image_torch = torch.einsum('ijk->jki', output_image_torch)

    y_shape = output_image_torch.shape[1]
    x_shape = output_image_torch.shape[2]

    y_pad = (256 - y_shape) // 2
    x_pad = (256 - x_shape) // 2

    output_image_torch_padded = pad(output_image_torch, (x_pad, x_pad, y_pad, y_pad), mode="constant", value=0)

    if output_image_torch_padded.shape[2] != 256:
        output_image_torch_padded = pad(output_image_torch_padded, (1, 0), mode="constant", value=0)

    if output_image_torch_padded.shape[1] != 256:
        output_image_torch_padded = pad(output_image_torch_padded, (0, 0, 1, 0), mode="constant", value=0)

    return output_image_torch_padded


class SpaceRt(Dataset):
    def __init__(self, root_dir, fold=None, rand_slices=None, val=False, percentage_val_samples=15, spacing=None,
                 rand_spacing=False, augmentation=False, input_sequence=None):

        if input_sequence is None:
            raise ValueError("No input sequence given")

        if input_sequence not in ["space", "mprage"]:
            raise ValueError("input sequence must be 'space' or 'mprage'")

        if fold is None:
            raise ValueError("No fold given")

        if not fold == "all":
            if not isinstance(fold, int):
                raise ValueError("Fold must be int or 'all'")

        random.seed(fold)

        self.root_dir = root_dir
        self.all_samples = os.listdir(self.root_dir)

        # 15% val. data

        num_val_samples = len(self.all_samples) // 100 * percentage_val_samples
        self.val_samples = random.sample(self.all_samples, num_val_samples)

        self.train_samples = list(filter(lambda sample: sample not in self.val_samples, self.all_samples))
        self.rand_slices = rand_slices
        self.rand_spacing = rand_spacing
        self.spacing = spacing
        self.augmentation = augmentation
        self.input_sequence = input_sequence

        if self.spacing is None and not self.rand_spacing:
            raise ValueError("Ether random spacing must be active or define spacing (0: space, 1: mprage)")

        if self.spacing is not None and self.rand_spacing:
            raise ValueError(
                "Ether random spacing is active or you define the spacing (0: space, 1: mprage). Not both!")

        if not self.rand_spacing:
            if self.spacing not in [0, 1]:
                raise ValueError(
                    "Wrong spacing given. Must be 0: space or 1: mprage")

        if val:
            self.list_samples = self.val_samples
        else:
            self.list_samples = self.train_samples

        if fold == "all":
            self.list_samples = self.all_samples

    def __len__(self):
        return len(self.list_samples)

    def __getitem__(self, index):

        sample = self.list_samples[index]

        if self.input_sequence == "space":
            input_sequence = "space"
            output_sequence = "mprage"
        else:
            output_sequence = "space"
            input_sequence = "mprage"

        # Get image path
        if self.rand_spacing:
            spacing = random.randint(0, 1)

            input_image_path = os.path.join(self.root_dir, sample, input_sequence + "_" + str(spacing) + ".nii.gz")
            output_image_path = os.path.join(self.root_dir, sample, output_sequence + "_" + str(spacing) + ".nii.gz")
        else:
            input_image_path = os.path.join(self.root_dir, sample, input_sequence + "_" + str(self.spacing) + ".nii.gz")
            output_image_path = os.path.join(self.root_dir, sample, output_sequence + "_" + str(self.spacing) + ".nii.gz")

        # Load image
        input_image = nib.load(input_image_path)
        output_image = nib.load(output_image_path)

        # Get tensor from image
        input_image = input_image.get_fdata()
        output_image = output_image.get_fdata()

        # data augemtation
        if self.augmentation:
            input_image, output_image = DefaultDataAugmentation.execute(input_image, output_image)

        # cast to torch tensor
        input_image_torch = torch.tensor(input_image)
        output_image_torch = torch.tensor(output_image)

        if self.input_sequence == "space":
            input_image_max_value = config.MAX_VALUE_SPACE
            output_image_max_value = config.MAX_VALUE_MPRAGE
        else:
            output_image_max_value = config.MAX_VALUE_SPACE
            input_image_max_value = config.MAX_VALUE_MPRAGE

        # normalize
        input_image_torch = input_image_torch / input_image_max_value  # / 5154.285
        output_image_torch = output_image_torch / output_image_max_value  # / 1435.0

        # padding
        input_image_torch = pad_image(input_image_torch)
        output_image_torch = pad_image(output_image_torch)

        # pick random slice
        if self.rand_slices:
            indices = torch.randperm(output_image_torch.shape[0])[:BATCH_SIZE]
            return input_image_torch[indices, :, :].float(), output_image_torch[indices, :, :].float()

        return input_image_torch.float(), output_image_torch.float()
