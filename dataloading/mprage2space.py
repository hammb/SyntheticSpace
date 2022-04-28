import nibabel as nib
import torch
import os
from torch.utils.data import Dataset
from torch.nn.functional import pad

import config
from config import RAND_SAMPLE_SIZE
import random
from augmentation.data_augmentation import DefaultDataAugmentation


class Mprage2space(Dataset):
    def __init__(self, root_dir, fold=None, rand_slices=None, val=False, percentage_val_samples=15, augmentation=False):

        if fold is None:
            raise ValueError("No fold given")

        if not fold == "all":
            if not isinstance(fold, int):
                raise ValueError("Fold must be int or 'all'")

        random.seed(fold)

        self.root_dir = root_dir
        self.all_samples = os.listdir(self.root_dir)

        # 15% val. data

        num_val_samples = int(len(self.all_samples) / 100 * percentage_val_samples)
        self.val_samples = random.sample(self.all_samples, num_val_samples)

        self.train_samples = list(filter(lambda sample: sample not in self.val_samples, self.all_samples))
        self.rand_slices = rand_slices
        self.augmentation = augmentation

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

        input_image_path = os.path.join(self.root_dir, sample, "mprage_0.nii.gz")
        output_image_path = os.path.join(self.root_dir, sample, "space_0.nii.gz")

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

        output_image_max_value = config.MAX_VALUE_SPACE
        input_image_max_value = config.MAX_VALUE_MPRAGE

        # normalize
        input_image_torch = input_image_torch / input_image_max_value  # / 5154.285
        output_image_torch = output_image_torch / output_image_max_value  # / 1435.0

        input_image_torch = torch.einsum('ijk->kji', input_image_torch)
        output_image_torch = torch.einsum('ijk->kji', output_image_torch)

        # pick random slice
        if self.rand_slices:
            indices = torch.randperm(output_image_torch.shape[0])[:RAND_SAMPLE_SIZE]
            return input_image_torch[indices, :, :].float(), output_image_torch[indices, :, :].float()

        return input_image_torch.float(), output_image_torch.float()

