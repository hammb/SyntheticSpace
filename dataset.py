import nibabel as nib
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ScratchDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_samples = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_samples)

    def __getitem__(self, index):
        sample = self.list_samples[index]

        input_image_path = os.path.join(self.root_dir, sample, "mprage.nii.gz")
        output_image_path = os.path.join(self.root_dir, sample, "space.nii.gz")

        input_image = nib.load(input_image_path)
        output_image = nib.load(output_image_path)

        rand_slice = int(torch.rand(1) * input_image.shape[1])

        space_slice = input_image.get_fdata()[:, rand_slice, :] / 2351.892578125
        mprage_slice = output_image.get_fdata()[:, rand_slice, :] / 1033.0

        return space_slice, mprage_slice
