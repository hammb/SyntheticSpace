import nibabel as nib
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

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
        rand_slice = 100

        space_slice = input_image.get_fdata()[:, rand_slice, :] / 2351.892578125
        mprage_slice = output_image.get_fdata()[:, rand_slice, :] / 1033.0

        space_slice = cv2.resize(space_slice, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        mprage_slice = cv2.resize(mprage_slice, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        space_slice = np.expand_dims(space_slice, 0).astype(np.float32)
        mprage_slice = np.expand_dims(mprage_slice, 0).astype(np.float32)

        #space_slice = np.einsum('ijk->kji', space_slice)
        #mprage_slice = np.einsum('ijk->kji', mprage_slice)

        return space_slice, mprage_slice
