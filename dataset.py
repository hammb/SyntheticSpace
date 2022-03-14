import nibabel as nib
import torch
import os
from torch.utils.data import Dataset
from torch.nn.functional import pad
from config import BATCH_SIZE


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

        output_image = output_image.get_fdata()
        output_image_torch = torch.tensor(output_image)
        output_image_torch = output_image_torch / 1435.0

        output_image_torch = pad_image(output_image_torch)

        input_image = input_image.get_fdata()
        input_image_torch = torch.tensor(input_image)
        input_image_torch = input_image_torch / 5154.285

        input_image_torch = pad_image(input_image_torch)

        indices = torch.randperm(output_image_torch.shape[0])[:BATCH_SIZE]

        return output_image_torch[indices,:,:].float(), input_image_torch[indices,:,:].float()
