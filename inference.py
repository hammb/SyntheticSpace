import os.path

import numpy as np
import torch
from utils import load_checkpoint
import torch.optim as optim
import config
from dataset import ScratchDataset
from generator_model import Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk

gen = Generator(in_channels=1, features=64).to(config.DEVICE)
gen.eval()
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

load_checkpoint(
    config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
)

test_dataset = ScratchDataset(root_dir=config.TEST_DIR, rand_slices=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

loop = tqdm(test_loader, leave=True)


def safe_image(head, rid):
    reference_image = sitk.ReadImage(os.path.join(config.TEST_DIR, rid, "mprage.nii.gz"))
    shape = reference_image.GetSize()
    head = head * 1435.0

    y_crop = (256 - shape[0]) // 2
    x_crop = (256 - shape[2]) // 2

    head = head[:, x_crop:-x_crop, y_crop:-y_crop]

    if not np.shape(head)[1] == shape[2]:
        head = head[:, 1:, :]

    if not np.shape(head)[2] == shape[0]:
        head = head[:, :, 1:]

    head = np.einsum('ijk->jik', head)
    head = sitk.GetImageFromArray(head)
    head.SetOrigin(reference_image.GetOrigin())
    head.SetDirection(reference_image.GetDirection())
    head.SetSpacing(reference_image.GetSpacing())

    os.makedirs(os.path.join(config.PRED_DIR, rid), exist_ok=True)

    sitk.WriteImage(head, os.path.join(config.PRED_DIR, rid, "fake.nii.gz"))


with torch.no_grad():
    for idx, (x, y) in enumerate(loop):
        for slice_idx in range(x.shape[1]):

            x_slice = x[:, slice_idx, :, :][None, :, :, :].to(config.DEVICE)
            y_fake = gen(x_slice)

            head = y_fake.cpu().detach().numpy()[0] if slice_idx == 0 else np.append(head, y_fake.cpu().detach().numpy()[0],
                                                                                  axis=0)
        safe_image(head, test_dataset.list_samples[idx])

