import os

import torch
from utils import save_checkpoint, load_checkpoint, evaluate
import torch.nn as nn
import torch.optim as optim
import config
from space_rt import SpaceRt
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from vgg_loss import VGGLoss

from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True


def pre_train_fn(
        disc, gen, loader, opt_disc, opt_gen, bce, VGG_Loss, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):

        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        # Train generator
        with torch.cuda.amp.autocast():
            # D_fake = disc(x, y_fake)
            # G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            # L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            # G_loss = G_fake_loss + L1
            G_loss = VGG_Loss(y_fake.expand(1, 3, 256, 256),
                              y.expand(1, 3, 256, 256))  # Expand single value to RGB

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def train_fn(
        disc, gen, loader, opt_disc, opt_gen, bce, VGG_Loss, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for batch_idx, (x, y) in enumerate(loop):

        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        for idx in range(x.shape[1]):
            x_slice = x[:, idx, :, :].unsqueeze(1)
            y_slice = y[:, idx, :, :].unsqueeze(1)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = gen(x_slice)
                D_real = disc(x_slice, y_slice)
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake = disc(x_slice, y_fake.detach())
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
            # Train generator
            with torch.cuda.amp.autocast():
                # D_fake = disc(x, y_fake)
                # G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
                # L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
                # G_loss = G_fake_loss + L1
                G_loss = VGG_Loss(y_fake.expand(1, 3, 256, 256),
                                  y_slice.expand(1, 3, 256, 256))  # Expand single value to RGB

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

        if batch_idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


disc = Discriminator(in_channels=1).to(config.DEVICE)
gen = Generator(in_channels=1, features=64).to(config.DEVICE)

opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

BCE = nn.BCEWithLogitsLoss()
# L1_LOSS = nn.L1Loss()
VGG_Loss = VGGLoss()

if config.LOAD_MODEL:
    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    )

fold = 1
rand_slices = False
percentage_val_samples = 15
rand_spacing = True

pre_train_dataset = SpaceRt(root_dir=config.TRAIN_DIR, fold=fold, rand_slices=True, val=False,
                            percentage_val_samples=percentage_val_samples,
                            rand_spacing=rand_spacing, augmentation=False, input_sequence=config.INPUT_SEQUENCE)

pre_train_loader = DataLoader(
    pre_train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
)

train_dataset = SpaceRt(root_dir=config.TRAIN_DIR, fold=fold, rand_slices=rand_slices, val=False,
                        percentage_val_samples=percentage_val_samples,
                        rand_spacing=rand_spacing, augmentation=True, input_sequence=config.INPUT_SEQUENCE)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
)
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

val_dataset = SpaceRt(root_dir=config.TRAIN_DIR, fold=fold, rand_slices=rand_slices, val=True,
                      percentage_val_samples=percentage_val_samples,
                      rand_spacing=rand_spacing, augmentation=False, input_sequence=config.INPUT_SEQUENCE)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

os.makedirs("fold_" + str(fold), exist_ok=True)

for epoch in range(config.NUM_EPOCHS):
    pre_train_fn(
        disc, gen, pre_train_loader, opt_disc, opt_gen, BCE, VGG_Loss, g_scaler, d_scaler,
    )

    evaluate(gen, disc, val_loader, epoch, VGG_Loss, opt_disc, opt_gen, fold)

for epoch in range(config.NUM_EPOCHS, config.NUM_EPOCHS * 2):
    train_fn(
        disc, gen, train_loader, opt_disc, opt_gen, BCE, VGG_Loss, g_scaler, d_scaler,
    )

    evaluate(gen, disc, val_loader, epoch, VGG_Loss, opt_disc, opt_gen, fold)
