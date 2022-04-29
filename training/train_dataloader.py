#!/usr/bin/env python3
import os
import random
import argparse
import torch
from utils.utils_dataloader import load_checkpoint, evaluate
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from dataloading.mprage2space import Mprage2space
from architectures.generator_model import Generator
from architectures.discriminator_model import Discriminator
from tqdm import tqdm
from loss_functions.vgg_loss import VGGLoss
import json

torch.backends.cudnn.benchmark = True


def train_fn(
        disc, gen, loader, opt_disc, opt_gen, bce, VGG_Loss, g_scaler, d_scaler, epoch
):
    loop = tqdm(loader, leave=True)
    loop.set_description("Training Epoch Nr.: " + str(epoch))
    random.seed(epoch)

    for batch_idx, (x, y) in enumerate(loop):

        x = torch.reshape(x, (-1, 1, 256, 256))
        y = torch.reshape(y, (-1, 1, 256, 256))

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
                G_loss = VGG_Loss(y_fake.expand(config.RAND_SAMPLE_SIZE * config.BATCH_SIZE, 3, 256, 256),
                                  y.expand(config.RAND_SAMPLE_SIZE * config.BATCH_SIZE, 3, 256, 256))
                # Expand single value to RGB

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

        if batch_idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', default=0,
                        help='The fold number for k-fold-crossval', required=False, type=int)
    parser.add_argument('-ss', '--sample_size', default=2,
                        help='Num of sLices used per patient', required=False, type=int)
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('-p', '--path', default=None,
                        help='Path to model parameters', required=False, type=str)

    args = parser.parse_args()

    config.FOLD = args.fold
    config.RAND_SAMPLE_SIZE = args.sample_size
    config.LOAD_MODEL = args.load
    config.RESUME_TRAINING = args.resume

    if config.LOAD_MODEL:
        config.LOAD_MODEL_PATH = args.path

    # DATA -------------
    train_dataset = Mprage2space(root_dir=config.TRAIN_DIR, fold=config.FOLD, rand_slices=True, val=False,
                                 percentage_val_samples=15, augmentation=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_dataset = Mprage2space(root_dir=config.TRAIN_DIR, fold=config.FOLD, rand_slices=True, val=True,
                               percentage_val_samples=15, augmentation=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    # MODEL -------------
    disc = Discriminator(in_channels=1).to(config.DEVICE)
    gen = Generator(in_channels=1, features=64).to(config.DEVICE)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    VGG_Loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            os.path.join(config.LOAD_MODEL_PATH, config.CHECKPOINT_GEN_BEST), gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join(config.LOAD_MODEL_PATH, config.CHECKPOINT_DISC_BEST), disc, opt_disc, config.LEARNING_RATE,
        )
    if config.RESUME_TRAINING:
        with open(os.path.join(config.CHECKPOINTS, "fold_" + str(config.FOLD), 'training_info.json'), 'r') as train_file:
            data = json.load(train_file)
            config.RESUME_TRAINING_EPOCH = sorted(list(map(int, data.keys())), reverse=True)[0]

        config.RESUME_TRAINING_PATH = os.path.join(config.CHECKPOINTS, "fold_" + str(config.FOLD))

        load_checkpoint(
            os.path.join(config.RESUME_TRAINING_PATH, config.CHECKPOINT_GEN_BEST), gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join(config.RESUME_TRAINING_PATH, config.CHECKPOINT_DISC_BEST), disc, opt_disc, config.LEARNING_RATE,
        )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.RESUME_TRAINING_EPOCH, config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, BCE, VGG_Loss, g_scaler, d_scaler, epoch
        )

        evaluate(gen, disc, val_loader, epoch, VGG_Loss, opt_disc, opt_gen, config.FOLD)
