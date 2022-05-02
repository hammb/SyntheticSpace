#!/usr/bin/env python3
import json
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from tqdm import tqdm

import config
from architectures.discriminator_model import Discriminator
from architectures.generator_model import Generator
from dataloading.batchgenerators_mprage2space import Mprage2space
from loss_functions.vgg_loss import VGGLoss
from utils.utils_batchgenerators import load_checkpoint, evaluate

torch.backends.cudnn.benchmark = True
from command_line_arguments.command_line_arguments import CommandLineArguments

config.TRAINER = os.path.basename(__file__)[:-3]


def get_split():
    random.seed(config.FOLD)

    all_samples = os.listdir(config.TRAIN_DIR)

    percentage_val_samples = 15
    # 15% val. data

    num_val_samples = int(len(all_samples) / 100 * percentage_val_samples)
    val_samples = random.sample(all_samples, num_val_samples)

    train_samples = list(filter(lambda sample: sample not in val_samples, all_samples))

    return train_samples, val_samples


def get_train_transform(patch_size):
    tr_transforms = []

    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=False,
            do_rotation=True,
            do_scale=False,
            random_crop=False,
            p_rot_per_sample=0.66,
        )
    )

    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def train_fn(
        disc, gen, loader, opt_disc, opt_gen, bce, VGG_Loss, g_scaler, d_scaler, epoch
):
    loop = tqdm(loader, total=len(loader.generator.indices), leave=True)
    loop.set_description("Training Epoch Nr.: " + str(epoch))
    random.seed(epoch)

    for batch_idx, batch in enumerate(loop):

        start_idx = random.randint(0, 244 - config.RAND_SAMPLE_SIZE - 1)

        x = torch.from_numpy(batch["data"][:, :, :, :, start_idx:start_idx + config.RAND_SAMPLE_SIZE]).to(config.DEVICE)
        y = torch.from_numpy(batch["seg"][:, :, :, :, start_idx:start_idx + config.RAND_SAMPLE_SIZE]).to(config.DEVICE)

        x = torch.reshape(x, (-1, 1, 256, 256))
        y = torch.reshape(y, (-1, 1, 256, 256))

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
                G_loss = VGG_Loss(y_fake.expand(config.RAND_SAMPLE_SIZE, 3, 256, 256),
                                  y.expand(config.RAND_SAMPLE_SIZE, 3, 256, 256))  # Expand single value to RGB

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

    cma = CommandLineArguments()
    cma.parser.add_argument('-ss', '--sample_size', default=2,
                            help='Slices per patient', required=True, type=int)
    cma.parse_args()
    config.RAND_SAMPLE_SIZE = cma.args.sample_size

    # DATA
    train_samples, val_samples = get_split()

    dl_train = Mprage2space(train_samples, config.BATCH_SIZE, config.PATCH_SIZE, config.NUM_WORKERS,
                            seed_for_shuffle=config.FOLD,
                            return_incomplete=False, shuffle=True)

    transform = get_train_transform(config.PATCH_SIZE)

    mt_train = MultiThreadedAugmenter(
        data_loader=dl_train,
        transform=transform,
        num_processes=config.NUM_WORKERS,
    )

    dl_val = Mprage2space(val_samples, config.BATCH_SIZE, config.PATCH_SIZE, config.NUM_WORKERS,
                          return_incomplete=False, shuffle=False)

    mt_val = MultiThreadedAugmenter(
        data_loader=dl_val,
        transform=Compose([]),
        num_processes=config.NUM_WORKERS,
    )
    # MODEL
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
        with open(os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(config.FOLD), 'training_info.json'),
                  'r') as train_file:
            data = json.load(train_file)
            config.RESUME_TRAINING_EPOCH = sorted(list(map(int, data.keys())), reverse=True)[0]

        config.RESUME_TRAINING_PATH = os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(config.FOLD))

        load_checkpoint(
            os.path.join(config.RESUME_TRAINING_PATH, config.CHECKPOINT_GEN_BEST), gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join(config.RESUME_TRAINING_PATH, config.CHECKPOINT_DISC_BEST), disc, opt_disc,
            config.LEARNING_RATE,
        )

    rand_slices = False
    percentage_val_samples = 15
    rand_spacing = True

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(0, config.NUM_EPOCHS):
        train_fn(
            disc, gen, mt_train, opt_disc, opt_gen, BCE, VGG_Loss, g_scaler, d_scaler, epoch
        )

        evaluate(gen, disc, mt_val, epoch, VGG_Loss, opt_disc, opt_gen, config.FOLD, BCE)
