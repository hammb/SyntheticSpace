import os
import random
import argparse
import torch
from utils import load_checkpoint, evaluate
import torch.nn as nn
import torch.optim as optim
import config
from dataloading.space_rt import SpaceRt
from architectures.generator_model import Generator
from architectures.discriminator_model import Discriminator
from tqdm import tqdm
from loss_functions.vgg_loss import VGGLoss

torch.backends.cudnn.benchmark = True


def get_split():
    random.seed(config.FOLD)

    all_samples = os.listdir(config.TRAIN_DIR)

    percentage_val_samples = 15
    # 15% val. data

    num_val_samples = int(len(all_samples) / 100 * percentage_val_samples)
    val_samples = random.sample(all_samples, num_val_samples)

    train_samples = list(filter(lambda sample: sample not in val_samples, all_samples))

    return train_samples, val_samples


def train_fn(
        disc, gen, loader, opt_disc, opt_gen, bce, VGG_Loss, g_scaler, d_scaler, epoch
):
    loop = tqdm(loader, total=len(loader.generator.indices), leave=True)
    loop.set_description("Training Epoch Nr.: " + str(epoch))
    random.seed(epoch)

    for batch_idx, batch in enumerate(loop):

        start_idx = random.randint(0, 244 - config.RAND_SAMPLE_SIZE - 1)

        x = torch.from_numpy(batch["data"][:, :, :, :, start_idx:start_idx+config.RAND_SAMPLE_SIZE]).to(config.DEVICE)
        y = torch.from_numpy(batch["seg"][:, :, :, :, start_idx:start_idx+config.RAND_SAMPLE_SIZE]).to(config.DEVICE)

        x = torch.einsum('ijkl->lijk', torch.squeeze(x)[None, :, :, :])
        y = torch.einsum('ijkl->lijk', torch.squeeze(y)[None, :, :, :])

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

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', default=0,
                        help='The fold number for k-fold-crossval', required=False, type=int)
    parser.add_argument('-ss', '--sample_size', default=2,
                        help='Num of sLices used per patient', required=False, type=int)

    args = parser.parse_args()

    config.FOLD = args.fold
    config.RAND_SAMPLE_SIZE = args.sample_size
    # DATA

    train_dataset = SpaceRt(root_dir=config.TRAIN_DIR, fold=config.FOLD, rand_slices=True, val=False,
                            percentage_val_samples=15, rand_spacing=False)

    # MODEL
    disc = Discriminator(in_channels=1).to(config.DEVICE)
    gen = Generator(in_channels=1, features=64).to(config.DEVICE)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    # L1_LOSS = nn.L1Loss()
    VGG_Loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            os.path.join("fold_" + str(config.FOLD), config.CHECKPOINT_GEN_BEST), gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join("fold_" + str(config.FOLD), config.CHECKPOINT_DISC_BEST), disc, opt_disc, config.LEARNING_RATE,
        )

    rand_slices = False
    percentage_val_samples = 15
    rand_spacing = True

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    os.makedirs("fold_" + str(config.FOLD), exist_ok=True)

    for epoch in range(0, config.NUM_EPOCHS):
        train_fn(
            disc, gen, mt_train, opt_disc, opt_gen, BCE, VGG_Loss, g_scaler, d_scaler, epoch
        )

        evaluate(gen, disc, dl_val, epoch, VGG_Loss, opt_disc, opt_gen, config.FOLD)
