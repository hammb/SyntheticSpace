import os.path
import os
import torch
import config
from torchvision.utils import save_image
import json
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def save_some_examples(gen, val_loader, epoch, folder):
    """

    :rtype: object
    """
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")

    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_validation_loss_plot(data, fold):

    G_losses = []
    D_losses = []

    for key in data:
        G_losses.append(data[key][0])
        D_losses.append(data[key][1])

    plt.plot(data.keys(), G_losses, 'ro')
    plt.plot(data.keys(), D_losses, 'bo')
    plt.ylabel('Losses')
    plt.savefig(
        os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(fold), "validation_loss.png"))


def ev(gen, disc, val_loader, epoch, VGG_Loss, opt_disc, opt_gen, fold):
    os.makedirs(os.path.join(config.CHECKPOINTS, config.TRAINER), exist_ok=True)
    os.makedirs(os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(config.FOLD)),
                exist_ok=True)

    gen.eval()
    loop = tqdm(val_loader, total=len(val_loader.generator.indices), leave=True)
    loop.set_description("Evaluating Epoch Nr.: " + str(epoch))

    losses = []
    for batch_idx, batch in enumerate(loop):
        x = torch.from_numpy(batch["data"]).to(config.DEVICE)
        y = torch.from_numpy(batch["seg"]).to(config.DEVICE)
        print(x.shape)


def evaluate(gen, disc, val_loader, epoch, VGG_Loss, opt_disc, opt_gen, fold, bce):
    """

    :rtype: object
    """

    os.makedirs(os.path.join(config.CHECKPOINTS, config.TRAINER), exist_ok=True)
    os.makedirs(os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(config.FOLD)),
                exist_ok=True)

    gen.eval()

    loop = tqdm(val_loader, total=len(val_loader.generator.indices), leave=True)
    loop.set_description("Evaluating Epoch Nr.: " + str(epoch))

    random.seed(epoch)
    G_losses = []
    D_losses = []

    for batch_idx, batch in enumerate(loop):

        start_idx = random.randint(0, 244 - config.RAND_SAMPLE_SIZE - 1)

        if config.TRAINER == "train_batchgenerators":
            x = torch.from_numpy(batch["data"][:, :, :, :, start_idx:start_idx + config.RAND_SAMPLE_SIZE]).to(
                config.DEVICE)
            y = torch.from_numpy(batch["seg"][:, :, :, :, start_idx:start_idx + config.RAND_SAMPLE_SIZE]).to(
                config.DEVICE)
        else:
            x = torch.from_numpy(batch["data"]).to(config.DEVICE)
            y = torch.from_numpy(batch["seg"]).to(config.DEVICE)

        x = torch.reshape(x, (-1, 1, 256, 256))
        y = torch.reshape(y, (-1, 1, 256, 256))

        with torch.no_grad():
            y_fake = gen(x)

            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            G_loss = VGG_Loss(y_fake.expand(config.RAND_SAMPLE_SIZE * config.BATCH_SIZE, 3, 256, 256),
                            y.expand(config.RAND_SAMPLE_SIZE * config.BATCH_SIZE, 3, 256, 256))

            G_losses.append(float(G_loss.detach().to('cpu').numpy()))
            D_losses.append(float(D_loss.detach().to('cpu').numpy()))

    mean_G_loss = statistics.mean(G_losses)
    mean_D_loss = statistics.mean(D_losses)

    if os.path.exists(os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(fold),
                                   "training_info.json")):
        f = open(os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(fold),
                              "training_info.json"), )
        json_dict = json.load(f)
        f.close()

        previous_G_losses = []

        for key in json_dict:
            previous_G_losses.append(json_dict[key][0])

        if min(previous_G_losses) > mean_G_loss:
            save_checkpoint(gen, opt_gen, filename=os.path.join(config.CHECKPOINTS, config.TRAINER,
                                                                "fold_" + str(fold), config.CHECKPOINT_GEN_BEST))
            save_checkpoint(disc, opt_disc, filename=os.path.join(config.CHECKPOINTS, config.TRAINER,
                                                                  "fold_" + str(fold), config.CHECKPOINT_DISC_BEST))

        json_dict[int(epoch)] = [mean_G_loss, mean_D_loss]
    else:
        json_dict = {int(epoch): [mean_G_loss, mean_D_loss]}

    out_file = open(
        os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(fold), "training_info.json"),
        "w")
    json.dump(json_dict, out_file, indent=6)
    out_file.close()

    save_validation_loss_plot(json_dict, fold)

    if epoch % 10 == 0:
        save_checkpoint(gen, opt_gen,
                        filename=os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(fold),
                                              config.CHECKPOINT_GEN))
        save_checkpoint(disc, opt_disc,
                        filename=os.path.join(config.CHECKPOINTS, config.TRAINER, "fold_" + str(fold),
                                              config.CHECKPOINT_DISC))

    gen.train()
