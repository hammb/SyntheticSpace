import os.path
import os
import torch
import config
from torchvision.utils import save_image
import json
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    plt.plot(data.keys(), data.values(), 'ro')
    plt.ylabel('VGG Loss')
    plt.savefig(os.path.join(config.CHECKPOINTS, "fold_" + str(fold), "validation_loss.png"))


def evaluate(gen, disc, val_loader, epoch, VGG_Loss, opt_disc, opt_gen, fold):
    """

    :rtype: object
    """
    os.makedirs(os.path.join(config.CHECKPOINTS, "fold_" + str(config.FOLD)), exist_ok=True)

    gen.eval()
    loop = tqdm(val_loader, total=len(val_loader.indices), leave=True)
    loop.set_description("Evaluating Epoch Nr.: " + str(epoch))

    losses = []
    for batch_idx, batch in enumerate(loop):

        x = torch.from_numpy(batch["data"]).to(config.DEVICE)
        y = torch.from_numpy(batch["seg"]).to(config.DEVICE)

        x = torch.einsum('ijkl->lijk', torch.squeeze(x)[None, :, :, :])
        y = torch.einsum('ijkl->lijk', torch.squeeze(y)[None, :, :, :])

        for slice_idx in range(244 // config.RAND_SAMPLE_SIZE):

            start_idx = slice_idx * config.RAND_SAMPLE_SIZE
            end_idx = ((slice_idx + 1) * config.RAND_SAMPLE_SIZE)

            x_slice = x[start_idx:end_idx, :, :, :]
            y_slice = y[start_idx:end_idx, :, :, :]

            with torch.no_grad():
                y_fake = gen(x_slice)
                loss = VGG_Loss(y_fake.expand(config.RAND_SAMPLE_SIZE, 3, 256, 256), y_slice.expand(config.RAND_SAMPLE_SIZE, 3, 256, 256))
                losses.append(float(loss.detach().to('cpu').numpy()))

    mean_loss = statistics.mean(losses)

    if os.path.exists(os.path.join(config.CHECKPOINTS, "fold_" + str(fold), "training_info.json")):
        f = open(os.path.join(config.CHECKPOINTS, "fold_" + str(fold), "training_info.json"), )
        json_dict = json.load(f)
        f.close()

        if min(json_dict.values()) > mean_loss:
            save_checkpoint(gen, opt_gen, filename=os.path.join(config.CHECKPOINTS, "fold_" + str(fold), config.CHECKPOINT_GEN_BEST))
            save_checkpoint(disc, opt_disc, filename=os.path.join(config.CHECKPOINTS, "fold_" + str(fold), config.CHECKPOINT_DISC_BEST))

        json_dict[int(epoch)] = mean_loss
    else:
        json_dict = {int(epoch): mean_loss}

    out_file = open(os.path.join(config.CHECKPOINTS, "fold_" + str(fold), "training_info.json"), "w")
    json.dump(json_dict, out_file, indent=6)
    out_file.close()

    save_validation_loss_plot(json_dict, fold)

    if epoch % 10 == 0:
        save_checkpoint(gen, opt_gen, filename=os.path.join(config.CHECKPOINTS, "fold_" + str(fold), config.CHECKPOINT_GEN))
        save_checkpoint(disc, opt_disc, filename=os.path.join(config.CHECKPOINTS, "fold_" + str(fold), config.CHECKPOINT_DISC))

    gen.train()
