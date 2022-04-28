from torch.utils.data import DataLoader
import config
from dataloading.mprage2space import Mprage2space
from tqdm import tqdm
import torch

if __name__ == '__main__':
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

    for x, y in tqdm(train_loader):
        y = torch.reshape(y, (-1, 1, 256, 256))
        print(y.shape)

    for x, y in tqdm(val_loader):
        y = torch.reshape(y, (-1, 1, 256, 256))
        print(y.shape)