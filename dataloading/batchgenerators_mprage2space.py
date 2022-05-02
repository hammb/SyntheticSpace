import os
import random

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from batchgenerators.transforms.abstract_transforms import Compose
import config
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import random


class Mprage2space(DataLoader):

    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False,
                 shuffle=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         False)

        self.patch_size = patch_size
        self.num_modalities = 1
        self.indices = list(range(len(self._data)))

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)

        for i, j in enumerate(patients_for_batch):
            sample = j

            # Get image path

            if os.path.exists(os.path.join(config.TRAIN_DIR, sample, "mprage_0.nii.gz")):
                input_image_path = os.path.join(config.TRAIN_DIR, sample, "mprage_0.nii.gz")
                output_image_path = os.path.join(config.TRAIN_DIR, sample, "space_0.nii.gz")

                # Load image
                input_image = nib.load(input_image_path)
                output_image = nib.load(output_image_path)

                # Get tensor from image
                input_image = input_image.get_fdata()
                output_image = output_image.get_fdata()
            else:

                input_image_path = os.path.join(config.TRAIN_DIR, sample, "mprage_0.npy")
                output_image_path = os.path.join(config.TRAIN_DIR, sample, "space_0.npy")

                input_image = np.load(input_image_path, mmap_mode="r")
                output_image = np.load(output_image_path, mmap_mode="r")

            # Get tensor from image
            input_image = input_image / config.MAX_VALUE_MPRAGE
            output_image = output_image / config.MAX_VALUE_SPACE

            data[i] = input_image
            seg[i] = output_image

        return {'data': data, 'seg': seg, "sample": sample}


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


if __name__ == '__main__':

    fold = 1
    batch_size = 1
    patch_size = (256, 256, 244)
    num_threads_in_mt = 2

    random.seed(fold)

    config.TASK = "Task004_mprage2space"
    config.TRAIN_DIR = "/home/AD/b556m/data/SyntheticSpace/preprocessed_data/tasks/Task004_mprage2space"
    all_samples = os.listdir("/home/AD/b556m/data/SyntheticSpace/preprocessed_data/tasks/Task004_mprage2space")

    percentage_val_samples = 15
    # 15% val. data

    num_val_samples = int(len(all_samples) / 100 * percentage_val_samples)
    val_samples = random.sample(all_samples, num_val_samples)

    train_samples = list(filter(lambda sample: sample not in val_samples, all_samples))

    dl_train = Mprage2space(val_samples, config.BATCH_SIZE, config.PATCH_SIZE, num_threads_in_mt,
                            seed_for_shuffle=config.FOLD,
                            return_incomplete=False, shuffle=True)

    transform = get_train_transform(config.PATCH_SIZE)

    mt_train = MultiThreadedAugmenter(
        data_loader=dl_train,
        transform=transform,
        num_processes=num_threads_in_mt
    )

    for batch in mt_train:
        print(batch["sample"])
        print("test")
