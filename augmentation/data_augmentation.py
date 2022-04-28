from volumentations import *


def get_rot_augmentation():
    return Compose([
        Rotate((-360, 360), (-360, 360), (-360, 360), p=0.5),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
    ], p=1.0)


class DefaultDataAugmentation:

    @staticmethod
    def execute(x, y):
        aug = get_rot_augmentation()
        data = {'image': x, 'mask': y}
        aug_data = aug(**data)
        return aug_data['image'], aug_data['mask']