import albumentations as A


def create_augmentations(use_default=False) -> A.Compose:

    if use_default:
        return create_default_augmentations()

    return create_hard_augmentations()


def create_default_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.ColorJitter(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]
    )


def create_hard_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.ColorJitter(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        ]
    )
