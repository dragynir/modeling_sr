import albumentations as A


def create_default_augmentations() -> A.Compose:

    return A.Compose(
        [
            A.ColorJitter(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]
    )
