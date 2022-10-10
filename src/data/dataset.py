from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from albumentations import Compose
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        hr_image_size: int,
        lr_image_size: int,
        augmentations: Compose,
        mode: str = "train",
    ) -> None:
        self.df = df
        self.augmentations = augmentations
        self.crop_hr = A.RandomCrop(hr_image_size, hr_image_size, always_apply=True)
        self.hr_size = hr_image_size
        self.lr_size = lr_image_size

        self.mode = mode

        self.post_process = Compose(
            [
                ToTensorV2(),
                A.Normalize([0.5], [0.5]),  # [-1, 1] normalization
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # TODO добавить чтение tiff изображений
        data_point = self.df.iloc[idx]

        hr_image = cv2.imread(data_point.path)

        hr_image = self.crop_hr(image=hr_image)["image"]

        if self.mode == "train":
            hr_image = self.augmentations(image=hr_image)["image"]

        # downsample image to remove details
        lr_image = cv2.resize(hr_image, (self.lr_size, self.lr_size))
        # resize image back to fit hr_image size
        lr_image = cv2.resize(lr_image, (self.hr_size, self.hr_size))

        lr_image = self.post_process(image=lr_image)["image"]
        hr_image = self.post_process(image=hr_image)["image"]

        result = {
            "hr_image": hr_image[0, ...].unsqueeze(0),
            "lr_image": lr_image[0, ...].unsqueeze(0),
        }

        return result


def create_dataset(
    df: pd.DataFrame,
    config: dataclass,
    augmentations: Optional[Compose],
    mode: str = "train",
) -> SuperResolutionDataset:
    """Create dataset from data.csv DataFrame"""
    return SuperResolutionDataset(df, config.image_size, config.lr_image_size, augmentations, mode)


def create_dataloader(
    dataset: SuperResolutionDataset,
    batch_size,
    num_workers,
    shuffle,
    drop_last=True,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )
