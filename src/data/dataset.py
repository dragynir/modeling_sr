from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import torch
import numpy as np
from tifffile import tifffile
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
        self.crop_hr = A.RandomCrop(
            hr_image_size,
            hr_image_size,
            always_apply=True,
        )
        self.hr_size = hr_image_size
        self.lr_size = lr_image_size

        self.mode = mode

        # benchmark dataset post process
        self.post_process = Compose(
            [
                A.Normalize([0.5], [0.5]),  # [-1, 1] normalization
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def __read_image_source(image_path: str) -> np.ndarray:

        if ".tiff" in image_path:
            image = tifffile.imread(image_path)
        else:
            # all channels are equal
            image = cv2.imread(image_path)[:, :, 0]
        return image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_point = self.df.iloc[idx]

        hr_image = self.__read_image_source(data_point.path)
        lr_image = None

        if "lr_path" in self.df.columns:
            lr_image = self.__read_image_source(data_point.lr_path)
            lr_image = cv2.resize(lr_image, (hr_image.shape[0], hr_image.shape[1]))
            stacked = np.stask([hr_image, lr_image, lr_image], axis=-1)
            stacked = self.crop_hr(image=stacked)["image"]
            hr_image = stacked[:, :, 0]
            lr_image = stacked[:, :, 1]
        else:
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
            "hr_image": hr_image,
            "lr_image": lr_image,
        }

        return result


class SuperResolutionTestDataset(SuperResolutionDataset):
    def __init__(self, test_image_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_size = test_image_size
        self.pad = A.PadIfNeeded(
            self.test_size,
            self.test_size,
            always_apply=True,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_point = self.df.iloc[idx]

        hr_image = self.__read_image_source(data_point.path)
        lr_image = None

        if "lr_path" in self.df.columns:
            lr_image = self.__read_image_source(data_point.lr_path)
            lr_image = cv2.resize(lr_image, (self.hr_size, self.hr_size))
        else:
            hr_image = self.pad(image=hr_image)["image"]
            lr_size = self.test_size // (self.hr_size // self.lr_size)

            # downsample image to remove details
            lr_image = cv2.resize(hr_image, (lr_size, lr_size))
            # resize image back to fit hr_image size
            lr_image = cv2.resize(lr_image, (self.test_size, self.test_size))

        lr_image = self.post_process(image=lr_image)["image"]
        hr_image = self.post_process(image=hr_image)["image"]

        result = {
            "hr_image": hr_image,
            "lr_image": lr_image,
        }
        return result


def create_dataset(
    df: pd.DataFrame,
    config: dataclass,
    augmentations: Optional[Compose],
    mode: str = "train",
) -> SuperResolutionDataset:
    """Create dataset from data.csv DataFrame"""

    if mode == "test":
        return SuperResolutionTestDataset(
            config.test_image_size,
            df,
            config.image_size,
            config.lr_image_size,
            augmentations,
            mode,
        )

    return SuperResolutionDataset(
        df,
        config.image_size,
        config.lr_image_size,
        augmentations,
        mode,
    )


def create_dataloader(
    dataset: SuperResolutionDataset,
    batch_size,
    num_workers,
    shuffle,
    drop_last=True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )
