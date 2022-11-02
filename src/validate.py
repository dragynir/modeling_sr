from dataclasses import dataclass, field
import os
from typing import Callable, Dict

import numpy as np
import torch
from tqdm import tqdm

from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
# LearnedPerceptualImagePatchSimilarity
# UniversalImageQualityIndex

from configs import config
import click


@dataclass
class ValidationMetric:
    metric_f: Callable
    name: str
    result: float = 0.0
    images_results: Dict[str, float] = field(default_factory=dict)


def compute_metric(images_results: Dict[str, float]) -> float:
    return np.mean(np.array(list(images_results.values())))


@click.command()
@click.option('--tag', default='np_all', help='Tag for experiment', required=False)
@click.option('--images_path', default='', help='Folder with np images', required=False)
@click.option('--img_size', default=512, help='Image size', required=False)
def validate(tag: str, images_path: str, img_size: int):

    saved_images_path = os.path.join(
        config.checkpoints_path, config.experiment, f"validate_results_{tag}"
    )

    saved_images_path = images_path if images_path else saved_images_path

    # img_size = config.test_image_size

    images_paths = os.listdir(saved_images_path)

    metrics = (
        ValidationMetric(SSIM(), name="SSIM"),
        ValidationMetric(MSSIM(), name="MSSIM"),
        ValidationMetric(PSNR(), name="PSNR"),
    )

    print(f"Found {len(images_paths)} images for validation.")

    # TODO rewrite to faster implementation: batch
    for img_name in tqdm(images_paths):
        image = np.load(os.path.join(saved_images_path, img_name), allow_pickle=False)
        # lr_image = image[:, :img_size, :]
        hr_image = image[:, img_size * 2:, :].squeeze()
        sr_image = image[:, img_size: img_size * 2, :].squeeze()

        for metric in metrics:
            sr_tensor = torch.tensor(sr_image[None, None, ...])
            hr_tensor = torch.tensor(hr_image[None, None, ...])
            value = metric.metric_f(sr_tensor, hr_tensor).item()
            metric.images_results.update({img_name: value})

    for metric in metrics:
        metric.result = compute_metric(metric.images_results)
        print(metric.name, ": ", metric.result)


if __name__ == "__main__":
    validate()
