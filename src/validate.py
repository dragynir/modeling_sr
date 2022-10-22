from collections import Callable
from dataclasses import dataclass, field
import os
from plistlib import Dict

import numpy as np
import torch

from metrics.image_metrics import SSIM, PeakSignalNoiseRatio
from configs.benchmark_config_ import config


@dataclass
class ValidationMetric:
    metric_f: Callable
    name: str
    result: float = 0.0
    images_results: Dict[str, float] = field(default_factory=dict)


def compute_metric(images_results: Dict[str, float]) -> float:
    return np.mean(np.array(images_results.values()))


def validate(config: dataclass, tag: str):

    saved_images_path = os.path.join(
        config.checkpoints_path, config.experiment, f"validate_results_{tag}"
    )

    img_size = config.test_image_size

    images_paths = os.listdir(saved_images_path)

    metrics = (
        ValidationMetric(SSIM(), name="SSIM"),
        ValidationMetric(PeakSignalNoiseRatio(), name="PSNR"),
    )

    print(f"Found {len(images_paths)} images for validation.")

    # TODO rewrite to faster implementation: batch
    for img_name in images_paths:
        image = np.load(os.path.join(saved_images_path, img_name), allow_pickle=False)
        # lr_image = image[:, :img_size, :]
        hr_image = image[:, img_size * 2:, :]
        sr_image = image[:, img_size: img_size * 2, :]

        for metric in metrics:
            value = metric.metric_f(torch.tensor(sr_image), torch.tensor(hr_image)).item()
            metric.images_results.update({img_name: value})

    for metric in metrics:
        metric.result = compute_metric(metric.images_results)
        print(metric.name, ": ", metric.result)


if __name__ == "__main__":
    validate(config, "np_all")
