from typing import List

import numpy as np
from PIL import Image
from torch import Tensor


def make_grid(images: List) -> np.ndarray:
    """Make images grid from list of PIL images"""
    if len(images) == 1:
        return np.array(images[0])

    cols = len(images) // 2
    rows = cols + 1
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return np.array(grid)


def tensor_to_image(image: Tensor) -> np.ndarray:
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(1, 2, 0).numpy()
    return image


def make_sr_grid(
    lr_image: Tensor,
    hr_image: Tensor,
    sr_image: Image,
    as_image=True,
) -> np.ndarray:

    lr_image = tensor_to_image(lr_image)
    hr_image = tensor_to_image(hr_image)
    sr_image = np.array(sr_image)

    if as_image:
        lr_image = np.concatenate([lr_image] * 3, axis=-1) * 255
        hr_image = np.concatenate([hr_image] * 3, axis=-1) * 255

    # print(lr_image.min(), lr_image.max())

    return np.concatenate([lr_image, sr_image, hr_image], axis=1)
