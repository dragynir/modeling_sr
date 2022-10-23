import warnings
from typing import Tuple

import torch

from diffusers.pipeline_utils import DiffusionPipeline
from collections import OrderedDict
from dataclasses import fields
import PIL
from typing import List, Optional, Union, Any
import numpy as np
import dataclasses as dc


class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.
    <Tip warning={true}>
    You can't unpack a `BaseOutput` directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
    before.
    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            if (
                self.__class__.__name__ in ["StableDiffusionPipelineOutput", "ImagePipelineOutput"]
                and k == "sample"
            ):
                warnings.warn(
                    "The keyword 'samples' is deprecated and will be removed in version 0.4.0. Please use `.images` or"
                    " `'images'` instead.",
                    DeprecationWarning,
                )
                return inner_dict["images"]
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


@dc.dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.
    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class DDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        condition_images,
        sample_size: int = None,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        num_train_timesteps: int = 1000,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        sample_size = sample_size if sample_size is not None else self.unet.sample_size
        # Sample gaussian noise to begin loop
        in_channels_condition = self.unet.in_channels // 2
        noise_batch_image = torch.randn(
            (batch_size, in_channels_condition, sample_size, sample_size),
            generator=generator,
        )

        condition_images = condition_images.to(self.device)
        noise_batch_image = noise_batch_image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_train_timesteps)

        for t in self.scheduler.timesteps:
            # 1. predict noise model_output
            input_condition = torch.cat((noise_batch_image, condition_images), dim=1)
            # print(image_condition.shape, image.shape)
            model_output = self.unet(input_condition, t)["sample"]
            # print(model_output)

            # 2. compute previous image: x_t -> t_t-1
            noise_batch_image = self.scheduler.step(
                model_output,
                t,
                noise_batch_image,
                generator=generator,
            )["prev_sample"]

        noise_batch_image = (noise_batch_image / 2 + 0.5).clamp(0, 1)
        noise_batch_image = noise_batch_image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            # fro gray images
            if noise_batch_image.shape[-1] == 1:
                noise_batch_image = np.concatenate(
                    (
                        noise_batch_image,
                        noise_batch_image,
                        noise_batch_image,
                    ),
                    axis=-1,
                )
            noise_batch_image = self.numpy_to_pil(noise_batch_image)

        if not return_dict:
            return noise_batch_image

        return ImagePipelineOutput(images=noise_batch_image)
