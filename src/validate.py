from collections import OrderedDict
from typing import Mapping, Any, Tuple

import torch
from catalyst import dl
from catalyst.utils import set_global_seed
from diffusers.optimization import get_cosine_schedule_with_warmup
import pandas as pd

from models.ddpm.ddpm_pipeline import DDPMPipeline
from models.ddpm.sheduler import create_noise_scheduler
from models.unet2d import UNet2D
from data.dataset import create_dataset, create_dataloader
from data.augmentations import create_default_augmentations
from pipelines.sr_pipeline import SRPipeline, CustomRunner
from visualization.plot import make_sr_grid

from utils.debug import get_debug_dataloaders
import os
from metrics.image_metrics import SSIM


from configs.sr_config_v1 import config  # TODO set config for train and val


def validate(model, config):

    val_loader = SRPipeline.create_val_loader(config)
    # _, val_loader, _ = get_debug_dataloaders(config)

    noise_scheduler = create_noise_scheduler(config)

    callbacks = OrderedDict({
        "criterion": dl.CriterionCallback(
            input_key="noise_pred", target_key="noise_target", metric_key="loss"
        ),
        # "visualization": VisualizationCallback(vis_loader),
    })

    runner = CustomRunner(
        noise_scheduler=noise_scheduler,
        config=config,
        input_key=["conditioned_noise", "timesteps"],
        output_key=["noise_pred"],
        target_key=["noise_target"],
        loss_key="loss",
    )

    metrics = runner.evaluate_loader(
        model=model,
        loader=val_loader,
        callbacks=callbacks,
        seed=config.seed,
        verbose=True,
    )

    return metrics


    # TODO use for fast inference generation; put output to input again in batch mode
    # CustomRunner.predict_batch
    # CustomRunner.predict_loader


if __name__ == '__main__':
    set_global_seed(config.seed)

    model_path = "best.pt"

    # add jit model

    model = torch.load(model_path, map_location=torch.device(config.device))
    metrics = validate(model, config)

    print(metrics)
