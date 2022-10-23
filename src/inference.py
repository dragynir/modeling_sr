from dataclasses import dataclass

import torch
from catalyst.utils import set_global_seed
from models.ddpm.sheduler import create_noise_scheduler
from pipelines.sr_pipeline import SRPipeline
from utils.debug import get_debug_dataloaders
from models.ddpm.ddpm_pipeline import DDPMPipeline

from visualization.plot import make_sr_grid
from catalyst import utils
from models.unet2d import UNet2D
import os
import numpy as np
from tqdm import tqdm

from configs import config


def validate(model, config: dataclass, tag: str, debug=False):

    if debug:
        _, val_loader, _ = get_debug_dataloaders(config)
    else:
        val_loader = SRPipeline.create_test_loader(config)

    noise_scheduler = create_noise_scheduler(config)

    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

    save_images_path = os.path.join(
        config.checkpoints_path, config.experiment, f"validate_results_{tag}"
    )
    os.makedirs(save_images_path, exist_ok=True)

    print("Start validation...")
    # Переписать на предикты по батчам в рамках одной картинки
    for batch_ind, batch in tqdm(enumerate(val_loader)):
        condition_images = batch["lr_image"]
        hr_images = batch["hr_image"]

        # print(image_condition.shape)
        # print(high_resolution.shape)

        sr_images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
            condition_images=condition_images,
            num_train_timesteps=config.num_train_timesteps,
            output_type="np",
            sample_size=config.test_image_size,
        )["images"]

        for ind, (hr_image, sr_image, lr_image) in enumerate(
            zip(hr_images, sr_images, condition_images)
        ):
            # print(image_condition.shape)
            # print(high_resolution.shape)
            # print(sr_image.shape)
            image_grid = make_sr_grid(lr_image, hr_image, sr_image, as_image=False)
            save_path = os.path.join(save_images_path, f"img_{batch_ind}_{ind}.npy")
            np.save(save_path, image_grid)
            # break
        # break

    # example for noise metrics validation
    # runner = CustomRunner(
    #     noise_scheduler=noise_scheduler,
    #     config=config,
    #     input_key=["conditioned_noise", "timesteps"],
    #     output_key=["noise_pred"],
    #     target_key=["noise_target"],
    #     loss_key="loss",
    #     # criterion=config.criterion,
    # )
    # metrics = runner.evaluate_loader(
    #     model=model,
    #     loader=val_loader,
    #     seed=config.seed,
    #     verbose=True,
    # )
    # return metrics
    # CustomRunner.predict_batch
    # CustomRunner.predict_loader


# CUDA_VISIBLE_DEVICES="1" nohup python inference.py &
# CUDA_VISIBLE_DEVICES="1" python inference.py

if __name__ == "__main__":
    set_global_seed(config.seed)

    model_path = "/home/d_korostelev/Projects/super_resolution/modeling_sr/checkpoints/experiment0_tomo_x4/runner.best.pth"
    # add jit model

    model = UNet2D.create_from_config(config)
    checkpoint = utils.load_checkpoint(path=model_path)
    utils.unpack_checkpoint(
        checkpoint=checkpoint,
        model=model,
    )
    validate(model, config, tag="np_all", debug=False)
