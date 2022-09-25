from collections import OrderedDict
from typing import Mapping, Any, Tuple

import torch
from catalyst import dl
from diffusers.optimization import get_cosine_schedule_with_warmup
import pandas as pd

from models.ddpm.ddpm_pipeline import DDPMPipeline
from models.ddpm.sheduler import create_noise_scheduler
from models.unet2d import UNet2D
from data.dataset import create_dataset, create_dataloader
from data.augmentations import create_default_augmentations
from visualization.plot import make_grid


class SRPipeline(object):
    @staticmethod
    def create_train_loader(config):
        df = pd.read_csv(config.data_path)
        df = df[df.split == "train"]

        dataset = create_dataset(
            df=df,
            config=config,
            augmentations=create_default_augmentations(),
            mode="train",
        )

        return create_dataloader(
            dataset=dataset,
            batch_size=config.train_batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            drop_last=True,
        )

    @staticmethod
    def create_val_loader(config):
        df = pd.read_csv(config.data_path)
        df = df[df.split == "valid"]

        dataset = create_dataset(
            df=df,
            config=config,
            augmentations=None,
            mode="valid",
        )

        return create_dataloader(
            dataset=dataset,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

    def run(self, config):

        model = UNet2D.create_from_config(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        train_loader = self.create_train_loader(config)
        valid_loader = self.create_val_loader(config)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_loader) * config.num_epochs),
        )

        noise_scheduler = create_noise_scheduler(config)

        # runner
        loaders = OrderedDict({"train": train_loader, "valid": valid_loader})
        criterion = torch.nn.MSELoss()

        runner = CustomRunner(
            noise_scheduler=noise_scheduler,
            config=config,
            input_key=["conditioned_noise", "timesteps"],
            output_key=["noise_pred"],
            target_key=["noise_target"],
            loss_key="loss",
        )

        callbacks = [
            dl.BatchTransformCallback(
                input_key=["hr_image", "lr_image"],
                output_key=["conditioned_noise", "timesteps", "noise_target"],
                transform=runner.generate_noise,
                scope="on_batch_start",
            ),
            dl.CriterionCallback(
                input_key="noise_pred", target_key="noise_target", metric_key="loss"
            ),
        ]

        runner.train(
            model=model,
            optimizer=optimizer,
            loaders=loaders,
            criterion=criterion,
            scheduler=lr_scheduler,
            callbacks=callbacks,
            num_epochs=config.num_epochs,
            verbose=True,  # you can pass True for more precise training process logging
            timeit=False,  # you can pass True to measure execution time of different parts of train process
        )


class CustomRunner(dl.SupervisedRunner):

    def __init__(self, noise_scheduler, config, **kwargs):
        super().__init__(**kwargs)
        self.noise_scheduler = noise_scheduler
        self.config = config

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        noise_pred = self.model(batch["conditioned_noise"], batch["timesteps"])["sample"]
        return {'noise_pred': noise_pred}

    def handle_batch(self, batch):
        if self.is_train_loader:
            super().handle_batch(batch)
        else:
            self.handle_valid_batch(batch)

    def handle_valid_batch(self, batch):
        pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)

        sr_images = pipeline(
            batch_size=self.config.eval_batch_size,
            generator=torch.manual_seed(self.config.seed),
            condition_images=batch["lr_image"],
        )["images"]

        image_grid = make_grid(sr_images)

    def generate_noise(
        self, hr_images, lr_images,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Sample noise to add to the images
        noise = torch.randn(hr_images.shape).to(hr_images.device)
        bs = hr_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (bs,),
            device=hr_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(hr_images, noise, timesteps)

        conditioned_images = torch.cat((noisy_images, lr_images), dim=1)

        return conditioned_images, timesteps, noise

    @property
    def logger(self) -> Any:
        return {
            "console": dl.ConsoleLogger(),
            # "wandb": dl.WandbLogger(
            #     project=self.config.project,
            #     name=self.config.experiment,
            # ),
        }
