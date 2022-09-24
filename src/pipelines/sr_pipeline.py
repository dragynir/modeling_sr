from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Any

import torch
from torch.nn import functional as F
from catalyst import dl, metrics, utils
from diffusers.optimization import get_cosine_schedule_with_warmup
import pandas as pd

from models.ddpm.sheduler import create_noise_scheduler
from models.unet2d import UNet2D
from data.dataset import create_dataset, create_dataloader
from data.augmentations import create_default_augmentations


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
            num_workers=2,
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
            num_workers=2,
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

        runner = CustomRunner(noise_scheduler)

        runner.train(
            model=model,
            optimizer=optimizer,
            loaders=loaders,
            criterion=criterion,
            scheduler=lr_scheduler,
            num_epochs=3,
            verbose=True,  # you can pass True for more precise training process logging
            timeit=False,  # you can pass True to measure execution time of different parts of train process
        )


class CustomRunner(dl.Runner):

    @property
    def logger(self) -> Any:
        pass

    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        return self.model(batch)

    def __init__(self, noise_scheduler, **kwargs):
        super(CustomRunner, self).__init__(**kwargs)
        self.noise_scheduler = noise_scheduler

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        # self.meters = {
        #     key: metrics.AdditiveMetric(compute_on_call=False) for key in ["loss", "mae"]
        # }

    def handle_batch(self, batch):
        if self.is_train_loader:
            self.handle_train_batch(batch)
        else:
            self.handle_valid_batch(batch)

    def handle_train_batch(self, batch):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `train()`.
        hr_images = batch['hr_image']
        lr_images = batch['lr_image']

        # Sample noise to add to the images
        noise = torch.randn(hr_images.shape).to(hr_images.device)
        bs = hr_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=hr_images.device,).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(hr_images, noise, timesteps)

        conditioned_images = torch.cat((noisy_images, lr_images), dim=1)

        noise_pred = self.model(conditioned_images, timesteps)["sample"]
        loss = F.mse_loss(noise_pred, noise)


        loss.backward()
        self.optimizer.step()
        # self.scheduler.step() # TODO
        self.optimizer.zero_grad()



        # y_pred = self.model(x)  # Forward pass
        #
        # # Compute the loss value
        # loss = F.mse_loss(y_pred, y)
        #
        # # Update metrics (includes the metric that tracks the loss)
        # self.batch_metrics.update({"loss": loss, "mae": F.l1_loss(y_pred, y)})
        # for key in ["loss", "mae"]:
        #     self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        #
        # if self.is_train_loader:
        #     # Compute gradients
        #     loss.backward()
        #     # Update weights
        #     # (the optimizer is stored in `self.state`)
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()

    def on_loader_end(self, runner):
        for key in ["loss", "mae"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)
