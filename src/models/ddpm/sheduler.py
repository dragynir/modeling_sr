from dataclasses import dataclass

from diffusers import DDPMScheduler


def create_noise_scheduler(config: dataclass) -> DDPMScheduler:
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        tensor_format="pt",
    )
    return noise_scheduler
