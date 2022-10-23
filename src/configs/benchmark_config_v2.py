from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:

    # project
    project = 'super_resolution_tomo_diffusion'
    experiment = 'experiment1_tomo_x4'
    checkpoints_path = '/home/d_korostelev/Projects/super_resolution/modeling_sr/checkpoints/'
    debug = False  # use overfit dataset
    criterion = torch.nn.L1Loss()

    # tomo benchmark dataset
    data_path = "/home/d_korostelev/Projects/super_resolution/data/v1_dataset_DeepRockSR.csv"

    # hyperparameters
    in_channels = 2
    out_channels = 1
    image_size = 128  # the generated image resolution
    test_image_size = 512  # inference image resolution
    lr_image_size = 32
    num_epochs = 120
    train_batch_size = 16
    eval_batch_size = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    seed = 42

    # system
    num_workers = 2

    # diffusion
    num_train_timesteps = 1000


config = TrainingConfig()
