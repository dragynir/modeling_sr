from dataclasses import dataclass


@dataclass
class TrainingConfig:

    # project
    project = 'super_resolution_tomo_diffusion'
    name = 'experiment0'

    # dataset
    data_path = r"C:\Users\dkoro\PythonProjects\SuperResolution\modeling\data\data.csv"

    # hyperparameters
    in_channels = 2
    out_channels = 1
    image_size = 256  # the generated image resolution
    lr_image_size = 64
    train_batch_size = 1
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 50
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
