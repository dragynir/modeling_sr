from datasets import load_dataset
from torchvision import transforms
import torch
from functools import partial

# another datasets for debug
# Feel free to try other datasets from https://hf.co/huggan/ too! 
# Here's is a dataset of flower photos:
# config.dataset_name = "huggan/flowers-102-categories"
# dataset = load_dataset(config.dataset_name, split="train")

# Or just load images from a local folder!
# config.dataset_name = "imagefolder"
# dataset = load_dataset(config.dataset_name, data_dir="path/to/folder")


def transform(examples, preprocess_high_res, preprocess_low_res):
    images = [preprocess_high_res(image.convert("RGB")) for image in examples["image"]]
    low_res_images = [preprocess_low_res(image.convert("RGB")) for image in examples["image"]]
    return {"hr_image": images, "lr_image": low_res_images}


def get_debug_dataloaders(config):
    
    train_samples = 1000
    val_samples = 10
    
    preprocess_high_res = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    preprocess_low_res = transforms.Compose(
        [
            transforms.Resize((config.lr_image_size, config.lr_image_size)),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    config.dataset_name = "huggan/smithsonian_butterflies_subset"
    
    dataset = load_dataset(config.dataset_name, split="train")#.select(range(train_samples))
    
    dataset.set_transform(
        partial(transform,
                preprocess_high_res=preprocess_high_res,
                preprocess_low_res=preprocess_low_res,
               )
    )
    
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    
    val_dataset = load_dataset(config.dataset_name, split="train").select(range(val_samples))
    
    val_dataset.set_transform(
        partial(
                transform,
                preprocess_high_res=preprocess_high_res,
                preprocess_low_res=preprocess_low_res,
               ))
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, val_dataloader


