# %%

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

import torch
from torch import nan_to_num
from torchvision import transforms as T
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
import requests

from datasets import load_dataset
from torchvision.utils import save_image
from torch.optim import Adam

from copy import deepcopy
import os

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# %%
from mylib import *
import mylora

# %%
sampler = Sampler(linear_beta_schedule, 300)

# %%
settings = Settings(
    results_folder = Path("./results-cifar/3-baseline-bw"),
    image_size = 28,
    channels = 3,
    batch_size = 128,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint = 'checkpoints/5-cifar-bw.pt'
)
settings

# %%
settings.results_folder.mkdir(exist_ok=True, parents=True)

# %%
dataset = load_dataset("cifar10")
# define image transformations (e.g. using torchvision)
transform = Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
])

# define function
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert('L').convert('RGB')) for image in examples["img"]]
   del examples["img"]

   return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=settings.batch_size, shuffle=True)

# %% [markdown]
# # Train baseline

# %%
set_all_seeds()
model = Unet(
    dim=settings.image_size,
    channels=settings.channels,
    dim_mults=(1, 2, 4,)
)

model.to(settings.device)
mylora.model_summary(model)

# %%
settings

# %%
optimizer = Adam(model.parameters(), lr=1e-3)
train(model, optimizer, dataloader, sampler, settings, epochs=10)

# %%
os.system(f"ffmpeg -f image2 -framerate 7 -i {str(settings.results_folder)}/sample-%d.png -loop -0 {str(settings.results_folder)}/sample.gif")

# %%
torch.save(model.state_dict(), settings.checkpoint)

# %% [markdown]
# # Train lora 

# %%
set_all_seeds()
model = Unet(
    dim=settings.image_size,
    channels=settings.channels,
    dim_mults=(1, 2, 4,)
)
model.load_state_dict(torch.load(settings.checkpoint))

mylora.inject_lora(
    model, 2, 0.4,
    ['LinearAttention'],
    [nn.Conv2d]
)
model.to(settings.device)

mylora.freeze_lora(model)
print()
mylora.model_summary(model)

# %%
# from mylib.train import save_images
# save_images(model, sampler, settings, 'grid', 100)

# %%
def transforms(examples):
   examples["pixel_values"] = [transform(image) for image in examples["img"]]
   del examples["img"]

   return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=settings.batch_size, shuffle=True)

# %%
settings.results_folder = Path("./results-cifar/4-rank=2_do=0.25")
settings.results_folder.mkdir(exist_ok=True)

optimizer = Adam(model.parameters(), lr=1e-3)
train(model, optimizer, dataloader, sampler, settings, epochs=10)

# %%
os.system(f"ffmpeg -f image2 -framerate 7 -i {str(settings.results_folder)}/sample-%d.png -loop -0 {str(settings.results_folder)}/sample.gif -y")


