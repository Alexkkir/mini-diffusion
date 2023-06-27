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
from mylib import *
import mylora

# %%
sampler = Sampler(linear_beta_schedule, 300)

# %%
settings = Settings(
    results_folder = Path("./results-mnist/1-baseline"),
    image_size = 28,
    channels = 1,
    batch_size = 128,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint = 'checkpoints/3-mnist-0:1,3:9.pt'
)
settings

# %%
settings.results_folder.mkdir(exist_ok=True, parents=True)

# %%
dataset = load_dataset("mnist")
# define image transformations (e.g. using torchvision)
transform = Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
])

# define function
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

transformed_dataset = dataset.with_transform(transforms).filter(lambda x: x['label'] != 2).remove_columns("label")

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
optimizer = Adam(model.parameters(), lr=1e-3)
train(model, optimizer, dataloader, sampler, settings, epochs=10)

# %%
settings

# %%
os.system(f"ffmpeg -f image2 -framerate 7 -i {str(settings.results_folder)}/sample-%d.png -loop -0 {str(settings.results_folder)}/sample.gif -y")

# %%
torch.save(model.state_dict(), settings.checkpoint)

# %% [markdown]
# ### Train lora 

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
transformed_dataset = dataset.with_transform(transforms).filter(lambda x: x['label'] == 2).remove_columns("label")
dataloader = DataLoader(transformed_dataset["train"], batch_size=settings.batch_size, shuffle=True)

# %%
settings.results_folder = Path("./results-mnist/2-rank=2_do=0.25")
settings.results_folder.mkdir(exist_ok=True, parents=True)

optimizer = Adam(model.parameters(), lr=1e-3)
train(model, optimizer, dataloader, sampler, settings, epochs=10)

# %%
os.system(f"ffmpeg -f image2 -framerate 7 -i {str(settings.results_folder)}/sample-%d.png -loop -0 {str(settings.results_folder)}/sample.gif -y")


