from inspect import isfunction

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, reduce
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Resize,
    ToPILImage,
    ToTensor,
)

import os
import random

import torch
import json


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(
        figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False
    )
    for row_idx, row in enumerate(imgs):
        # row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def get_transform_aux(image_size):
    transform = Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),  # turn into torch Tensor of shape CHW, divide by 255
            Lambda(lambda t: (t * 2) - 1),
        ]
    )
    return transform


transform = Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)

reverse_transform = Compose(
    [
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.0),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ]
)


class Settings:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__setattr__(key, val)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4, default=lambda x: x.__repr__())


def plot_images(samples, settings: Settings, n=64):
    sqrt = int(n**0.5)
    fig, ax = plt.subplots(8, 8, figsize=(20, 20))
    ax = ax.ravel()
    for i in range(n):
        ax[i].imshow(
            samples[-1][i].reshape(
                settings.image_size, settings.image_size, settings.channels
            ),
            cmap="gray",
        )
    plt.tight_layout()
    plt.show()
    

def set_all_seeds(seed=0):
    # python's seeds
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # torch's seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
