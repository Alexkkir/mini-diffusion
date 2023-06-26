import json
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.optim import Adam

from mylora import *
from mylora.lora import LORA_MODULES
from mylora.utils import isinstance_by_class, unfreeze_module


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.QKV = nn.Linear(1, 1)
        self.C = nn.Linear(1, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        return self.C(self.lrelu(self.QKV(x)))


class TimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_proj = nn.Linear(1, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        return self.lrelu(self.time_proj(x))


class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.just_linear = nn.Linear(1, 1)
        self.attn = Attention()
        self.time_embedder = TimeEmbedding()

    def forward(self, x):
        return self.attn(self.just_linear(x) + self.time_embedder(x))


def models_eq(sd1, sd2, keys):
    return all(torch.allclose(sd1[key], sd2[key]) for key in keys)


def models_neq(sd1, sd2, keys):
    return all(not torch.allclose(sd1[key], sd2[key]) for key in keys)


def find_lora_modules(model: nn.Module, lora_modules=LORA_MODULES):
    for name, module in model.named_modules():
        if isinstance_by_class(module, lora_modules):
            yield name


def test_only_lora_updating():
    torch.manual_seed(0)

    model = A()
    inject_lora(
        model, 2, 0, ["Attention"], [nn.Linear], [LoraInjectedLinear], verbose=True
    )
    unfreeze_module(model)
    freeze_lora(model)

    optim = Adam(model.parameters())
    sd1 = deepcopy(model.state_dict())

    x = torch.tensor([[1]], dtype=torch.float32)
    optim.zero_grad()
    loss1 = model(x).mean()
    loss1.backward()
    optim.step()

    optim.zero_grad()
    loss2 = model(x).mean()
    loss2.backward()
    optim.step()

    loss3 = model(x).mean()
    sd2 = deepcopy(model.state_dict())

    assert loss1 != loss2
    assert loss1 != loss3
    assert loss2 != loss3

    all_keys = set(sd2.keys())
    lora_keys = {k for k in all_keys if "lora_up" in k or "lora_down" in k}

    print("all_keys:")
    print(json.dumps(list(all_keys), indent=4))
    print("lora_keys:")
    print(json.dumps(list(lora_keys), indent=4))

    assert models_eq(sd1, sd2, all_keys - lora_keys)
    assert models_neq(sd1, sd2, lora_keys)
