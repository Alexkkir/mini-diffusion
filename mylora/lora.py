from typing import List, Optional, Set, Tuple, Type

import torch
from torch import nn

from .utils import (
    freeze_module,
    isinstance_by_class,
    isinstance_by_str,
    unfreeze_module,
)

__all__ = [
    "find_modules",
    "LoraInjectedLinear",
    "LoraInjectedConv2d",
    "inject_lora",
    "freeze_lora",
]


class LoraInjectedLinear(nn.Module):
    def __init__(self, src_linear: nn.Linear, rank: int, dropout: float, bias=False):
        super().__init__()
        self.rank = rank
        self.in_features = src_linear.in_features
        self.out_features = src_linear.out_features
        self.dropout = dropout

        self.src_linear = src_linear  # maybe deepcopy
        device = self.src_linear.weight.device
        dtype = self.src_linear.weight.dtype

        self.lora_down = nn.Linear(
            self.in_features, rank, bias=bias, device=device, dtype=dtype
        )
        self.lora_up = nn.Linear(
            rank, self.out_features, bias=bias, device=device, dtype=dtype
        )
        self.dropout_layer = nn.Dropout1d(dropout)

        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.src_linear(x) + self.dropout_layer(self.lora_up(self.lora_down(x)))

    def freeze_lora(self):
        freeze_module(self.src_linear)
        unfreeze_module(self.lora_up)
        unfreeze_module(self.lora_down)


class LoraInjectedConv2d(nn.Module):
    def __init__(self, src_conv: nn.Conv2d, rank: int, dropout: float):
        super().__init__()
        self.rank = rank
        self.dropout = dropout

        self.in_channels = src_conv.in_channels
        self.out_channels = src_conv.out_channels
        self.kernel_size = src_conv.kernel_size
        self.bias = src_conv.bias

        self.stride = src_conv.stride
        self.dilation = src_conv.dilation
        self.padding = src_conv.padding
        self.groups = src_conv.groups

        self.src_conv = src_conv
        self.lora_down = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.rank,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=False,
        )
        self.dropout_layer = nn.Dropout(self.dropout)  # was nn.Dropout
        self.lora_up = nn.Conv2d(
            in_channels=self.rank,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.normal_(self.lora_down.weight, std=1 / self.rank)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.src_conv(x) + self.dropout_layer(self.lora_up(self.lora_down(x)))

    def freeze_lora(self):
        freeze_module(self.src_conv)
        unfreeze_module(self.lora_up)
        unfreeze_module(self.lora_down)


LORA_MODULES = [LoraInjectedLinear, LoraInjectedConv2d]


def find_modules(
    model,
    parent_module: List[str],
    injected_modules: List[nn.Module],
    lora_modules: List[nn.Module] = LORA_MODULES,
) -> Tuple[nn.Module, nn.Module, str, str]:
    """
    Find layers, which should be replaced with their lora verions. Return parents of these modules
    and modules themselves
    :param model: Model
    :param target_modules: In which modules we should make lora-version of their childrens
    :param injected_modules: Modules, which will be replaces with their lora-versions. For example nn.Linear
    :param lora_modules: Modules, which are already injected. For example LoraInjectedLinear
    1. target_module => injected_module (chained link)
    2. excluded_module !-> injected_module (direct link)
    """
    # if not isinstance(target_modules, list):
    #     target_modules = [target_modules]
    # if not isinstance(injected_modules, list):
    #     injected_modules = [injected_modules]
    # if not isinstance(lora_modules, list):
    #     lora_modules = [lora_modules]

    ancestors: List[nn.Module] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance_by_str(module, parent_module)
    ]

    for anctestor_name, ancestor in ancestors:
        for parent_name, parent in ancestor.named_modules():
            if isinstance_by_class(parent, lora_modules):
                continue
            for children_name, children in parent.named_children():
                if isinstance_by_class(children, injected_modules):
                    yield parent, children, anctestor_name, children_name,


def inject_lora(
    model: nn.Module,
    rank: int,
    dropout: float,
    target_modules: List[str],
    injected_modules: List[nn.Module],
    lora_modules: List[nn.Module] = LORA_MODULES,
    verbose: bool = True,
):
    # возможно лучше напрямую искать модуль
    # lst = list(find_modules(model, target_modules, injected_modules, lora_modules))
    for parent, children, parent_name, children_name in find_modules(
        model, target_modules, injected_modules, lora_modules
    ):
        if isinstance(children, nn.Linear):
            lora = LoraInjectedLinear(children, rank, dropout)
            parent._modules[children_name] = lora
            if verbose:
                print(
                    f"Injected lora ({lora.in_features}x{rank}x{lora.out_features}) in {parent_name}.{children_name}"
                )
        elif isinstance(children, nn.Conv2d):
            lora = LoraInjectedConv2d(children, rank, dropout)
            parent._modules[children_name] = lora
            if verbose:
                print(
                    f"Injected lora {lora.in_channels:>5d} x {rank:1d} x {lora.out_channels:<5d} in {parent_name}.{children_name}"
                )


def find_lora_modules():
    pass


def find_lora_parameters():
    pass


def freeze_lora(module: nn.Module, lora_modules: List[nn.Module] = LORA_MODULES):
    is_leaf = len(list(module.children())) == 0
    is_lora = isinstance_by_class(module, lora_modules)

    if is_lora:
        module.freeze_lora()
    else:
        freeze_module(module)

    if not is_leaf and not is_lora:
        for children in module.children():
            freeze_lora(children, lora_modules)


def save_lora():
    pass


def load_lora():
    pass
