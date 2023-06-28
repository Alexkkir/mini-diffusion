from torch import nn
from collections import OrderedDict

__all__ = ["model_summary", "freeze_module", "unfreeze_module"]


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def isinstance_by_class(module, _classes):
    return isinstance(module, tuple(_classes))


def isinstance_by_str(module, cls_names):
    return any(module.__class__.__name__ == cls_name for cls_name in cls_names)


def model_summary(model: nn.Module):
    cnt = OrderedDict([
        ("total layers", 0),
        ("trainable layers", 0), 
        ("frozen layers", 0), 
        ("total params", 0),
        ("trainable params", 0),
        ("frozen params", 0),
    ])
    for p in model.parameters():
        if p.requires_grad == True:
            cnt["trainable layers"] += 1
            cnt["trainable params"] += p.numel()
        else:
            cnt["frozen layers"] += 1
            cnt["frozen params"] += p.numel()
        cnt["total layers"] += 1
        cnt["total params"] += p.numel()

    n = 3
    for i, (k, v) in enumerate(cnt.items()):
        if i == 3:
            print()
        print(f'{k + ":":20s} {v:10d}')
