from torch import nn

__all__ = ["model_summary"]


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
    cnt = {"trainable layers": 0, "frozen layers": 0, "total params": 0}
    for p in model.parameters():
        if p.requires_grad == True:
            cnt["trainable layers"] += 1
        else:
            cnt["frozen layers"] += 1
        cnt["total params"] += p.numel()
    for k, v in cnt.items():
        print(f'{k + ":":20s} {v:10d}')
