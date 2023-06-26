import pytest
from torch import nn

from mylora import find_modules


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.QKV = nn.Linear(1, 1)
        self.C = nn.Linear(1, 1)
        self.relu = nn.ReLU()


class TimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_proj = nn.Linear(1, 1)
        self.relu = nn.ReLU()


class LoraInjected(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_linear = nn.Linear(1, 1)
        self.A = nn.Linear(1, 1)
        self.B = nn.Linear(1, 1)
        self.dropout = nn.Dropout1d()


class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.just_linear = nn.Linear(1, 1)
        self.attn = Attention()
        self.time_embedder = TimeEmbedding()


def test_search_in_clean_model():
    a = A()
    names = sorted(
        list(x[3] for x in find_modules(a, ["Attention"], [nn.Linear], [LoraInjected]))
    )
    true_names = sorted(["QKV", "C"])
    assert names == true_names


def test_search_with_already_injected():
    a = A()
    a.attn.QKV = LoraInjected()
    names = sorted(
        list(x[3] for x in find_modules(a, ["Attention"], [nn.Linear], [LoraInjected]))
    )
    true_names = sorted(["C"])
    assert names == true_names
