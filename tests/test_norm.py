import torch
import pytest
from src.norm import LayerNorm

CONFIG = {"EMBED_DIM": 8}


@pytest.fixture
def layer_norm():
    return LayerNorm(CONFIG)


def test_output_shape(layer_norm):
    x = torch.randn(2, 5, 8)
    assert layer_norm(x).shape == x.shape


def test_mean_near_zero(layer_norm):
    x = torch.randn(2, 5, 8)
    out = layer_norm(x)
    # scale/shift are 1/0 by init, so output mean should be ~0
    means = out.mean(dim=-1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)


def test_variance_near_one(layer_norm):
    x = torch.randn(2, 5, 8)
    out = layer_norm(x)
    vars_ = out.var(dim=-1, unbiased=False)
    assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-5)


def test_scale_shift_are_learnable(layer_norm):
    assert layer_norm.scale.requires_grad
    assert layer_norm.shift.requires_grad


def test_scale_shift_initial_values(layer_norm):
    assert torch.all(layer_norm.scale == 1.0)
    assert torch.all(layer_norm.shift == 0.0)


def test_constant_input_does_not_nan():
    # all-same values → variance=0, eps should prevent NaN
    norm = LayerNorm(CONFIG)
    x = torch.ones(2, 5, 8)
    out = norm(x)
    assert not torch.isnan(out).any()


def test_shift_offsets_output():
    norm = LayerNorm(CONFIG)
    with torch.no_grad():
        norm.shift.fill_(3.0)
    x = torch.randn(2, 5, 8)
    out = norm(x)
    means = out.mean(dim=-1)
    assert torch.allclose(means, torch.full_like(means, 3.0), atol=1e-5)
