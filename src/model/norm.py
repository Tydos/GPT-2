import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalisation: squeezes outputs to mean 0 and variance 1.
    eps avoids division by zero.
    scale/shift are learnable affine parameters.
    """

    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.ones(self.embed_dim))
        self.shift = nn.Parameter(torch.zeros(self.embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
