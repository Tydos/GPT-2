import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    This class implements layer normalisation
    squeeze model outputs with mean 0 and variance 1
    eps is a constant parameter to avoid div by zero errors
    keepdim preserves the shape of the original tensor

    dim = 0 : mean acorss row/one mean per column
    dim = 1/-1 : mean across col/one mean per row

    each row has embeddings across one

    variance unbiased for small input samples, as we increase the input length, it does not matter
    shift parameter acts as a bias for the NN

    """

    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = 1e-5
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.ones(self.embed_dim))
        self.shift = nn.Parameter(torch.zeros(self.embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
