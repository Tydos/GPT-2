import torch.nn as nn
from src.norm import LayerNorm
from src.feed_forward import FeedForwardNetwork
from src.attention import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config["EMBED_DIM"]
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, config["HEAD_DIM"], config["DROP_RATE"], config["NUM_HEADS"])
        self.norm2 = LayerNorm(embed_dim)
        self.ff = FeedForwardNetwork(embed_dim)
        self.drop = nn.Dropout(config["DROP_RATE"])

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
