import torch.nn as nn
from src.model.norm import LayerNorm
from src.model.feed_forward import FeedForwardNetwork
from src.model.attention import MultiHeadAttention
from src.model.config import GPTConfig


class Transformer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim)
        self.attn = MultiHeadAttention(
            config.embed_dim, config.head_dim, config.drop_rate, config.num_heads
        )
        self.norm2 = LayerNorm(config.embed_dim)
        self.ff = FeedForwardNetwork(config.embed_dim)
        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
