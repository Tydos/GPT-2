import torch.nn as nn
from src.model.norm import LayerNorm
from src.model.feed_forward import FeedForwardNetwork
from src.model.attention import MultiHeadAttentionSDPA
from src.model.config import ModelConfig


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim)
        self.attn = MultiHeadAttentionSDPA(
            config.embed_dim, config.num_heads, config.drop_rate
        )
        self.norm2 = LayerNorm(config.embed_dim)
        self.ff = FeedForwardNetwork(config.embed_dim)
        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
