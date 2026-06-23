import math

import torch
import torch.nn as nn
from src.model.config import GPTConfig
from src.model.transformer import Transformer
from src.model.norm import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.dropout_layer = nn.Dropout(config.drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[Transformer(config) for _ in range(config.n_layer)]
        )
        self.final_norm = LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight # tied weights with output head and embedding layer

        self.apply(self._init_weights)
       
        # shrink down the reiduals so that variance is bounded, and model is more stable
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith(("out_proj.weight", "net.2.weight")): # last layer
                nn.init.normal_(param, std=residual_std)

    def _init_weights(self, module):
        """GPT-2 init: N(0, 0.02) weights and set zero biases(numbers start at 0)"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        T = tokens.shape[1]
        positions = torch.arange(T, device=tokens.device)
        x = self.dropout_layer(self.token_embedding(tokens) + self.position_embedding(positions))
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.output_head(x)
