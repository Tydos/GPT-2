import torch
import torch.nn as nn
from src.model.config import GPTConfig
from src.model.transformer import Transformer
from src.model.norm import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.dropout_layer = nn.Dropout(config.drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[Transformer(config) for _ in range(config.n_layer)]
        )
        self.final_norm = LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight # tied weights with output head and embedding layer

    def forward(self, input):
        batch_size, sequence_length = input.shape
        tok = self.token_embedding(input)
        pos = self.position_embedding(torch.arange(sequence_length, device=input.device))
        x = self.dropout_layer(tok + pos)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.output_head(x)
