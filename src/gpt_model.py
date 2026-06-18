# implement layer normalization, feed forward network


import torch
import torch.nn as nn
from transformers import Transformer
from norm import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config["VOCAB_SIZE"], config["EMBED_DIM"])
        self.position_embedding = nn.Embedding(
            config["CONTEXT_LENGTH"], config["EMBED_DIM"]
        )
        self.dropout_layer = nn.Dropout(config["DROP_RATE"])
        self.transformer_blocks = nn.Sequential(
            *[Transformer(config) for _ in range(config["N_LAYER"])]
        )
        self.final_norm = LayerNorm(config["EMBED_DIM"])
        self.output_head = nn.Linear(
            config["EMBED_DIM"], config["VOCAB_SIZE"], bias=False
        )

    def forward(self, input):
        batch_size, sequence_length = input.shape
        self.token_embedding = self.token_embedding(input)
        self.position_embedding = self.position_embedding(
            torch.arange(sequence_length, device=input.device)
        )
        x = self.token_embedding + self.position_embedding
        x = self.dropout_layer(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits
