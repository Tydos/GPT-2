import torch
import torch.nn as nn
from src.transformers import Transformer
from src.norm import LayerNorm
import src.config as cfg


class GPTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.VOCAB_SIZE, cfg.EMBED_DIM)
        self.position_embedding = nn.Embedding(cfg.CONTEXT_LENGTH, cfg.EMBED_DIM)
        self.dropout_layer = nn.Dropout(cfg.DROP_RATE)
        self.transformer_blocks = nn.Sequential(
            *[Transformer({"EMBED_DIM": cfg.EMBED_DIM, "HEAD_DIM": cfg.HEAD_DIM,
                           "NUM_HEADS": cfg.NUM_HEADS, "DROP_RATE": cfg.DROP_RATE})
              for _ in range(cfg.N_LAYER)]
        )
        self.final_norm = LayerNorm(cfg.EMBED_DIM)
        self.output_head = nn.Linear(cfg.EMBED_DIM, cfg.VOCAB_SIZE, bias=False)

    def forward(self, input):
        batch_size, sequence_length = input.shape
        tok = self.token_embedding(input)
        pos = self.position_embedding(torch.arange(sequence_length, device=input.device))
        x = self.dropout_layer(tok + pos)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.output_head(x)
