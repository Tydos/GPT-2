import torch
import torch.nn as nn


class Embedding:
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """Create a embedding matrix"""
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

    def generate_token_embedding(self, tokens: list[int]) -> torch.Tensor:
        """Map token IDs to dense vectors"""
        token_tensor = torch.tensor(tokens)
        return self.embedding(token_tensor)  # (seq_len, embed_dim)

    def generate_positional_embedding(self, max_len: int) -> torch.Tensor:
        """Return positional embeddings for positions 0..max_len-1."""
        pos_embedding = nn.Embedding(max_len, self.embed_dim)
        positions = torch.arange(max_len)
        return pos_embedding(positions)  # (max_len, embed_dim)

    def generate_input_embedding(self, tokens: list[int]) -> torch.Tensor:
        """Token embeddings + positional embeddings — the transformer input."""
        seq_len = len(tokens)
        token_emb = self.generate_token_embedding(tokens)
        pos_emb = self.generate_positional_embedding(seq_len)
        return token_emb + pos_emb  # (seq_len, embed_dim)
