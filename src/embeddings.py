import torch
import torch.nn as nn


def convert_id_to_embedding(
    tokens: list[int], vocab_size: int, embed_dim: int
) -> torch.Tensor:
    """Map token IDs to dense vectors via a learnable embedding table."""
    embedding = nn.Embedding(vocab_size, embed_dim)
    token_tensor = torch.tensor(tokens)
    return embedding(token_tensor)  # (seq_len, embed_dim)


def generate_positional_embedding(max_len: int, embed_dim: int) -> torch.Tensor:
    """Return positional embeddings for positions 0..max_len-1."""
    pos_embedding = nn.Embedding(max_len, embed_dim)
    positions = torch.arange(max_len)
    return pos_embedding(positions)  # (max_len, embed_dim)


def generate_input_embedding(
    tokens: list[int], vocab_size: int, embed_dim: int
) -> torch.Tensor:
    """Token embeddings + positional embeddings — the transformer input."""
    seq_len = len(tokens)
    token_emb = convert_id_to_embedding(tokens, vocab_size, embed_dim)
    pos_emb = generate_positional_embedding(seq_len, embed_dim)
    return token_emb + pos_emb  # (seq_len, embed_dim)
