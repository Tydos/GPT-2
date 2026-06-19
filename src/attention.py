import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.W_query = nn.Linear(
            embed_dim, head_dim, bias=False
        )  # bias layer is not needed
        self.W_key = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_value = nn.Linear(embed_dim, head_dim, bias=False)
        self.head_dim = head_dim

    def forward(self, x):
        Q = self.W_query(x)  # (input_sequence_length, head_dim)
        K = self.W_key(x)  # (input_sequence_length, head_dim)
        V = self.W_value(x)  # (input_sequence_length, head_dim)

        scores = Q @ K.T / self.head_dim**0.5
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V  # (input_sequence_length, head_dim)
        return output, weights


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, dropout=0.0) -> None:
        super().__init__()
        self.W_query = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_key = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_value = nn.Linear(embed_dim, head_dim, bias=False)
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.shape[-2]
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        scores = Q @ K.transpose(-2, -1) / self.head_dim**0.5
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()  # not reusing a mask for simplicity, tradeoff - mask created at every forward pass
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = weights @ V
        return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, dropout=0.0, num_heads=2) -> None:
        super().__init__()
        # stacking multiple heads - slow approach
        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(embed_dim, head_dim, dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x)[0] for head in self.heads], dim=-1)


def plot_attention_heatmap(
    weights: torch.Tensor, tokens: list[str], output_path: str
) -> None:
    sns.heatmap(
        weights.detach().numpy(),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=tokens,
        yticklabels=tokens,
    )
    plt.xlabel("Key (attended to)")
    plt.ylabel("Query (attending)")
    plt.title("Self-Attention Weights")
    plt.savefig(output_path)
    plt.close()
