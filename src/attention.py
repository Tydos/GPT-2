import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.W_query = nn.Linear(embed_dim, head_dim, bias=False)  # bias layer is not needed
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


def plot_attention_heatmap(weights: torch.Tensor, tokens: list[str], output_path: str) -> None:
    sns.heatmap(weights.detach().numpy(), annot=True, fmt=".2f", cmap="Blues",
                xticklabels=tokens, yticklabels=tokens)
    plt.xlabel("Key (attended to)")
    plt.ylabel("Query (attending)")
    plt.title("Self-Attention Weights")
    plt.savefig(output_path)
    plt.close()
