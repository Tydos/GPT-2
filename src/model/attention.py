import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


class SelfAttention(nn.Module):
    """ Dot product self-attention """
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.W_query = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_key = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_value = nn.Linear(embed_dim, head_dim, bias=False)
        self.head_dim = head_dim

    def forward(self, x):
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        scores = Q @ K.T / self.head_dim**0.5
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V
        return output, weights


class CausalSelfAttention(nn.Module):
    """ Attention with a causal mask """
    def __init__(self, embed_dim, head_dim, dropout=0.0) -> None:
        super().__init__()
        # Kept bias=True to be faithful to the original implementation, but modern LLMs use bias=False
        self.W_query = nn.Linear(embed_dim, head_dim, bias=True)
        self.W_key = nn.Linear(embed_dim, head_dim, bias=True)
        self.W_value = nn.Linear(embed_dim, head_dim, bias=True)
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.shape[-2]
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # Scale dot-product: large Q@K.T values push softmax to near-1, killing gradients
        scores = Q @ K.transpose(-2, -1) / self.head_dim**0.5

        # Upper-triangular mask prevents attending to future tokens
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = weights @ V
        return output, weights


class MultiHeadAttention(nn.Module):
    """ Concatenated multiple causal self-attention heads """
    def __init__(self, embed_dim, head_dim, dropout=0.0, num_heads=2) -> None:
        super().__init__()
        self.out_proj = nn.Linear(embed_dim, embed_dim)  
        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(embed_dim, head_dim, dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return self.out_proj(torch.cat([head(x)[0] for head in self.heads], dim=-1))


class MultiHeadAttentionSDPA(nn.Module):
    """ Optimised multi-head causal self-attention using SDPA """

    def __init__(self, embed_dim, num_heads, dropout=0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        # Fused projection: one matmul produces Q, K and V (concatenated) - 
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, E = x.shape
        q, k, v = self.qkv(x).split(E, dim=-1)

        # [B, T, E] -> [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # is_causal=True applies the autoregressive mask inside the kernel
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # [B, num_heads, T, head_dim] -> [B, T, E]
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        return self.out_proj(out)


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
