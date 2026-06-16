import torch
import tiktoken

torch.manual_seed(123)

sentence = "I love India"

# --- Embeddings ---

enc = tiktoken.get_encoding("gpt2")
token_ids = torch.tensor(enc.encode(sentence))  # (input_sequence_length,)

vocab_size = 50257
embed_dim = 10
input_sequence_length = token_ids.shape[0]  # 3 tokens: "I", "love", "India"

token_emb = torch.nn.Embedding(vocab_size, embed_dim)  # (50257, 10)
pos_emb = torch.nn.Embedding(input_sequence_length, embed_dim)  # (3, 10)

inputs = token_emb(token_ids) + pos_emb(torch.arange(input_sequence_length))  # (3, 10)

# --- Simple self-attention (single query) ---

query = inputs[0]  # (3,) — context vector for token 0

attn_scores = torch.zeros(input_sequence_length)  # (3,)
for idx in range(input_sequence_length):
    attn_scores[idx] = torch.dot(query, inputs[idx])  # scalar dot product

attn_weights = torch.softmax(attn_scores, dim=0)  # (3,) — sums to 1

context_vector = torch.zeros(embed_dim)  # (3,)
for idx in range(input_sequence_length):
    context_vector += attn_weights[idx] * inputs[idx]  # weighted sum → (3,)

print("Context vector (single query):", context_vector)

# --- Self-attention (all tokens, matmul) ---

attn_scores = inputs @ inputs.T  # (3, 3) — score for every token pair
attn_weights = torch.softmax(attn_scores, dim=1)  # (3, 3) — each row sums to 1
context_vectors = attn_weights @ inputs  # (3, 3) — one context vector per token

print("Context vectors (all tokens):\n", context_vectors)
