import torch
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(123)

sentence = "I love India"

# --- Embeddings ---

enc = tiktoken.get_encoding("gpt2")
token_ids = torch.tensor(enc.encode(sentence))  # (input_sequence_length,)
tokens = sentence.split()
vocab_size = 50257
embed_dim = 10
input_sequence_length = token_ids.shape[0]  # 3 tokens: "I", "love", "India"

token_emb = torch.nn.Embedding(vocab_size, embed_dim)  # (50257, 10)
pos_emb = torch.nn.Embedding(input_sequence_length, embed_dim)  # (3, 10)

inputs = token_emb(token_ids) + pos_emb(torch.arange(input_sequence_length))  # (3, 10)

# --- Weight Matrices ---

head_dim = embed_dim  # query/key dimension

w_query = torch.rand(embed_dim, head_dim)  # (10, 10)
w_key = torch.rand(embed_dim, head_dim)  # (10, 10)
w_value = torch.rand(embed_dim, head_dim)  # (10, 10)

Q = inputs @ w_query
K = inputs @ w_key
V = inputs @ w_value

scores = Q @ K.T
scores = scores / head_dim**0.5
weights = torch.softmax(scores, dim=1)
output = weights @ V

print(output)

# Plot heatmap
plt.figure()
sns.heatmap(
    weights.detach().numpy(),
    annot=True,
    cmap="Blues",
    xticklabels=tokens,
    yticklabels=tokens,
)
plt.xlabel("Key (attended to)")
plt.ylabel("Query (attending)")
plt.title("Self-Attention Weights")
plt.show()
