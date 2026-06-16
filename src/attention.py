import torch
import tiktoken

sentence = "I love India"
print(sentence)

# Tokenization
enc = tiktoken.get_encoding("gpt2")
token_ids = enc.encode(sentence)
print(token_ids)

# convert token_id to tensor
token_ids = torch.tensor(token_ids)
print(token_ids)

# token embedding
vocab_size = 50257
embed_dim = 3
embedding = torch.nn.Embedding(vocab_size, embed_dim)

# positional embedding
seq_len = token_ids.shape[0]  # how many words?
pos_embedding = torch.nn.Embedding(seq_len, embed_dim)
pos_ids = torch.arange(seq_len)  # [0,1,...]

inputs = embedding(token_ids) + pos_embedding(pos_ids)
print(inputs)


# Step 1: do the dot product between the query and context vectors
query = inputs[0]
print("q:", query)

attention_scores = torch.zeros(inputs.shape[0])  # 6 zeros
for idx, ele in enumerate(inputs):
    attention_scores[idx] = torch.dot(query, inputs[idx])

print(attention_scores)

# Step 2: softmax normalization (0-1 score)
attn_weights = torch.softmax(attention_scores, dim=0)  # linear

# Step 3: context vector (weighted sum of weights x score)
context_vector = torch.zeros(query.shape)  # 3
for idx, x_i in enumerate(inputs):
    context_vector += attn_weights[idx] * inputs[idx]

print(context_vector)
