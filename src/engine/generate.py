import torch
import torch.nn.functional as F
from src.data.tokenizer import BaseTokenizer


def generate_greedy(model, tokenizer: BaseTokenizer, device, prompt, num_tokens=20):
    """ Greedy generation - take the token with the highest probability """
    model.eval()
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(num_tokens):
            input_ids = torch.tensor([ids[-1023:]]).to(device) # pick the last 1023 tokens from the prompt
            logits = model(input_ids) # forward pass through the model to get logits
            ids.append(logits[0, -1, :].argmax().item()) # the last row of the logits is the logits for the next token
    return tokenizer.decode(ids)


def generate_temperature(model, tokenizer: BaseTokenizer, device, prompt, num_tokens=20, temperature=0.7):
    """ Temperature sampling - sample from the distribution with a given temperature. """
    model.eval()
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(num_tokens):
            input_ids = torch.tensor([ids[-1023:]]).to(device)
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)  
            ids.append(torch.multinomial(probs, 1).item())
    return tokenizer.decode(ids)

def generate_top_k(model, tokenizer: BaseTokenizer, device, prompt, num_tokens=20, k=50, temperature=0.7):
    """ Top-k sampling - sample from the k most likely tokens. """
    model.eval()
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(num_tokens):
            input_ids = torch.tensor([ids[-1023:]]).to(device)
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature
            topk_logits, topk_indices = torch.topk(next_logits, k)  # Keep only k
            topk_probs = torch.softmax(topk_logits, dim=-1)
            sampled_idx = torch.multinomial(topk_probs, 1).item()
            ids.append(topk_indices[sampled_idx].item())
    return tokenizer.decode(ids)

def generate_top_p(model, tokenizer: BaseTokenizer, device, prompt, num_tokens=20, p=0.9, temperature=0.7):
    """ Top-p sampling - sample from the smallest set of tokens whose cumulative probability exceeds p. """
    model.eval()
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(num_tokens):
            input_ids = torch.tensor([ids[-1023:]]).to(device)
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs > p
            # shift right so the token that crosses the threshold is kept
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum()
            sampled_idx = torch.multinomial(sorted_probs, 1).item()
            ids.append(sorted_indices[sampled_idx].item())
    return tokenizer.decode(ids)