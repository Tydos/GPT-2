import torch
from src.tokenizer import BaseTokenizer


def generate(model, tokenizer: BaseTokenizer, device, prompt, num_tokens=20):
    model.eval()
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(num_tokens):
            input_ids = torch.tensor([ids[-1023:]]).to(device)
            logits = model(input_ids)
            ids.append(logits[0, -1, :].argmax().item())
    return tokenizer.decode(ids)
