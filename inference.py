import argparse
import logging
import os
import torch
from huggingface_hub import hf_hub_download

from src.model.config import GPT124M_CONFIG
from src.model.gpt import GPTModel
from src.data.tokenizer import TikTokenizer
from src.engine.generate import generate_greedy,generate_temperature,generate_top_k,generate_top_p

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a pretrained GPT model")
    parser.add_argument("--repo-id", type=str, default="triton329/gpt2",
                        help="HuggingFace repo ID")
    parser.add_argument("--filename", type=str, default="GPT-2/artifacts/model.pth",
                        help="Path to model weights")
    parser.add_argument("--prompt", type=str, default="Once upon a time in India")
    parser.add_argument("--num-tokens", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GPT124M_CONFIG
    tokenizer = TikTokenizer("gpt2")

    local_path = os.path.join("artifacts", "model.pth")
    if os.path.exists(local_path):
        weights_path = local_path
        logging.info(f"Loading local model from {weights_path}")
    else:
        logging.info(f"Downloading weights from {args.repo_id} ...")
        weights_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename)

    model = GPTModel(cfg).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("Model loaded.")

    print("\n\n--- Generated Text ---")
    output_greedy = generate_greedy(model, tokenizer, device, args.prompt, num_tokens=args.num_tokens)
    print("\n Greedy:", output_greedy)
    output_temperature = generate_temperature(model, tokenizer, device, args.prompt, num_tokens=args.num_tokens, temperature=0.7)
    print("\n Temperature:", output_temperature)
    output_top_k = generate_top_k(model, tokenizer, device, args.prompt, num_tokens=args.num_tokens, k=50, temperature=0.7)
    print("\n Top-k:", output_top_k)
    output_top_p = generate_top_p(model, tokenizer, device, args.prompt, num_tokens=args.num_tokens, p=0.9, temperature=0.7)
    print("\n Top-p:", output_top_p)
