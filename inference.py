import argparse
import logging

import torch

from src.model.config import (
    GPT124M_MODEL,
    NANO_MODEL,
    DEFAULT_WEIGHTS_SOURCE,
    WEIGHTS_SOURCE_CHOICES,
)
from src.model.gpt import GPTModel
from src.model.load_weights import load_model
from src.data.tokenizer import BPETokenizer
from src.engine.generate import generate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a pretrained GPT model")
    parser.add_argument(
        "--weights-source",
        choices=WEIGHTS_SOURCE_CHOICES,
        default=DEFAULT_WEIGHTS_SOURCE,
        help="local=artifacts/model.pth, official=openai-community/gpt2",
    )
    parser.add_argument(
    "--config",
    choices=("gpt124m", "nano"),
    default="nano",
    help="gpt124m=124M param GPT-2, nano=tiny model for CPU",
    )
    parser.add_argument("--prompt", type=str, default="Once upon a time in India")
    parser.add_argument("--num-tokens", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = NANO_MODEL if args.config == "nano" else GPT124M_MODEL
    tokenizer = BPETokenizer("gpt2")

    model = load_model(GPTModel(model_cfg).to(device), args.weights_source)
    model.eval()

    ctx = dict(model=model, tokenizer=tokenizer, device=device,
               prompt=args.prompt, num_tokens=args.num_tokens,
               context_length=model_cfg.context_length)

    print("\n\n--- Generated Text ---")
    print("\n Greedy:     ", generate(**ctx, strategy="greedy"))
    print("\n Temperature:", generate(**ctx, strategy="temperature", temperature=0.7))
    print("\n Top-k:      ", generate(**ctx, strategy="top_k", k=50, temperature=0.7))
    print("\n Top-p:      ", generate(**ctx, strategy="top_p", p=0.9, temperature=0.7))
