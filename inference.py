import argparse
import logging

import torch

from src.model.config import (
    GPT124M_CONFIG,
    NANO_GPT_CONFIG,
    DEFAULT_WEIGHTS_SOURCE,
    WEIGHTS_SOURCE_CHOICES,
)
from src.model.gpt import GPTModel
from src.model.load_weights import load_model
from src.data.tokenizer import TikTokenizer
from src.engine.generate import (
    generate_greedy,
    generate_temperature,
    generate_top_k,
    generate_top_p,
)

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
    cfg = NANO_GPT_CONFIG if args.config == "nano" else GPT124M_CONFIG
    tokenizer = TikTokenizer("gpt2")

    model = load_model(GPTModel(cfg).to(device), args.weights_source)
    model.eval()

    print("\n\n--- Generated Text ---")
    output_greedy = generate_greedy(
        model, tokenizer, device, args.prompt, num_tokens=args.num_tokens
    )
    print("\n Greedy:", output_greedy)
    output_temperature = generate_temperature(
        model, tokenizer, device, args.prompt, num_tokens=args.num_tokens, temperature=0.7
    )
    print("\n Temperature:", output_temperature)
    output_top_k = generate_top_k(
        model,
        tokenizer,
        device,
        args.prompt,
        num_tokens=args.num_tokens,
        k=50,
        temperature=0.7,
    )
    print("\n Top-k:", output_top_k)
    output_top_p = generate_top_p(
        model,
        tokenizer,
        device,
        args.prompt,
        num_tokens=args.num_tokens,
        p=0.9,
        temperature=0.7,
    )
    print("\n Top-p:", output_top_p)
