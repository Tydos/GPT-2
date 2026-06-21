import argparse
import logging
import torch
from huggingface_hub import hf_hub_download

from src.config import GPU_CONFIG, CPU_CONFIG
from src.gpt_model import GPTModel
from src.tokenizer import TikTokenizer, SimpleTokenizer, build_vocab
from src.generate_text import generate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a pretrained GPT model")
    parser.add_argument("--repo-id", type=str, default="triton329/gpt2",
                    help="HuggingFace repo ID")
    parser.add_argument("--filename", type=str, default="GPT-2/artifacts/model.pth",
                        help="Path to model weights")
    parser.add_argument("--prompt", type=str, default="Once upon a time in India")
    parser.add_argument("--num-tokens", type=int, default=50)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--tokenizer", choices=["tiktoken", "simple"], default="tiktoken")
    # Only needed if --tokenizer=simple
    parser.add_argument("--vocab-file", type=str, default=None,
                        help="Path to vocab.json (required for SimpleTokenizer)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    cfg = GPU_CONFIG if args.gpu else CPU_CONFIG

    # 1. Build tokenizer
    if args.tokenizer == "tiktoken":
        tokenizer = TikTokenizer("gpt2")
    else:
        import json
        with open(args.vocab_file) as f:
            vocab = json.load(f)
        tokenizer = SimpleTokenizer(vocab)
        from dataclasses import replace
        cfg = replace(cfg, vocab_size=len(vocab))

    # 2. Download weights from HuggingFace Hub
    logging.info(f"Downloading weights from {args.repo_id} ...")
    weights_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename)

    # 3. Load model
    model = GPTModel(cfg).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("Model loaded.")

    # 4. Generate
    output = generate(model, tokenizer, device, args.prompt, num_tokens=args.num_tokens)
    print("\n--- Generated Text ---")
    print(output)