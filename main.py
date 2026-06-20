import argparse
import logging
import os

import matplotlib.pyplot as plt
import torch

from dataclasses import replace
from src.config import CPU_CONFIG, GPU_CONFIG, OUTPUT_DIR
from src.data_utils import download_text, load_text
from src.dataset import create_gpt_dataloader
from src.gpt_model import GPTModel
from src.tokenizer import build_vocab, SimpleTokenizer, TikTokenizer
from src.train import train

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    """
    Model to use GPU/CPU for training, custom prompts and data urls
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--sample-prompt", type=str, default=None)
    parser.add_argument("--tokenizer", choices=["simple", "tiktoken"], default="simple")
    parser.add_argument("--data-url", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = GPU_CONFIG if args.gpu else CPU_CONFIG
    sample_prompt = args.sample_prompt or "Gisburn had a evil smile, and"

    logging.info("Loading data...")
    if args.data_url:
        download_text(url=args.data_url)
    raw_text = load_text()
    logging.info(f"  {len(raw_text):,} chars loaded")
    tokenizer = None
    if args.tokenizer == "tiktoken":
        tokenizer = TikTokenizer("gpt2")
    else:
        vocab = build_vocab(raw_text)
        tokenizer = SimpleTokenizer(vocab)
        cfg = replace(cfg, vocab_size=len(vocab))

    split = int(len(raw_text) * 0.9)
    train_loader = create_gpt_dataloader(
        raw_text[:split],
        tokenizer=tokenizer,
        max_len=cfg.context_window_size,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    val_loader = create_gpt_dataloader(
        raw_text[split:],
        tokenizer=tokenizer,
        max_len=cfg.context_window_size,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    logging.info(
        f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    logging.info(
        f"  Model: {sum(p.numel() for p in model.parameters()):,} params | device: {device}"
    )

    logging.info(f"Starting training for {cfg.num_epochs} epochs...\n")
    history = train(
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        cfg,
        sample_prompt,
        tokenizer,
    )

    plt.plot(history[:, 0], label="train")
    plt.plot(history[:, 1], label="val")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "train_validation_curve.png"))
    plt.close()

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pth"))
    logging.info("Saved model")
