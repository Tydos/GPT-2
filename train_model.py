import argparse
import logging
import os

import matplotlib.pyplot as plt
import torch

from src.model.config import (
    GPT124M_CONFIG,
    DEFAULT_DATA_URL,
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_DIR,
)
from src.data.utils import download_text, load_text, load_wikitext
from src.data.dataset import create_gpt_dataloader
from src.model.gpt import GPTModel
from src.data.tokenizer import TikTokenizer
from src.engine.train import train
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 on GPU")
    parser.add_argument("--sample-prompt", type=str, default=None)
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", type=str, default="triton329/gpt2")
    parser.add_argument("--filename", type=str, default="GPT-2/artifacts/model.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = GPT124M_CONFIG
    sample_prompt = args.sample_prompt or "Gisburn had a evil smile, and"

    logging.info("Loading data...")
    download_text(url=args.data_url, file_path=args.dataset_path)
    raw_text = load_text(file_path=args.dataset_path)
    logging.info(f"  {len(raw_text):,} chars loaded")

    train_text, val_text = load_wikitext()
    logging.info(f"train_text:{train_text[:10]}")

    tokenizer = TikTokenizer("gpt2")

    train_loader = create_gpt_dataloader(
        train_text,
        tokenizer=tokenizer,
        max_len=cfg.context_window_size,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = create_gpt_dataloader(
        val_text,
        tokenizer=tokenizer,
        max_len=cfg.context_window_size,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    logging.info(
        f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
    )

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found. CPU training is not supported.")

    model = GPTModel(cfg).to(device)
    local_path = os.path.join(args.output_dir, "model.pth")
    if os.path.exists(local_path):
        weights_path = local_path
        logging.info(f"Loading local weights from {weights_path}")
    else:
        logging.info(f"Downloading weights from {args.repo_id} ...")
        weights_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    logging.info("Loaded pretrained weights")
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
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, "train_validation_curve.png"))
    plt.close()

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))
    logging.info("Saved model")
