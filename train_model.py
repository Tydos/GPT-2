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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

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

    # parse the arguments and set the seed
    args = parse_args()
    torch.manual_seed(GPT124M_CONFIG.seed) 
    cfg = GPT124M_CONFIG
    sample_prompt = args.sample_prompt or "Gisburn had a evil smile, and"

    # loading the data, dataloader and tokenizer
    logging.info("Loading data...")
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

    # use local model weights or download from huggingface
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
    
    # define the optimizer and scheduler, eta_min is the floor value, start and end factor are 1% to 100% of learning rate
    total_steps = cfg.num_epochs*len(train_loader)
    warmup_steps = min(cfg.warmup_steps, max(1, total_steps -1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps-warmup_steps, eta_min=cfg.lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
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
        scheduler,
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
