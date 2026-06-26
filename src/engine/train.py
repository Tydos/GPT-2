import os
import torch
import torch.nn.functional as F
import logging
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import numpy as np
import wandb
from torch.amp import autocast
from tqdm import tqdm

from src.engine.generate import generate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def calc_cross_entropy_loss(logits, targets):
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

def calc_perplexity(loss: float) -> float:
    return float(np.exp(loss))

def setup_wandb(model_cfg, train_cfg):
    run = wandb.init(
        entity="pjawale-student",
        project="gpt2-pytorch",
        config={**asdict(model_cfg), **asdict(train_cfg)},
    )
    return run

def build_optimizer_and_scheduler(model, train_cfg, train_loader):
    total_steps = train_cfg.num_epochs * len(train_loader)
    warmup_steps = min(train_cfg.warmup_steps, max(1, total_steps - 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=train_cfg.lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    logging.info(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    return optimizer, scheduler


def plot_history(history: np.ndarray, output_dir: str) -> None:
    """Save train/val/test loss curves to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    plt.plot(history[:, 0], label="train")
    plt.plot(history[:, 1], label="val")
    if history.shape[1] > 2:
        plt.plot(history[:, 2], label="test")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


def train(
    model, optimizer, train_loader, val_loader, test_loader, device, model_cfg, train_cfg,
    sample_prompt, tokenizer, scheduler,
):
    run = setup_wandb(model_cfg, train_cfg)

    history = []
    tokens_per_batch = train_cfg.batch_size * model_cfg.context_length
    global_step = 0
    device_type = device.type
    amp_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    try:
        for epoch in range(1, train_cfg.num_epochs + 1):
            model.train()
            train_loss = 0.0
            epoch_start = time.perf_counter()
            window_start = epoch_start
            window_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{train_cfg.num_epochs}",
                unit="batch",
                leave=True,
            )

            for batch_idx, (inputs, targets) in enumerate(pbar, 1):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                with autocast(device_type=device_type, dtype=amp_dtype):
                    loss = calc_cross_entropy_loss(model(inputs), targets)

                loss.backward()

                grad_norm = None
                if train_cfg.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

                optimizer.step()
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                train_loss += loss.item()
                window_batches += 1
                global_step += 1

                if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                    now = time.perf_counter()
                    elapsed = now - window_start
                    tok_s = (window_batches * tokens_per_batch) / elapsed if elapsed > 0 else 0.0
                    window_start = now
                    window_batches = 0

                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        tok_s=f"{tok_s:,.0f}",
                    )

                    log_dict = {
                            "train/batch_loss": loss.item(),
                            "train/batch_perplexity": calc_perplexity(loss.item()),
                            "train/tok_s": tok_s,
                            "epoch": epoch,
                            "train/lr": current_lr,
                    }
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm.item()
                    wandb.log(log_dict, step=global_step)

            epoch_elapsed = time.perf_counter() - epoch_start
            epoch_tokens = len(train_loader) * tokens_per_batch
            epoch_tok_s = epoch_tokens / epoch_elapsed if epoch_elapsed > 0 else 0.0

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    with autocast(device_type=device_type, dtype=amp_dtype):
                        val_loss += calc_cross_entropy_loss(model(inputs), targets).item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            avg_train_perplexity = calc_perplexity(avg_train)
            avg_val_perplexity = calc_perplexity(avg_val)
            sample = generate(model, tokenizer, device, sample_prompt,
                              context_length=model_cfg.context_length)

            logging.info(
                f"Epoch {epoch:2d}/{train_cfg.num_epochs} | train={avg_train:.4f} | val={avg_val:.4f} | "
                f"train_perplexity={avg_train_perplexity:.4f} | val_perplexity={avg_val_perplexity:.4f} | "
                f"{epoch_tok_s:,.0f} tok/s (epoch avg)"
            )

            wandb.log(
                {
                    "train/loss": avg_train,
                    "val/loss": avg_val,
                    "train/perplexity": avg_train_perplexity,
                    "val/perplexity": avg_val_perplexity,
                    "train/tok_s_epoch": epoch_tok_s,
                    "sample": sample,
                    "epoch": epoch,
                },
                step=global_step,
            )
            
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in tqdm(test_loader, desc="Testing", leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    with autocast(device_type=device_type, dtype=amp_dtype):
                        test_loss += calc_cross_entropy_loss(model(inputs), targets).item()
            avg_test = test_loss / len(test_loader)

            history.append((avg_train, avg_val, avg_test))
            logging.info(f"  Sample: {sample}\n")

    finally:
        wandb.finish()

    return np.array(history)