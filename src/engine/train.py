import torch
import torch.nn.functional as F
import logging
import time
from dataclasses import asdict

import numpy as np
import wandb
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.engine.generate import generate_greedy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def calc_cross_entropy_loss(logits, targets):
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

def calc_perplexity(loss: float) -> float:
    return float(np.exp(loss))

def setup_wandb(cfg):
    run = wandb.init(
        entity="pjawale-student",
        project="gpt2-pytorch",
        config=asdict(cfg),
    )
    return run

def train(
    model, optimizer, train_loader, val_loader, device, cfg, sample_prompt, tokenizer, 
    scheduler,
):
    run = setup_wandb(cfg)

    scaler = GradScaler()
    history = []
    tokens_per_batch = cfg.batch_size * cfg.context_window_size
    global_step = 0

    try:
        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            train_loss = 0.0
            epoch_start = time.perf_counter()
            window_start = epoch_start
            window_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{cfg.num_epochs}",
                unit="batch",
                leave=True,
            )

            for batch_idx, (inputs, targets) in enumerate(pbar, 1):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                with autocast(dtype=torch.bfloat16):
                    loss = calc_cross_entropy_loss(model(inputs), targets)

                scaler.scale(loss).backward()
                
                grad_norm = None
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                
                scaler.step(optimizer)
                scaler.update()
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
                    with autocast(dtype=torch.bfloat16):
                        val_loss += calc_cross_entropy_loss(model(inputs), targets).item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            avg_train_perplexity = calc_perplexity(avg_train)
            avg_val_perplexity = calc_perplexity(avg_val)
            sample = generate_greedy(model, tokenizer, device, sample_prompt)

            logging.info(
                f"Epoch {epoch:2d}/{cfg.num_epochs} | train={avg_train:.4f} | val={avg_val:.4f} | "
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
            
            history.append((avg_train, avg_val))
            logging.info(f"  Sample: {sample}\n")

    finally:
        wandb.finish()

    return np.array(history)