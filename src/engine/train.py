import torch
import torch.nn.functional as F
import logging
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from src.engine.generate import generate_greedy

# Enable TF32 for faster matmul
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
scaler = GradScaler()


def calc_loss(logits, targets):
    """Cross-entropy loss over a batch."""
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

def train(
    model, optimizer, train_loader, val_loader, device, cfg, sample_prompt, tokenizer
):
    scaler = GradScaler()  
    history = []
    
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # MIXED PRECISION FORWARD
            with autocast(dtype=torch.bfloat16):
                loss = calc_loss(model(inputs), targets)
            
            # MIXED PRECISION BACKWARD
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

            if batch_idx % 50 == 0:
                logging.info(f"  Epoch {epoch}/{cfg.num_epochs} | batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast(dtype=torch.bfloat16):  # ADD FOR CONSISTENCY
                    val_loss += calc_loss(model(inputs), targets).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch:2d}/{cfg.num_epochs} | train={avg_train:.4f} | val={avg_val:.4f}")
        history.append((avg_train, avg_val))
        logging.info(f"  Sample: {generate_greedy(model, tokenizer, device, sample_prompt)}\n")

    return np.array(history)