import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.model.config import DEFAULT_OUTPUT_DIR, GPT124M_MODEL, GPT124M_TRAIN
from src.model.gpt import GPTModel
from src.model.load_weights import load_pretrained_weights
from src.data.finetune import create_sms_spam_dataloader
from src.data.tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

NUM_CLASSES = 2
DEFAULT_CLASSIFIER_FILENAME = "spam_classifier.pth"


class SpamClassifier(nn.Module):
    def __init__(self, gpt: GPTModel) -> None:
        super().__init__()
        self.gpt = gpt
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(GPT124M_MODEL.embed_dim, NUM_CLASSES)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            T = tokens.shape[1]
            positions = torch.arange(T, device=tokens.device)
            x = self.gpt.dropout_layer(
                self.gpt.token_embedding(tokens) + self.gpt.position_embedding(positions)
            )
            x = self.gpt.transformer_blocks(x)
            x = self.gpt.final_norm(x)
            last_token = x[:, -1, :]
        return self.classifier(last_token)


def load_model() -> SpamClassifier:
    """Load GPT-2 with frozen weights and a trainable 2-class head."""
    gpt = GPTModel(GPT124M_MODEL)
    state_dict, strict = load_pretrained_weights("official")
    gpt.load_state_dict(state_dict, strict=strict)
    return SpamClassifier(gpt)


def load_data():
    tokenizer = BPETokenizer("gpt2")
    return create_sms_spam_dataloader(
        tokenizer,
        batch_size=GPT124M_TRAIN.batch_size,
        num_workers=GPT124M_TRAIN.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                if GPT124M_TRAIN.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        GPT124M_TRAIN.grad_clip,
                    )
                optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for SMS spam classification")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cfg = GPT124M_TRAIN
    torch.manual_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading data...")
    train_loader, val_loader = load_data()
    logging.info(
        f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
    )

    logging.info("Loading model...")
    model = load_model().to(device)
    if device.type == "cuda":
        model = torch.compile(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"  Trainable params: {trainable_params:,} | device: {device}")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_cfg.lr,
    )

    logging.info(f"Starting fine-tuning for {train_cfg.num_epochs} epochs...\n")
    for epoch in range(1, train_cfg.num_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        logging.info(
            f"Epoch {epoch:2d}/{train_cfg.num_epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, DEFAULT_CLASSIFIER_FILENAME)
    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved model to {save_path}")
