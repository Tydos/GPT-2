import json
import logging
import os

import torch
import torch.nn.functional as F

from src.config import BATCH_SIZE, CONTEXT_WINDOW_SIZE, OUTPUT_DIR, STRIDE, OUTPUT_DIR
from src.config import NUM_EPOCHS
from src.data_utils import download_text, load_text
from src.dataset import create_gpt_dataloader
from src.gpt_model import GPTModel
from src.tokenizer import Tokenizer, build_vocab, save_vocab, TikTokenizer
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 1. Load data
logging.info("Loading data...")
download_text()
raw_text = load_text()
logging.info(f"  {len(raw_text):,} chars loaded")

# 2. Load vocab
# vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
# if os.path.exists(vocab_path):
#     with open(vocab_path) as f:
#         vocab = json.load(f)
#     logging.info(f"  Loaded vocab ({len(vocab):,} tokens)")
# else:
#     vocab = build_vocab(raw_text)
#     save_vocab(vocab, OUTPUT_DIR)
#     logging.info(f"  Built vocab ({len(vocab):,} tokens)")

tokenizer = TikTokenizer("gpt2")

# 3. Create dataloaders (90% train, 10% val)
split = int(len(raw_text) * 0.9)
train_loader = create_gpt_dataloader(
    raw_text[:split],
    tokenizer=tokenizer,
    max_len=CONTEXT_WINDOW_SIZE,
    stride=STRIDE,
    batch_size=BATCH_SIZE,
)
val_loader = create_gpt_dataloader(
    raw_text[split:],
    tokenizer=tokenizer,
    max_len=CONTEXT_WINDOW_SIZE,
    stride=STRIDE,
    batch_size=BATCH_SIZE,
)
logging.info(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# 4. Model + optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
total_params = sum(p.numel() for p in model.parameters())
logging.info(f"  Model: {total_params:,} params | device: {device}")


# 5. Loss function
def calc_loss(logits, targets):
    # batches, tokens, embeddings = logits.shape
    # losses = []
    # for batch in len(batches):
    #     loss = F.cross_entropy(logits[batch], targets[batch])
    #     losses.append(loss)
    # return torch.stack(losses).mean()
    B, T, V = logits.shape
    # flatten (batch, seq_len) into one dimension for cross_entropy
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


# 6. Generate samples from a prompt
def generate(prompt, num_tokens=20):
    model.eval()
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(num_tokens):
            # input context size - 1024 so this model can only have a prompt size of 1023
            # production approach is to have sinusoidal position embeddings
            input_ids = torch.tensor([ids]).to(device)  # (1, seq_len)
            if len(input_ids) > 1023:
                raise ValueError("input_ids greater than model context size")
            logits = model(input_ids)  # (1, seq_len, vocab_size)
            next_id = (
                logits[0, -1, :].argmax().item()
            )  # greedy: pick highest prob token
            ids.append(next_id)
    return tokenizer.decode(ids)


# 7. Train loop
logging.info(f"Starting training for {NUM_EPOCHS} epochs...\n")
history = []
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = calc_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 50 == 0:
            logging.info(
                f"  Epoch {epoch}/{NUM_EPOCHS} | batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}"
            )

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            val_loss += calc_loss(model(inputs), targets).item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    logging.info(
        f"Epoch {epoch:2d}/{NUM_EPOCHS} | train={avg_train:.4f} | val={avg_val:.4f}"
    )
    history.append((avg_train, avg_val))
    logging.info(f"  Sample: {generate('Gisburn had a curious smile')}\n")

# 8. Visualisation
history = np.array(history)  # (epochs,2)
plt.plot(history[:, 0], label="train")
plt.plot(history[:, 1], label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(os.path.join(OUTPUT_DIR, "train_validation_curve.png"))
plt.close


torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pth"))
logging.info("Saved model")
