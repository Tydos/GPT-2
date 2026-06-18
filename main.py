import torch
import logging
from src.config import (
    OUTPUT_DIR,
    CONTEXT_WINDOW_SIZE,
    STRIDE,
    BATCH_SIZE,
    EMBED_DIM,
    HEAD_DIM,
    NUM_HEADS,
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    N_LAYER,
)
from src.data_utils import download_text, load_text
from src.tokenizer import build_vocab, save_vocab, Tokenizer
from src.dataset import create_gpt_dataloader
from src.embeddings import Embedding
from src.attention import MultiHeadAttention
from src.gpt_model import GPTModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    """
    1. Download the dataset from a remote url
    2. Build a vocab set, and convert dataset into token_ids using the tokenizer class
    3. Create a dataloader to create context->target training samples
    4. Generate input and pos embeddings and sum them
    5. Run a forward pass through the GPT model
    6. Greedily generate tokens from a prompt using the untrained model
    """
    # --- Data ---
    download_text()
    raw_text = load_text()
    logging.info(f"Loaded text — {len(raw_text):,} chars | preview: {raw_text[:40]!r}")

    vocab = build_vocab(raw_text)
    save_vocab(vocab, OUTPUT_DIR)
    logging.info(
        f"Vocab — {len(vocab):,} unique tokens (coverage: {len(vocab) / VOCAB_SIZE * 100:.1f}% of GPT-2 vocab)"
    )

    tokenizer = Tokenizer(vocab)
    token_ids = tokenizer.encode(raw_text)
    logging.info(
        f"Tokenized — {len(token_ids):,} tokens | first 10 ids: {token_ids[:10]}"
    )

    dataloader = create_gpt_dataloader(
        raw_text,
        tokenizer=tokenizer,
        max_len=CONTEXT_WINDOW_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
    )
    logging.info(
        f"DataLoader — {len(dataloader):,} batches "
        f"(context={CONTEXT_WINDOW_SIZE}, stride={STRIDE}, batch={BATCH_SIZE})"
    )

    # --- Embedding + MHA sanity check ---
    embedding = Embedding(vocab_size=len(vocab), embed_dim=EMBED_DIM)
    mha = MultiHeadAttention(
        embed_dim=EMBED_DIM, head_dim=HEAD_DIM, num_heads=NUM_HEADS
    )
    for inputs, targets in dataloader:
        logging.info(
            f"Sample batch — input ids: {inputs[0].tolist()} → target ids: {targets[0].tolist()}"
        )
        x = embedding.generate_input_embedding(inputs[0].tolist())
        logging.info(
            f"Token + position embeddings — shape: {list(x.shape)} "
            f"(seq_len={inputs.shape[1]}, embed_dim={EMBED_DIM})"
        )
        output = mha(x)
        logging.info(
            f"Multi-head attention output — shape: {list(output.shape)} "
            f"({NUM_HEADS} heads × head_dim={HEAD_DIM})"
        )
        break

    # --- GPT model forward pass ---
    model = GPTModel()
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(
        f"GPTModel built — {total_params:,} params | "
        f"{N_LAYER} transformer blocks | embed_dim={EMBED_DIM} | vocab_size={VOCAB_SIZE}"
    )

    for inputs, targets in dataloader:
        logits = model(inputs)
        logging.info(
            f"Forward pass — logits shape: {list(logits.shape)} "
            f"(batch={logits.shape[0]}, seq_len={logits.shape[1]}, vocab={logits.shape[2]})"
        )
        break

    # --- Generation demo ---
    prompt = "Gisburn had a curious smile in his eyes."
    logging.info(
        f"Generation — prompt: {prompt!r} | generating {len(prompt) + 1} tokens"
    )
    ids = tokenizer.encode(prompt)
    for _ in range(20):
        input_tensor = torch.tensor([ids[-CONTEXT_LENGTH:]])
        with torch.no_grad():
            logits = model(input_tensor)
        next_id = torch.argmax(logits[0, -1, :]).item()
        ids.append(next_id)
    print("Generated:", tokenizer.decode(ids))


if __name__ == "__main__":
    main()
