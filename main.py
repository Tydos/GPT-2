import logging
import os
from src.config import (
    OUTPUT_DIR,
    CONTEXT_WINDOW_SIZE,
    STRIDE,
    BATCH_SIZE,
    EMBED_DIM,
    HEAD_DIM,
)
from src.data_utils import download_text, load_text
from src.tokenizer import build_vocab, save_vocab, Tokenizer
from src.dataset import create_gpt_dataloader
from src.embeddings import Embedding
from src.attention import SelfAttention, plot_attention_heatmap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    """
    1. Download the dataset from a remote url
    2. Build a vocab set, and convert dataset into token_ids using the tokenizer class
    3. Create a dataloader to create context->target training samples
    4. Generate input and pos embeddings and sum them
    """
    download_text()
    raw_text = load_text()
    logging.info(f"Loaded text — {len(raw_text)} chars, preview: {raw_text[:40]!r}")

    vocab = build_vocab(raw_text)
    save_vocab(vocab, OUTPUT_DIR)
    logging.info(f"Vocab built — {len(vocab)} unique tokens")

    tokenizer = Tokenizer(vocab)
    token_ids = tokenizer.encode(raw_text)
    logging.info(f"Encoded — {len(token_ids)} tokens, first 10: {token_ids[:10]}")

    dataloader = create_gpt_dataloader(
        raw_text,
        tokenizer=tokenizer,
        max_len=CONTEXT_WINDOW_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
    )
    logging.info(
        f"DataLoader ready — {len(dataloader)} batches (max_len={CONTEXT_WINDOW_SIZE}, stride={STRIDE}, batch_size={BATCH_SIZE})"
    )

    embedding = Embedding(vocab_size=len(vocab), embed_dim=EMBED_DIM)
    attention = SelfAttention(embed_dim=EMBED_DIM, head_dim=HEAD_DIM)
    for inputs, targets in dataloader:
        logging.info(
            f"Sample batch — inputs: {inputs.tolist()}, targets: {targets.tolist()}"
        )
        x = embedding.generate_input_embedding(inputs[0].tolist())
        logging.info(
            f"Embeddings — shape: {list(x.shape)} (input_sequence_length={inputs.shape[1]}, embed_dim={EMBED_DIM})"
        )

        output, weights = attention(x)
        logging.info(f"Attention output — shape: {list(output.shape)}")

        tokens = [tokenizer.decode([t]) for t in inputs[0].tolist()]
        plot_attention_heatmap(
            weights, tokens, os.path.join(OUTPUT_DIR, "attention_heatmap.png")
        )
        break


if __name__ == "__main__":
    main()
