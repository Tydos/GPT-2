import json
import logging
import os
from src.config import OUTPUT_DIR, MAX_LEN, STRIDE, BATCH_SIZE, EMBED_DIM, VOCAB_SIZE
from src.data_utils import download_text, load_text
from src.tokenizer import build_vocab, save_vocab, Tokenizer
from src.dataset import create_gpt_dataloader
from src.embeddings import Embedding

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

    tk = Tokenizer(vocab)
    tokenized_text = tk.encode(raw_text)
    logging.info(
        f"Encoded — {len(tokenized_text)} tokens, first 10: {tokenized_text[:10]}"
    )

    dataloader = create_gpt_dataloader(
        raw_text, tokenizer=tk, max_len=MAX_LEN, stride=STRIDE, batch_size=BATCH_SIZE
    )
    logging.info(
        f"DataLoader ready — {len(dataloader)} batches (max_len={MAX_LEN}, stride={STRIDE}, batch_size={BATCH_SIZE})"
    )

    inputs, targets = next(iter(dataloader))
    logging.info(
        f"Sample batch — inputs: {inputs.tolist()}, targets: {targets.tolist()}"
    )

    embed = Embedding(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    input_emb = embed.generate_input_embedding(inputs[0].tolist())
    logging.info(
        f"Embeddings — shape: {list(input_emb.shape)} (seq_len={inputs.shape[1]}, embed_dim={EMBED_DIM})"
    )


if __name__ == "__main__":
    main()
