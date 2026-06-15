import json
import os

from src.config import OUTPUT_DIR, MAX_LEN, STRIDE, BATCH_SIZE, EMBED_DIM
from src.data_utils import download_text, load_text
from src.tokenizer import build_vocab, Tokenizer
from src.dataset import create_gpt_dataloader
from src.embeddings import generate_input_embedding


def main() -> None:
    """
    1. Download the dataset from a remote url
    2. Build a vocab set, and convert dataset into token_ids using the tokenizer class
    3. Create a dataloader to create context->target training samples
    4. Generate input and pos embeddings and sum them
    """
    download_text()
    raw_text = load_text()

    all_words, vocab = build_vocab(raw_text)
    tk = Tokenizer(vocab)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    tokenized_text = tk.encode(raw_text)

    dataloader = create_gpt_dataloader(
        raw_text, tokenizer=tk, max_len=MAX_LEN, stride=STRIDE, batch_size=BATCH_SIZE
    )

    inputs, targets = next(iter(dataloader))
    input_emb = generate_input_embedding(
        inputs[0].tolist(), vocab_size=len(vocab), embed_dim=EMBED_DIM
    )

    print(f"chars={len(raw_text)}  tokens={len(tokenized_text)}  vocab={len(vocab)}")
    print(f"batch  inputs={inputs}  targets={targets}")
    print(f"embed  shape={list(input_emb.shape)}")


if __name__ == "__main__":
    main()
