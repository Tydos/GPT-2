import json
import os

from data_utils import download_text, load_text, ARTIFACTS_DIR
from tokenizer import build_vocab, Tokenizer
from dataset import GPTLoader


def main() -> None:
    # Download and load the text
    download_text()
    raw_text = load_text()
    print(f"Total characters: {len(raw_text)}")

    # Build vocabulary and tokenizer
    all_words, vocab = build_vocab(raw_text)
    print(f"Total unique tokens (vocab size): {len(vocab)}")

    tk = Tokenizer(vocab)

    # Save vocab
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    # Encode the full text
    tokenized_text = tk.encode(raw_text)
    print(f"Total tokens after encoding: {len(tokenized_text)}")
    print(f"First 10 token IDs: {tokenized_text[:10]}")

    # Demonstrate context -> target masking (next-token prediction)
    print("\nContext -> Target examples:")
    for i in range(5):
        context = tokenized_text[:i]
        target = tokenized_text[i + 1]
        print(f"  Context: {tk.decode(context)!r:30s} -> {tk.decode([target])!r}")

    # DataLoader demo: one batch
    dataloader = GPTLoader(raw_text, tokenizer=tk, max_len=3, stride=1, batch_size=1)
    for inputs, targets in dataloader:
        print(f"\nSample batch — inputs: {inputs}, targets: {targets}")
        break


if __name__ == "__main__":
    main()
