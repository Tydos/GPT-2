import re
import os
import requests
from datasets import load_dataset


def download_text(url: str, file_path: str, force: bool = True) -> None:
    """Download text from a URL to a file in the directory"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if force or not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)


def load_text(file_path: str) -> str:
    """Load text from a file"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_hf_text(
    name: str,
    config: str | None = None,
    text_column: str = "text",
    train_split: str = "train",
    val_split: str = "validation",
    test_split: str = "test",
) -> tuple[str, str, str]:
    """Load (train_text, val_text, test_text) from a HuggingFace text dataset.

    Falls back to an 80/10/10 split of the train split when val/test are absent.
    """
    dataset = load_dataset(name, config)

    train_text = "\n".join(dataset[train_split][text_column])

    if val_split in dataset and test_split in dataset:
        val_text = "\n".join(dataset[val_split][text_column])
        test_text = "\n".join(dataset[test_split][text_column])
    else:
        n = len(train_text)
        train_text, val_text, test_text = train_text[:int(0.8*n)], train_text[int(0.8*n):int(0.9*n)], train_text[int(0.9*n):]

    return train_text, val_text, test_text


def load_wikitext() -> tuple[str, str, str]:
    """Load and clean WikiText-103 train/val/test splits."""
    train_text, val_text, test_text = load_hf_text("Salesforce/wikitext", "wikitext-103-raw-v1")

    def clean(text: str) -> str:
        text = text.replace("@-@", "-")
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'=\s*[^=]+?\s*=', '', text)
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return '\n'.join(line.strip() for line in text.split('\n') if line.strip())

    return clean(train_text), clean(val_text), clean(test_text)