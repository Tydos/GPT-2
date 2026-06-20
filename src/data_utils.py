import os
import requests
from datasets import load_dataset


def download_text(url: str, file_path: str, force: bool = True) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if force or not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)


def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_wikitext() -> tuple[str, str]:

    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    train_text = "\n".join(dataset["train"]["text"][:3000])
    val_text = "\n".join(dataset["validation"]["text"][:500])

    # Clean up empty lines in the dataset
    train_text = "\n".join(line for line in train_text.splitlines() if line.strip())
    val_text = "\n".join(line for line in val_text.splitlines() if line.strip())

    return train_text, val_text
