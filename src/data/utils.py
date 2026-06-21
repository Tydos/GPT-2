import re
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

    train_text = "\n".join(dataset["train"]["text"])
    val_text = "\n".join(dataset["validation"]["text"])

    train_text = clean_wikitext(train_text)
    val_text = clean_wikitext(val_text)

    return train_text, val_text

def clean_wikitext(text: str) -> str:
    """Clean WikiText-103 raw text artifacts."""
    
    # Remove "@-@" (hyphenation markers in WikiText)
    text = text.replace("@-@", "-")
    
    # Remove other @ markers if present (rare in WikiText but possible)
    text = re.sub(r'@\S+', '', text)
    
    # Remove Wikipedia markup artifacts
    text = re.sub(r'=\s*[^=]+?\s*=', '', text)  # Section headers like "= Title ="
    
    # Remove empty parentheses and brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove lines that are just whitespace or empty
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    return '\n'.join(lines)