import os
import requests

from src.config import DATA_URL, DATASET_PATH


def download_text(url: str = DATA_URL, file_path: str = DATASET_PATH) -> None:
    """Download a text file if it doesn't already exist."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)


def load_text(file_path: str = DATASET_PATH) -> str:
    """Read and return the contents of a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
