import os
import requests


DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_FILE = os.path.join(ARTIFACTS_DIR, "the-verdict.txt")


def download_text(url: str = DATA_URL, file_path: str = DEFAULT_FILE) -> None:
    """Download a text file if it doesn't already exist."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)


def load_text(file_path: str = DEFAULT_FILE) -> str:
    """Read and return the contents of a text file."""
    with open(file_path, "r") as f:
        return f.read()
