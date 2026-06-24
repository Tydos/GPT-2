import argparse
import os
import re

import requests
import torch
import torch.nn.functional as F

from src.data.dataset import GPT2_EOS_TOKEN_ID
from src.data.tokenizer import TikTokenizer
from src.model.config import DEFAULT_OUTPUT_DIR
from fine_tune_model import DEFAULT_CLASSIFIER_FILENAME, load_model

MAX_LEN = 256
LABELS = ("ham", "spam")


def download_from_drive(url: str, path: str) -> None:
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url) or re.search(r"id=([a-zA-Z0-9_-]+)", url)
    file_id = match.group(1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    session = requests.Session()
    r = session.get("https://docs.google.com/uc?export=download", params={"id": file_id}, stream=True)
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            r = session.get("https://docs.google.com/uc?export=download", params={"id": file_id, "confirm": v})
            break
    with open(path, "wb") as f:
        f.write(r.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-url", required=True, help="Google Drive link to spam_classifier.pth")
    parser.add_argument("--prompt", required=True, help="SMS text to classify")
    args = parser.parse_args()

    weights_path = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_CLASSIFIER_FILENAME)
    download_from_drive(args.drive_url, weights_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    tokenizer = TikTokenizer("gpt2")
    tokens = tokenizer.encode(args.prompt)[:MAX_LEN]
    tokens += [GPT2_EOS_TOKEN_ID] * (MAX_LEN - len(tokens))
    inputs = torch.tensor([tokens], device=device)

    with torch.no_grad():
        probs = F.softmax(model(inputs), dim=1)[0]
    label = LABELS[int(probs.argmax())]
    print(f"{label} ({probs.max():.1%})")
