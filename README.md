# LLMs from Scratch 

Tokenization and dataset preparation for GPT-style language models, following the *Build a Large Language Model (From Scratch)* book by Sebastian Raschka.

## Project Structure

```
src/
  data_utils.py   — download and load the-verdict.txt
  tokenizer.py    — vocabulary building and Tokenizer class
  dataset.py      — GPTDataset (PyTorch Dataset) and GPTLoader
  main.py         — demo runner
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd src
python main.py
```

## What It Does

1. Downloads `the-verdict.txt` (a short story used as training data)
2. Builds a character-level vocabulary from the text
3. Encodes the full text into integer token IDs
4. Demonstrates next-token prediction (context → target pairs)
5. Creates a PyTorch `DataLoader` using a sliding-window approach
