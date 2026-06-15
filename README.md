# LLMs from Scratch

Building a GPT-style language model from scratch, following *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

## Project Structure

```
artifacts/
  the-verdict.txt   — downloaded training text
  vocab.json        — saved vocabulary
src/
  config.py         — hyperparameters and paths
  data_utils.py     — download and load text
  tokenizer.py      — vocabulary building and Tokenizer class
  dataset.py        — GPTDataset (sliding-window) and DataLoader factory
  embeddings.py     — token and positional embeddings
main.py             — pipeline runner
requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Pipeline

1. Download `the-verdict.txt` as training data
2. Build a word-level vocabulary and tokenize the text into integer IDs
3. Create a PyTorch `DataLoader` using a sliding-window (context → target pairs)
4. Generate token + positional embeddings and sum them into the model input
