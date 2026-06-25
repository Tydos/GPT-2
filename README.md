# GPT-2 in PyTorch

A GPT-2 (124M) language model built from scratch in PyTorch. It can train
from scratch, fine-tune from official OpenAI GPT-2 weights, or run inference with
several decoding strategies.

## Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell)
pip install -r requirements.txt
```

## Inference

Runs greedy, temperature, top-k, and top-p sampling and prints all four outputs.

```bash
# Official GPT-2 weights (downloaded automatically on first run)
python inference.py --weights-source official --config gpt124m --prompt "Once upon a time" --num-tokens 50

# Locally trained model
python inference.py --weights-source local --config gpt124m --prompt "Once upon a time"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--weights-source` | `official` | `official` (HF `openai-community/gpt2`), `local` (`artifacts/model.pth`), `scratch` |
| `--config` | `nano` | `gpt124m` = 124M GPT-2, `nano` = tiny CPU model |
| `--prompt` | `"Once upon a time in India"` | Prompt text |
| `--num-tokens` | `50` | Tokens to generate |

## Train

Metrics are logged to [Weights & Biases](https://wandb.ai). Pre-training on WikiText-103 on an A100 takes ~25 min/epoch.

```bash
# Train from scratch on a local text file
python train_model.py --weights-source scratch --dataset-file artifacts/mydata.txt

# Fine-tune from official GPT-2 weights using a HuggingFace dataset
python train_model.py --weights-source official --hf-dataset Salesforce/wikitext
```

| Flag | Default | Description |
|------|---------|-------------|
| `--weights-source` | `official` | `scratch`, `local`, or `official` |
| `--dataset-file PATH` | — | Local `.txt` file (80/10/10 split) — mutually exclusive with `--hf-dataset` |
| `--hf-dataset REPO_ID` | — | HuggingFace dataset repo (e.g. `Salesforce/wikitext`) |
| `--tokenizer` | `tiktoken` | `tiktoken` (GPT-2 BPE) or `simple` (word-level) |
| `--output-dir` | `artifacts/` | Where `model.pth` and loss curve are saved |
| `--sample-prompt` | `"Hello World"` | Prompt sampled after each epoch |

Hyperparameters live in `src/model/config.py`.

## Performance improvements

Training throughput on **NVIDIA A100** (WikiText-103, GPT-2 124M, batch size 32,
context length 1024, bf16 autocast, 5 epochs each).

| Configuration | Throughput |
|---------------|------------|
| Default MHA (`MultiHeadAttention`) | ~40k tok/s |
| Optimised MHA with SDPA (`MultiHeadAttentionSDPA`) | ~90k tok/s |
| SDPA + `torch.compile` | TBD |


## Project layout

```
src/
  model/      # config, attention, transformer block, layernorm, GPT model, weight loading
  data/       # tokenizer (tiktoken), sliding-window dataset, dataset utils
  engine/     # training loop and text-generation strategies
train_model.py  # training entrypoint
inference.py    # generation entrypoint
```
