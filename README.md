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

Generates text with greedy, temperature, top-k, and top-p sampling.

```bash
python inference.py --prompt "Once upon a time" --num-tokens 50
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--weights-source` | `official` | `official` (downloads `openai-community/gpt2`), `local` (`artifacts/model.pth`), or `scratch` (random init — outputs gibberish) |
| `--prompt` | `"Once upon a time in India"` | Prompt text |
| `--num-tokens` | `50` | Number of tokens to generate |

Official GPT-2 weights are downloaded automatically from
[`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) on first use.

## Train

Training runs on WikiText-103 and the model was pre-trained on Nvidia A100 with each epoch taking ~25 minutes. Metrics are logged to [Weights & Biases](https://wandb.ai).


**Pretrain from scratch** (random GPT-2 initialization):

```bash
python train_model.py --weights-source scratch
```

**Continue training / fine-tune from official GPT-2 weights:**

```bash
python train_model.py --weights-source official
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--weights-source` | `official` | `scratch`, `local`, or `official` |
| `--sample-prompt` | built-in prompt | Prompt sampled each epoch  |
| `--output-dir` | `artifacts/` | Where the trained model and loss curve are saved |
| `--data-url` / `--dataset-path` | see `config.py` | Dataset source overrides |

Outputs are written to `artifacts/`: the trained weights (`model.pth`) and the
train/validation loss curve (`train_validation_curve.png`).

Hyperparameters (model size, context length, batch size, LR, warmup, etc.) live
in `GPT124M_CONFIG` in `src/model/config.py`.

## Project layout

```
src/
  model/      # config, attention, transformer block, layernorm, GPT model, weight loading
  data/       # tokenizer (tiktoken), sliding-window dataset, dataset utils
  engine/     # training loop and text-generation strategies
train_model.py  # training entrypoint
inference.py    # generation entrypoint
```
