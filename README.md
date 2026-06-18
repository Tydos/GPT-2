# LLMs from Scratch

Building a GPT-style language model from scratch, following *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

## Model Architecture

Architecture follows GPT-2 proportions scaled to a custom word-level vocab.

| Component | Calculation | Params |
|---|---|---|
| Token embedding | 1078 × 128 | 138,000 |
| Position embedding | 1024 × 128 | 131,072 |
| Output head (linear) | 128 × 1078 | 138,000 |
| MultiHead Attention per block | 4 × (128 × 128) projections | 65,536 |
| Feed Forward Network per block | (128×512) + (512×128) + biases | 131,584 |
| LayerNorm per block | 2 × 2 × 128 | 512 |
| **3 transformer blocks** | 3 × (65,536 + 131,584 + 512) | 593,000 |
| **Total** | | **~951,424** |

## Example

Prompt : "Gisburn had a curious smile in his eyes."

Generated: "gisburn had a curious smile in his eyes . <eos> endless hanging gone hang inflexible genial fact admirers must endless on cleverer ll silver welcome established claimed set hang foreseen"

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
5. Run multi-head causal self-attention (`MultiHeadAttention`) on the embeddings — each head applies a causal mask so position `i` cannot attend to positions `j > i`
6. Pass embeddings through `GPTModel` (token + positional embeddings → 3 transformer blocks → LayerNorm → linear output head) to produce logits over the vocabulary
7. Greedily decode a prompt by repeatedly picking the highest-probability next token
