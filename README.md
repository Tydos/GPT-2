# GPT-2 in PyTorch

PyTorch implementation of a GPT-style language model built from first principles. Demonstrates transformer internals, tokenization strategies, and end-to-end training + generation.

## Highlights
- Decoder-only transformer (GPT architecture)
- Custom multi-head causal self-attention
- Word-level + GPT-2 BPE tokenization (tiktoken)
- Sliding-window dataset for next-token prediction
- Training loop with AdamW + cross-entropy
- Text generation via greedy decoding
- Outputs: model checkpoints + loss curves

## Architecture
- Token + positional embeddings  
- N × Transformer blocks:
  - Causal self-attention
  - Feed-forward network
  - LayerNorm  
- Linear output head → vocabulary logits

## Configs

| Config     | Dim | Heads | Layers | Vocab  | Use Case      |
|------------|-----|-------|--------|--------|---------------|
| CPU_CONFIG | 64  | 2     | 1      | Dynamic| Debug/local   |
| GPU_CONFIG | 768 | 12    | 12     | 50257  | Full training |

## Pipeline
1. Load dataset  
2. Tokenize (word-level or BPE)  
3. Create context → target pairs  
4. Forward pass → logits  
5. Optimize (AdamW, cross-entropy)  
6. Track loss + save artifacts  
7. Generate sample text per epoch  

## Usage

```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

python main.py
python main.py --tokenizer tiktoken
python main.py --gpu
python main.py --sample-prompt "She walked into the room" --data-url <url>