# GPT-2 in PyTorch

GPT-style language model built from scratch in PyTorch, trained on a subset of WikiText-103 corpus. Model trained on an NVIDIA A100 GPU — full run completes in ~6 minutes.

## Install

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python main.py --gpu --tokenizer tiktoken
```

## Inference

```bash
python inference.py --prompt "Once upon a time" --gpu
```

Pretrained weights load automatically from `triton329/gpt2` on HuggingFace.

## Example output

```
Once upon a time . At the early 20th century , though it was seen by the time
in 1864 , it lacks a 1804 Antigua – 5 @-@ class submarine of the hurricane...
```