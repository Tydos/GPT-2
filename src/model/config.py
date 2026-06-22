import os
from dataclasses import dataclass

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "dataset.txt")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int 
    context_window_size: int
    context_length: int
    batch_size: int
    stride: int
    embed_dim: int
    head_dim: int
    num_heads: int
    drop_rate: float
    n_layer: int
    num_epochs: int
    num_workers: int
    lr: float
    temperature: float
    seed: int

    # optimiser 
    warmup_steps: int
    grad_clip: float





GPT124M_CONFIG = GPTConfig(
    vocab_size=50257,
    context_window_size=256,
    context_length=1024, 
    batch_size=64,
    stride=256,
    embed_dim=768,
    head_dim=64,
    num_heads=12,
    drop_rate=0.0,
    n_layer=12,
    num_epochs=1,
    num_workers=4, 
    lr=3e-4,
    temperature=1.0,
    seed=22,
    warmup_steps=500,   
    grad_clip=1.0,
)
