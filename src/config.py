import os
from dataclasses import dataclass


DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.txt")


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


CPU_CONFIG = GPTConfig(
    vocab_size=50257,
    context_window_size=32,
    context_length=512,
    batch_size=10,
    stride=32,
    embed_dim=64,
    head_dim=32,
    num_heads=2,
    drop_rate=0.0,
    n_layer=1,
    num_epochs=1,
    num_workers=0,
    lr=3e-4,
)

GPU_CONFIG = GPTConfig(
    vocab_size=50257,
    context_window_size=512,
    context_length=1024,
    batch_size=64,
    stride=32,
    embed_dim=768,
    head_dim=64,
    num_heads=12,
    drop_rate=0.0,
    n_layer=12,
    num_epochs=1,
    num_workers=4,
    lr=3e-4,
)
