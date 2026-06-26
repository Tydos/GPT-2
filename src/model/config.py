import os
from dataclasses import dataclass

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "artifacts", "dataset.txt"
)
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
DEFAULT_MODEL_FILENAME = "model.pth"
DEFAULT_LOCAL_WEIGHTS_PATH = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_MODEL_FILENAME)
DEFAULT_WEIGHTS_SOURCE = "official"
WEIGHTS_SOURCE_CHOICES = ("scratch", "local", "official")

OFFICIAL_GPT2_REPO = "openai-community/gpt2"
OFFICIAL_GPT2_SAFETENSORS = "model.safetensors"
OFFICIAL_GPT2_PYTORCH = "pytorch_model.bin"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    embed_dim: int
    num_heads: int
    n_layer: int
    drop_rate: float

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass(frozen=True)
class TrainConfig:
    num_epochs: int
    batch_size: int
    stride: int
    num_workers: int
    lr: float
    warmup_steps: int
    grad_clip: float
    seed: int
    temperature: float


GPT124M_MODEL = ModelConfig(
    vocab_size=50257,
    context_length=1024,
    embed_dim=768,
    num_heads=12,
    n_layer=12,
    drop_rate=0.1,
)

GPT124M_TRAIN = TrainConfig(
    num_epochs=5,
    batch_size=16,
    stride=1024,
    num_workers=4,
    lr=3e-4,
    warmup_steps=500,
    grad_clip=1.0,
    seed=22,
    temperature=1.0,
)

NANO_MODEL = ModelConfig(
    vocab_size=50257,
    context_length=256,
    embed_dim=128,
    num_heads=2,
    n_layer=4,
    drop_rate=0.1,
)

NANO_TRAIN = TrainConfig(
    num_epochs=25,
    batch_size=32,
    stride=128,
    num_workers=0,
    lr=3e-4,
    warmup_steps=20,
    grad_clip=1.0,
    seed=22,
    temperature=0.8,
)
