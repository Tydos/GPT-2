import os

DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.txt")
CONTEXT_WINDOW_SIZE = 5
STRIDE = 1
VOCAB_SIZE = 1078  # based on the tokenizer
CONTEXT_LENGTH = 1024
BATCH_SIZE = 1
EMBED_DIM = 128
HEAD_DIM = 64  # GPT-2 always uses head_dim=64
NUM_HEADS = 2  # embed_dim / head_dim = 128 / 64
DROP_RATE = 0.1  # dropout rate
N_LAYER = 3  # number of repeated transformers
