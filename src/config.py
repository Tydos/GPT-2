import os

DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.txt")
CONTEXT_WINDOW_SIZE = 5
STRIDE = 1
VOCAB_SIZE = 50257  # based on the tokenizer
CONTEXT_LENGTH = 1024
BATCH_SIZE = 1
EMBED_DIM = 10
HEAD_DIM = 5  # each head working on 5 dims, D_K in the original attention paper
NUM_HEADS = 2  # total embed_dim 5*2 = 10
DROP_RATE = 0.1  # dropout rate
N_LAYER = 1  # number of repeated transformers
