import os

DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.txt")
MAX_LEN = 3
STRIDE = 1
BATCH_SIZE = 1
EMBED_DIM = 256
VOCAB_SIZE = None
