import os

DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.txt")
CONTEXT_WINDOW_SIZE = 5
STRIDE = 1
BATCH_SIZE = 1
EMBED_DIM = 10
HEAD_DIM = 15  # or D_K in the original attention paper
