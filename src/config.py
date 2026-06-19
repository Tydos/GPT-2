import os

DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.txt")
CONTEXT_WINDOW_SIZE = 64  # how many tokens are sent for training
VOCAB_SIZE = 1078  # based on the tokenizer
CONTEXT_LENGTH = 64  # how much back can my model read
BATCH_SIZE = 10  # pytorch dataloader batching
STRIDE = 1  # overlap between windows
EMBED_DIM = 64  # number of dimensions
HEAD_DIM = 32  # GPT-2 always uses head_dim=64
NUM_HEADS = 2  # embed_dim / head_dim = 128 / 64
DROP_RATE = 0.1  # dropout rate
N_LAYER = 1  # number of repeated transformers
