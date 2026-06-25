import argparse
import logging
import os
from dataclasses import replace
import torch

from src.model.config import (
    GPT124M_CONFIG, # runs on GPU
    NANO_GPT_CONFIG, # runs on CPU
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_WEIGHTS_SOURCE,
    WEIGHTS_SOURCE_CHOICES,
)
from src.data.utils import load_hf_text, load_text
from src.data.GPTDataset import create_dataloaders
from src.model.gpt import GPTModel
from src.data.tokenizer import TikTokenizer, SimpleTokenizer, build_vocab
from src.model.load_weights import load_model
from src.engine.train import train, build_optimizer_and_scheduler, plot_history
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 on GPU")
    parser.add_argument("--sample-prompt", type=str, default=None)

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Local text file to train on (90/10 train/val split)",
    )
    data_group.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="HuggingFace dataset repo id (e.g. Salesforce/wikitext)",
    )
    parser.add_argument(
    "--tokenizer",
    choices=("tiktoken", "simple"),
    default="tiktoken",
    help="tiktoken=gpt2 BPE tokenizer, simple=word-level (default: tiktoken for --hf-dataset, simple for --dataset-file)",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--weights-source",
        choices=WEIGHTS_SOURCE_CHOICES,
        default=DEFAULT_WEIGHTS_SOURCE,
        help="local=artifacts/model.pth, official=openai-community/gpt2",
    )
    return parser.parse_args()


if __name__ == "__main__":

    # parse the arguments and set the seed
    args = parse_args()
    cfg = NANO_GPT_CONFIG
    torch.manual_seed(NANO_GPT_CONFIG.seed) 
    sample_prompt = args.sample_prompt or "Hello World"

    # loading the data, dataloader and tokenizer
    if args.dataset_file:
        logging.info(f"Loading data from file: {args.dataset_file}")
        raw = load_text(args.dataset_file)
        n = len(raw)
        train_text = raw[:int(0.8 * n)]
        val_text   = raw[int(0.8 * n):int(0.9 * n)]
        test_text  = raw[int(0.9 * n):]
    elif args.hf_dataset:
        logging.info(f"Loading HuggingFace dataset: {args.hf_dataset}")
        train_text, val_text, test_text = load_hf_text(args.hf_dataset) 
    else:
        raise ValueError("Provide either --dataset-file or --hf-dataset")
    
    if args.tokenizer == "simple":
        vocab = build_vocab(raw)
        tokenizer = SimpleTokenizer(vocab)  
        cfg = replace(cfg, vocab_size = len(vocab))
    else:
        tokenizer = TikTokenizer("gpt2")
        cfg = replace(cfg, vocab_size=tokenizer.tokenizer.n_vocab) # 50257

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_text, val_text, test_text,
        tokenizer=tokenizer,
        max_len=cfg.context_window_size,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    logging.info(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    model = load_model(GPTModel(cfg).to(device), args.weights_source)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, train_loader)

    logging.info(f"Starting training for {cfg.num_epochs} epochs...\n")
    history = train(
        model, optimizer, train_loader, val_loader, test_loader,
        device, cfg, sample_prompt, tokenizer, scheduler,
    )
    plot_history(history, args.output_dir)

    torch.save(model.state_dict(), os.path.join(args.output_dir, DEFAULT_MODEL_FILENAME))
    logging.info("Saved model")
