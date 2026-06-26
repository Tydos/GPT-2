import argparse
import logging
import os
from dataclasses import replace
import torch

from src.model.config import (
    GPT124M_MODEL,
    GPT124M_TRAIN,
    NANO_MODEL,
    NANO_TRAIN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_WEIGHTS_SOURCE,
    WEIGHTS_SOURCE_CHOICES,
)
from src.data.utils import load_hf_text, load_text
from src.data.pretrain import create_dataloaders_from_splits
from src.model.gpt import GPTModel
from src.data.tokenizer import BPETokenizer, SimpleTokenizer
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
        "--config",
        choices=("gpt124m", "nano"),
        default="nano",
        help="gpt124m=124M param GPT-2 (GPU), nano=tiny model (CPU)",
    )
    parser.add_argument(
        "--weights-source",
        choices=WEIGHTS_SOURCE_CHOICES,
        default=DEFAULT_WEIGHTS_SOURCE,
        help="local=artifacts/model.pth, official=openai-community/gpt2",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    if args.config == "gpt124m":
        model_cfg, train_cfg = GPT124M_MODEL, GPT124M_TRAIN
    else:
        model_cfg, train_cfg = NANO_MODEL, NANO_TRAIN

    torch.manual_seed(train_cfg.seed)
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
        raw = train_text + val_text + test_text
    else:
        raise ValueError("Provide either --dataset-file or --hf-dataset")

    if args.tokenizer == "simple":
        tokenizer = SimpleTokenizer.from_text(raw)
        model_cfg = replace(model_cfg, vocab_size=len(tokenizer.str_to_int))
    else:
        tokenizer = BPETokenizer("gpt2")
        model_cfg = replace(model_cfg, vocab_size=tokenizer.tokenizer.n_vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        train_text, val_text, test_text,
        tokenizer,
        model_cfg.context_length,
        train_cfg.stride,
        train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    logging.info(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    model = load_model(GPTModel(model_cfg).to(device), args.weights_source)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_cfg, train_loader)

    logging.info(f"Starting training for {train_cfg.num_epochs} epochs...\n")
    history = train(
        model, optimizer, train_loader, val_loader, test_loader,
        device, model_cfg, train_cfg, sample_prompt, tokenizer, scheduler,
    )
    plot_history(history, args.output_dir)

    torch.save(model.state_dict(), os.path.join(args.output_dir, DEFAULT_MODEL_FILENAME))
    logging.info("Saved model")
