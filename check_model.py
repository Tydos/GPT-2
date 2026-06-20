import torch
import logging
from src.config import CPU_CONFIG, OUTPUT_DIR
from src.data_utils import download_text, load_text
from src.dataset import create_gpt_dataloader
from src.gpt_model import GPTModel
from src.tokenizer import build_vocab, save_vocab, SimpleTokenizer
from src.embeddings import Embedding
from src.attention import MultiHeadAttention
from src.generate_text import generate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
torch.manual_seed(22)

cfg = CPU_CONFIG


def main() -> None:
    """
    1. Download the dataset from a remote url
    2. Build a vocab set, and convert dataset into token_ids using the tokenizer class
    3. Create a dataloader to create context->target training samples
    4. Generate input and pos embeddings and sum them
    5. Run a forward pass through the GPT model
    6. Greedily generate tokens from a prompt using the untrained model
    """
    # --- Data ---
    logging.info("=== Stage 1: Data Loading ===")
    download_text()
    raw_text = load_text()
    num_words = len(raw_text.split())
    num_sentences = raw_text.count(".") + raw_text.count("!") + raw_text.count("?")
    logging.info(
        f"Loaded text — {len(raw_text):,} chars | ~{num_words:,} words | ~{num_sentences:,} sentences"
    )
    logging.info(f"  Preview (first 80 chars): {raw_text[:80]!r}")

    logging.info("=== Stage 2: Vocabulary & Tokenization ===")
    vocab = build_vocab(raw_text)
    save_vocab(vocab, OUTPUT_DIR)
    coverage_pct = len(vocab) / cfg.vocab_size * 100
    logging.info(
        f"Vocab — {len(vocab):,} unique tokens | "
        f"coverage: {coverage_pct:.1f}% of GPT-2 vocab ({cfg.vocab_size:,} tokens)"
    )
    sample_tokens = sorted(vocab.items(), key=lambda x: x[1])[:5]
    logging.info(f"  Sample token→id mappings: {sample_tokens}")

    tokenizer = SimpleTokenizer(vocab)
    token_ids = tokenizer.encode(raw_text)
    compression_ratio = len(raw_text) / len(token_ids)
    logging.info(
        f"Tokenized — {len(token_ids):,} tokens from {len(raw_text):,} chars "
        f"(~{compression_ratio:.1f} chars/token)"
    )
    logging.info(
        f"  Token id stats — shape: ({len(token_ids)},), min: {min(token_ids)}, "
        f"max: {max(token_ids)}, mean: {sum(token_ids) / len(token_ids):.1f}"
    )

    logging.info("=== Stage 3: DataLoader ===")
    dataloader = create_gpt_dataloader(
        raw_text,
        tokenizer=tokenizer,
        max_len=cfg.context_window_size,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
    )
    total_samples = len(dataloader) * cfg.batch_size
    logging.info(
        f"DataLoader — {len(dataloader):,} batches | "
        f"~{total_samples:,} samples total | "
        f"context={cfg.context_window_size}, stride={cfg.stride}, batch={cfg.batch_size}"
    )
    logging.info(
        f"  Each sample: {cfg.context_window_size} input tokens → {cfg.context_window_size} target tokens (shifted by 1)"
    )

    # --- Embedding + MHA sanity check ---
    logging.info("=== Stage 4: Embedding + Multi-Head Attention (sanity check) ===")
    embedding = Embedding(vocab_size=len(vocab), embed_dim=cfg.embed_dim)
    embed_params = len(vocab) * cfg.embed_dim + cfg.context_length * cfg.embed_dim
    logging.info(
        f"Embedding layer — vocab_size={len(vocab):,} × embed_dim={cfg.embed_dim} "
        f"+ pos_embed: context_len={cfg.context_length} × {cfg.embed_dim} "
        f"→ {embed_params:,} total parameters"
    )

    mha = MultiHeadAttention(
        embed_dim=cfg.embed_dim, head_dim=cfg.head_dim, num_heads=cfg.num_heads
    )
    logging.info(
        f"MultiHeadAttention — {cfg.num_heads} heads × head_dim={cfg.head_dim} "
        f"(total attention dim={cfg.num_heads * cfg.head_dim}) | input embed_dim={cfg.embed_dim}"
    )

    for inputs, targets in dataloader:
        logging.info(
            f"  Sample batch — input ids: shape={list(inputs[0].shape)}, min={inputs[0].min()}, max={inputs[0].max()} | "
            f"target ids: shape={list(targets[0].shape)}, min={targets[0].min()}, max={targets[0].max()}"
        )
        x = embedding.generate_input_embedding(inputs[0].tolist())
        logging.info(
            f"  Token + position embeddings — shape: {list(x.shape)} "
            f"(seq_len={inputs.shape[1]}, embed_dim={cfg.embed_dim}) "
            f"| dtype: {x.dtype} | device: {x.device}"
        )
        logging.info(
            f"  Embedding stats — mean: {x.mean().item():.4f}, "
            f"std: {x.std().item():.4f}, "
            f"min: {x.min().item():.4f}, max: {x.max().item():.4f}"
        )

        output = mha(x)
        logging.info(
            f"  MHA output — shape: {list(output.shape)} "
            f"({cfg.num_heads} heads × head_dim={cfg.head_dim}) "
            f"| dtype: {output.dtype}"
        )
        logging.info(
            f"  MHA output stats — mean: {output.mean().item():.4f}, "
            f"std: {output.std().item():.4f}"
        )
        break

    # --- GPT model forward pass ---
    logging.info("=== Stage 5: GPT Model Forward Pass ===")
    model = GPTModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024**2)
    logging.info(
        f"GPTModel built — {total_params:,} total params ({trainable_params:,} trainable) | "
        f"~{param_size_mb:.1f} MB (float32)"
    )
    logging.info(
        f"  Architecture: {cfg.n_layer} transformer blocks | "
        f"embed_dim={cfg.embed_dim} | {cfg.num_heads} heads | vocab_size={cfg.vocab_size}"
    )

    for inputs, targets in dataloader:
        logits = model(inputs)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top5_probs, top5_ids = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([i.item()]) for i in top5_ids]
        logging.info(
            f"  Forward pass — input shape: {list(inputs.shape)} "
            f"→ logits shape: {list(logits.shape)} "
            f"(batch={logits.shape[0]}, seq_len={logits.shape[1]}, vocab={logits.shape[2]})"
        )
        logging.info(
            f"  Top-5 next-token predictions (untrained): "
            + ", ".join(
                f"{t!r}({p:.3f})" for t, p in zip(top5_tokens, top5_probs.tolist())
            )
        )
        break

    # --- Generation demo ---
    logging.info("=== Stage 6: Greedy Token Generation ===")
    prompt = "Gisburn had a curious smile in his eyes."
    num_generate = 20
    prompt_ids = tokenizer.encode(prompt)
    logging.info(
        f"  Prompt: {prompt!r} | {len(prompt_ids)} prompt tokens | "
        f"generating {num_generate} new tokens"
    )
    logging.info(
        f"  Prompt token ids — shape: ({len(prompt_ids)},), min: {min(prompt_ids)}, max: {max(prompt_ids)}"
    )
    device = torch.device("cpu")
    result = generate(model, tokenizer, device, prompt, num_tokens=num_generate)
    print("Generated:", result)


if __name__ == "__main__":
    main()
