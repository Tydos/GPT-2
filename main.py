import torch
import logging
from src.config import (
    OUTPUT_DIR,
    CONTEXT_WINDOW_SIZE,
    STRIDE,
    BATCH_SIZE,
    EMBED_DIM,
    HEAD_DIM,
    NUM_HEADS,
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    N_LAYER,
)
from src.data_utils import download_text, load_text
from src.tokenizer import build_vocab, save_vocab, Tokenizer
from src.dataset import create_gpt_dataloader
from src.embeddings import Embedding
from src.attention import MultiHeadAttention
from src.gpt_model import GPTModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
torch.manual_seed(22)


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
    coverage_pct = len(vocab) / VOCAB_SIZE * 100
    logging.info(
        f"Vocab — {len(vocab):,} unique tokens | "
        f"coverage: {coverage_pct:.1f}% of GPT-2 vocab ({VOCAB_SIZE:,} tokens)"
    )
    sample_tokens = sorted(vocab.items(), key=lambda x: x[1])[:5]
    logging.info(f"  Sample token→id mappings: {sample_tokens}")

    tokenizer = Tokenizer(vocab)
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
        max_len=CONTEXT_WINDOW_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
    )
    total_samples = len(dataloader) * BATCH_SIZE
    logging.info(
        f"DataLoader — {len(dataloader):,} batches | "
        f"~{total_samples:,} samples total | "
        f"context={CONTEXT_WINDOW_SIZE}, stride={STRIDE}, batch={BATCH_SIZE}"
    )
    logging.info(
        f"  Each sample: {CONTEXT_WINDOW_SIZE} input tokens → {CONTEXT_WINDOW_SIZE} target tokens (shifted by 1)"
    )

    # --- Embedding + MHA sanity check ---
    logging.info("=== Stage 4: Embedding + Multi-Head Attention (sanity check) ===")
    embedding = Embedding(vocab_size=len(vocab), embed_dim=EMBED_DIM)
    embed_params = len(vocab) * EMBED_DIM + CONTEXT_LENGTH * EMBED_DIM
    logging.info(
        f"Embedding layer — vocab_size={len(vocab):,} × embed_dim={EMBED_DIM} "
        f"+ pos_embed: context_len={CONTEXT_LENGTH} × {EMBED_DIM} "
        f"→ {embed_params:,} total parameters"
    )

    mha = MultiHeadAttention(
        embed_dim=EMBED_DIM, head_dim=HEAD_DIM, num_heads=NUM_HEADS
    )
    logging.info(
        f"MultiHeadAttention — {NUM_HEADS} heads × head_dim={HEAD_DIM} "
        f"(total attention dim={NUM_HEADS * HEAD_DIM}) | input embed_dim={EMBED_DIM}"
    )

    for inputs, targets in dataloader:
        logging.info(
            f"  Sample batch — input ids: shape={list(inputs[0].shape)}, min={inputs[0].min()}, max={inputs[0].max()} | "
            f"target ids: shape={list(targets[0].shape)}, min={targets[0].min()}, max={targets[0].max()}"
        )
        x = embedding.generate_input_embedding(inputs[0].tolist())
        logging.info(
            f"  Token + position embeddings — shape: {list(x.shape)} "
            f"(seq_len={inputs.shape[1]}, embed_dim={EMBED_DIM}) "
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
            f"({NUM_HEADS} heads × head_dim={HEAD_DIM}) "
            f"| dtype: {output.dtype}"
        )
        logging.info(
            f"  MHA output stats — mean: {output.mean().item():.4f}, "
            f"std: {output.std().item():.4f}"
        )
        break

    # --- GPT model forward pass ---
    logging.info("=== Stage 5: GPT Model Forward Pass ===")
    model = GPTModel()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024**2)  # float32 = 4 bytes
    logging.info(
        f"GPTModel built — {total_params:,} total params ({trainable_params:,} trainable) | "
        f"~{param_size_mb:.1f} MB (float32)"
    )
    logging.info(
        f"  Architecture: {N_LAYER} transformer blocks | "
        f"embed_dim={EMBED_DIM} | {NUM_HEADS} heads | vocab_size={VOCAB_SIZE}"
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
    ids = list(prompt_ids)
    for step in range(num_generate):
        input_tensor = torch.tensor([ids[-CONTEXT_LENGTH:]])
        with torch.no_grad():
            logits = model(input_tensor)
        next_id = torch.argmax(logits[0, -1, :]).item()
        next_token = tokenizer.decode([next_id])
        ids.append(next_id)
        logging.debug(f"  Step {step + 1:2d}: next_id={next_id} → {next_token!r}")

    generated_tokens = ids[len(prompt_ids) :]
    logging.info(
        f"  Generated {num_generate} tokens: ids={generated_tokens} | "
        f"decoded: {[tokenizer.decode([t]) for t in generated_tokens]}"
    )
    print("Generated:", tokenizer.decode(ids))


if __name__ == "__main__":
    main()
