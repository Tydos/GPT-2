import torch
from src.model.config import GPT124M_CONFIG, DEFAULT_DATA_URL, DEFAULT_DATASET_PATH
from src.data.utils import download_text, load_text
from src.data.dataset import create_gpt_dataloader
from src.model.gpt import GPTModel
from src.data.tokenizer import TikTokenizer
from src.engine.generate import generate_greedy


def main():
    torch.manual_seed(22)
    cfg = GPT124M_CONFIG

    print("Downloading/loading text...")
    download_text(DEFAULT_DATA_URL, DEFAULT_DATASET_PATH)
    raw_text = load_text(DEFAULT_DATASET_PATH)
    print(f"Loaded text: {len(raw_text)} characters")

    tokenizer = TikTokenizer("gpt2")

    num_workers = 0  # Override for Windows compatibility

    dataloader = create_gpt_dataloader(
        raw_text,
        tokenizer=tokenizer,
        max_len=cfg.context_window_size,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=num_workers
    )
    print(f"Dataset size: {len(dataloader.dataset)} samples")

    # Show one train/target sample
    sample_inputs, sample_targets = next(iter(dataloader))
    print(f"\n=== Sample Batch (first sequence) ===")
    print(f"Input text:  {tokenizer.decode(sample_inputs[0].tolist())[:100]}...")
    print(f"Target text: {tokenizer.decode(sample_targets[0].tolist())[:100]}...")

    model = GPTModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nGPTModel — {total_params:,} params")

    # Group summary
    print("\n=== Parameter Budget by Component ===")
    groups = {
        "token_embedding": 0,
        "position_embedding": 0,
        "attention (W_q/k/v)": 0,
        "feed_forward": 0,
        "layer_norm": 0,
        "output_head": 0,
    }
    for name, param in model.named_parameters():
        n = param.numel()
        if "token_embedding" in name:
            groups["token_embedding"] += n
        elif "position_embedding" in name:
            groups["position_embedding"] += n
        elif "W_query" in name or "W_key" in name or "W_value" in name:
            groups["attention (W_q/k/v)"] += n
        elif "ff" in name:
            groups["feed_forward"] += n
        elif "norm" in name:
            groups["layer_norm"] += n
        elif "output_head" in name:
            groups["output_head"] += n

    for group, count in groups.items():
        pct = count / total_params * 100
        print(f"  {group:<25} {count:>10,}  ({pct:.1f}%)")

    # --- Forward pass ---
    print()
    inputs, targets = next(iter(dataloader))
    print(f"Batch shape: inputs={inputs.shape}, targets={targets.shape}")
    logits = model(inputs)
    print(f"Forward pass — input: {list(inputs.shape)} → logits: {list(logits.shape)}")

    # --- Generation ---
    prompt = "Gisburn had a curious smile in his eyes"
    result = generate_greedy(model, tokenizer, torch.device("cpu"), prompt, num_tokens=20)
    print("Generated:", result)


if __name__ == '__main__':
    main()
