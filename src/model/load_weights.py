import os
from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from src.model.config import (
    DEFAULT_LOCAL_WEIGHTS_PATH,
    GPT124M_CONFIG,
    OFFICIAL_GPT2_PYTORCH,
    OFFICIAL_GPT2_REPO,
    OFFICIAL_GPT2_SAFETENSORS,
)


def _download_hf_state(repo_id: str, filename: str) -> dict[str, torch.Tensor]:
    """ Download weights from Hugging Face """
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    if filename.endswith(".safetensors"):
        return load_safetensors(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _normalize_hf_state(hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Enable compatibility with different checkpoint layouts."""
    if "transformer.wte.weight" in hf_state:
        return hf_state
    if "wte.weight" not in hf_state:
        sample = list(hf_state.keys())[:5]
        raise KeyError(f"Unrecognized HF checkpoint layout. Sample keys: {sample}")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in hf_state.items():
        if key in ("wte.weight", "wpe.weight") or key.startswith(("h.", "ln_f.")):
            normalized[f"transformer.{key}"] = value
    return normalized


def convert_hf_to_gptmodel(
    hf_state: dict[str, torch.Tensor],
    n_layer: int = 12,
    num_heads: int = 12,
    head_dim: int = 64,
) -> dict[str, torch.Tensor]:
    """Convert Hugging Face checkpoint to GPTModel state dictionary."""
    hf_state = _normalize_hf_state(hf_state)
    state: dict[str, torch.Tensor] = {}

    state["token_embedding.weight"] = hf_state["transformer.wte.weight"]
    state["position_embedding.weight"] = hf_state["transformer.wpe.weight"]
    state["final_norm.scale"] = hf_state["transformer.ln_f.weight"]
    state["final_norm.shift"] = hf_state["transformer.ln_f.bias"]

    for i in range(n_layer):
        p = f"transformer.h.{i}"
        b = f"transformer_blocks.{i}"

        state[f"{b}.norm1.scale"] = hf_state[f"{p}.ln_1.weight"]
        state[f"{b}.norm1.shift"] = hf_state[f"{p}.ln_1.bias"]
        state[f"{b}.norm2.scale"] = hf_state[f"{p}.ln_2.weight"]
        state[f"{b}.norm2.shift"] = hf_state[f"{p}.ln_2.bias"]

        # GPT-2 stores QKV fused via a Conv1D (weight is [in, out]); our fused
        # nn.Linear expects [out, in], hence the transpose.
        state[f"{b}.attn.qkv.weight"] = hf_state[f"{p}.attn.c_attn.weight"].T.contiguous()
        state[f"{b}.attn.qkv.bias"] = hf_state[f"{p}.attn.c_attn.bias"]
        state[f"{b}.attn.out_proj.weight"] = hf_state[f"{p}.attn.c_proj.weight"].T.contiguous()
        state[f"{b}.attn.out_proj.bias"] = hf_state[f"{p}.attn.c_proj.bias"]

        state[f"{b}.ff.net.0.weight"] = hf_state[f"{p}.mlp.c_fc.weight"].T.contiguous()
        state[f"{b}.ff.net.0.bias"] = hf_state[f"{p}.mlp.c_fc.bias"]
        state[f"{b}.ff.net.2.weight"] = hf_state[f"{p}.mlp.c_proj.weight"].T.contiguous()
        state[f"{b}.ff.net.2.bias"] = hf_state[f"{p}.mlp.c_proj.bias"]

    state["output_head.weight"] = state["token_embedding.weight"]
    return state


def load_pretrained_weights(
    source: Literal["scratch", "local", "official"],
) -> tuple[dict[str, torch.Tensor], bool]:
    """Returns (state_dict, strict)."""
    if source == "local":
        if not os.path.exists(DEFAULT_LOCAL_WEIGHTS_PATH):
            raise FileNotFoundError(f"No local weights at {DEFAULT_LOCAL_WEIGHTS_PATH}")
        return torch.load(DEFAULT_LOCAL_WEIGHTS_PATH, map_location="cpu", weights_only=True), False

    try:
        hf_state = _download_hf_state(OFFICIAL_GPT2_REPO, OFFICIAL_GPT2_SAFETENSORS)
    except Exception:
        hf_state = _download_hf_state(OFFICIAL_GPT2_REPO, OFFICIAL_GPT2_PYTORCH)

    cfg = GPT124M_CONFIG
    state = convert_hf_to_gptmodel(
        hf_state,
        n_layer=cfg.n_layer,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
    )
    return state, True


def load_model(
    model: torch.nn.Module,
    source: Literal["scratch", "local", "official"],
) -> torch.nn.Module:
    """Load weights, compile, and return the model."""
    import logging
    if source == "scratch":
        logging.info("Training from scratch (random initialization)")
    else:
        state_dict, strict = load_pretrained_weights(source)
        model.load_state_dict(state_dict, strict=strict)
        logging.info(f"Loaded weights (source={source}, strict={strict})")
    if torch.cuda.is_available():
        model = torch.compile(model)
        logging.info("Compiled model")
    return model