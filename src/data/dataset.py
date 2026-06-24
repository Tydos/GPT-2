from typing import Any


import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from src.data.tokenizer import BaseTokenizer

GPT2_EOS_TOKEN_ID = 50256


class GPTDataset(Dataset):
    """Sliding-window dataset for GPT-style next-token prediction. (pretraining)"""

    def __init__(
        self, text: str, tokenizer: BaseTokenizer, max_len: int, stride: int
    ) -> None:
        self.max_len = max_len
        self.stride = stride
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.num_samples = max(0, (len(self.tokens) - max_len - 1) // stride + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        return (
            self.tokens[start : start + self.max_len],
            self.tokens[start + 1 : start + self.max_len + 1],
        )


class SMSSpamDataset(Dataset):
    """SMS Spam dataset for classification fine-tuning."""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        max_len: int | None = None,
        pad_token_id: int = GPT2_EOS_TOKEN_ID,
        hf_dataset: str = "ucirvine/sms_spam",
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        splits = load_dataset(hf_dataset)["train"].train_test_split(
            test_size=val_fraction, stratify_by_column="label", seed=seed
        )
        data = splits["train" if split == "train" else "test"]
        self.texts = data["sms"]
        self.labels = data["label"]
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_len = max_len or max(len(tokenizer.encode(t)) for t in self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer.encode(self.texts[idx])[: self.max_len]
        tokens += [self.pad_token_id] * (self.max_len - len(tokens))
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def create_gpt_dataloader(
    text: str,
    tokenizer: BaseTokenizer,
    max_len: int = 3,
    stride: int = 1,
    batch_size: int = 10,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader from raw text using sliding-window chunking."""
    return DataLoader(
        GPTDataset(text, tokenizer, max_len, stride),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_gpt_dataloader2(
    text: str,
    tokenizer: BaseTokenizer,
    max_len: int = 256,
    stride: int = 64,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from raw text using sliding-window chunking."""

    n = len(text)
    train_text = text[: int(0.7 * n)]  # 70% of the text for training
    val_text = text[int(0.7 * n) : int(0.9 * n)]  # 20% of the text for validation
    test_text = text[int(0.9 * n) :]  # 10% of the text for testing

    train_ds = GPTDataset(train_text, tokenizer, max_len, stride)
    val_ds = GPTDataset(val_text, tokenizer, max_len, stride)
    test_ds = GPTDataset(test_text, tokenizer, max_len, stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_sms_spam_dataloader(
    tokenizer: BaseTokenizer,
    max_len: int | None = None,
    batch_size: int = 16,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    hf_dataset: str = "ucirvine/sms_spam",
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for SMS spam classification."""
    train_ds = SMSSpamDataset(
        tokenizer=tokenizer,
        max_len=max_len,
        hf_dataset=hf_dataset,
        split="train",
        val_fraction=val_fraction,
        seed=seed,
    )
    val_ds = SMSSpamDataset(
        tokenizer=tokenizer,
        max_len=train_ds.max_len,
        hf_dataset=hf_dataset,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
