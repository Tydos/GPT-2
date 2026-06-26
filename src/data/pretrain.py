import torch
from torch.utils.data import Dataset, DataLoader
from src.data.tokenizer import BaseTokenizer


class GPTDataset(Dataset):
    """Sliding-window dataset for GPT-style next-token prediction."""

    def __init__(
        self, text: str, tokenizer: BaseTokenizer, seq_len: int, stride: int
    ) -> None:
        self.seq_len = seq_len
        self.stride = stride
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.num_samples = max(0, (len(self.tokens) - seq_len - 1) // stride + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        return (
            self.tokens[start : start + self.seq_len],
            self.tokens[start + 1 : start + self.seq_len + 1],
        )


def create_dataloaders_from_text(
    text: str,
    tokenizer: BaseTokenizer,
    seq_len: int,
    stride: int,
    batch_size: int,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split raw text and return train/val/test DataLoaders."""
    n = len(text)
    a, b = int(split[0] * n), int((split[0] + split[1]) * n)
    return create_dataloaders_from_splits(
        text[:a], text[a:b], text[b:],
        tokenizer, seq_len, stride, batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def create_dataloaders_from_splits(
    train_text: str,
    val_text: str,
    test_text: str,
    tokenizer: BaseTokenizer,
    seq_len: int,
    stride: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from pre-split text (e.g. HuggingFace splits)."""
    return (
        DataLoader(GPTDataset(train_text, tokenizer, seq_len, stride), batch_size=batch_size, 
        shuffle=True,  num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(GPTDataset(val_text,   tokenizer, seq_len, stride), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(GPTDataset(test_text,  tokenizer, seq_len, stride), batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    )
