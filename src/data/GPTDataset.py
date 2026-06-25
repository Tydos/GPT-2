import torch
from torch.utils.data import Dataset, DataLoader
from src.data.tokenizer import BaseTokenizer


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


def raw_text_dataloader(
    text: str,
    tokenizer: BaseTokenizer,
    max_len: int,
    stride: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Manually split raw test into train/test/val splits and returns dataloaders"""

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


def dataset_text_dataloader(
    train_text: str,
    val_text: str,
    test_text: str,
    tokenizer: BaseTokenizer,
    max_len: int,
    stride: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Importing data from HuggingFace which already have the train/test/val splits and returns dataloaders"""

    train_ds = GPTDataset(train_text, tokenizer, max_len, stride)
    val_ds = GPTDataset(val_text, tokenizer, max_len, stride)
    test_ds = GPTDataset(test_text, tokenizer, max_len, stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
