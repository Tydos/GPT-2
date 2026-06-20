import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import BaseTokenizer


class GPTDataset(Dataset):
    """Sliding-window dataset for GPT-style next-token prediction.

    Args:
        text: raw input text
        tokenizer: a Tokenizer instance
        max_len: number of tokens the model sees as context
        stride: step size between windows
    """

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
    dataset = GPTDataset(text, tokenizer, max_len, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
