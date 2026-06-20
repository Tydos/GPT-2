import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import Tokenizer, TikTokenizer


class GPTDataset(Dataset):
    """Sliding-window dataset for GPT-style next-token prediction.

    Args:
        text: raw input text
        tokenizer: a Tokenizer instance
        max_len: number of tokens the model sees as context
        stride: step size between windows
    """

    def __init__(
        self, text: str, tokenizer: TikTokenizer, max_len: int, stride: int
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        tokenized_text = tokenizer.encode(text)
        for i in range(0, len(tokenized_text) - max_len, stride):
            input_chunk = tokenized_text[i : i + max_len]
            target_chunk = tokenized_text[i + 1 : i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_gpt_dataloader(
    text: str,
    tokenizer: TikTokenizer,
    max_len: int = 3,
    stride: int = 1,
    batch_size: int = 10,
) -> DataLoader:
    """Create a DataLoader from raw text using sliding-window chunking."""
    dataset = GPTDataset(text, tokenizer, max_len, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
