import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from src.data.tokenizer import BaseTokenizer

GPT2_EOS_TOKEN_ID = 50256


class SMSSpamDataset(Dataset):
    """SMS Spam dataset for classification fine-tuning."""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        seq_len: int | None = None,
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
        self.seq_len = seq_len or max(len(tokenizer.encode(t)) for t in self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer.encode(self.texts[idx])[: self.seq_len]
        tokens += [self.pad_token_id] * (self.seq_len - len(tokens))
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def create_sms_spam_dataloader(
    tokenizer: BaseTokenizer,
    seq_len: int | None = None,
    batch_size: int = 16,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    hf_dataset: str = "ucirvine/sms_spam",
) -> tuple[DataLoader, DataLoader]:
    train_ds = SMSSpamDataset(tokenizer=tokenizer, seq_len=seq_len, hf_dataset=hf_dataset,
                               split="train", val_fraction=val_fraction, seed=seed)
    val_ds = SMSSpamDataset(tokenizer=tokenizer, seq_len=train_ds.seq_len, hf_dataset=hf_dataset,
                             split="val", val_fraction=val_fraction, seed=seed)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
    )
