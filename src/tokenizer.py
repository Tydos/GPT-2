import re
import os
import json
from abc import ABC, abstractmethod
from typing import Iterable

import tiktoken

_PATTERN = re.compile(r'([,.:;?_!"()\']|--|\s)')


def build_vocab(text: str) -> dict[str, int]:
    """Build a vocab set {word->int} mapping"""
    tokens = [t for t in _PATTERN.split(text.lower()) if t and not t.isspace()]
    all_words: list[str] = sorted(set(tokens))
    all_words.extend(["<unk>", "<eos>"])
    return {word: idx for idx, word in enumerate(all_words)}


def save_vocab(vocab: dict[str, int], file_path: str) -> None:
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: Iterable[int]) -> str: ...


# Vanilla Toknenizer that creates a str->int map through the whole dataset
class SimpleTokenizer(BaseTokenizer):
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = [t for t in _PATTERN.split(text.lower()) if t and not t.isspace()]
        ids = [self.str_to_int.get(t, self.str_to_int["<unk>"]) for t in tokens]
        ids.append(self.str_to_int["<eos>"])
        return ids

    def decode(self, tokens: Iterable[int]) -> str:
        return " ".join(self.int_to_str.get(t, "<unk>") for t in tokens)


# Optimised BPE tokenizer used in GPT2
class TikTokenizer(BaseTokenizer):
    def __init__(self, model_type: str) -> None:
        self.tokenizer = tiktoken.get_encoding(model_type)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(list(tokens))
