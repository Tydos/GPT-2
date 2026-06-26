import re
import os
import json
from abc import ABC, abstractmethod
from typing import Iterable
import tiktoken

_PATTERN = re.compile(r'([,.:;?_!"()\']|--|\s)')


class BaseTokenizer(ABC):
    """Base class for all tokenizers."""
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: Iterable[int]) -> str: ...


class SimpleTokenizer(BaseTokenizer):
    """Word-level tokenizer built from a raw text corpus."""

    def __init__(self, vocab: dict[str, int]) -> None:
        """ Build a str->int and int->str mapping from a vocab dict. """
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
        self.unk_id = vocab["<unk>"]
        self.eos_id = vocab["<eos>"]

    @staticmethod
    def _split(text: str) -> list[str]:
        """ Split text into words, handling punctuation and whitespace. """
        return [t for t in _PATTERN.split(text.lower()) if t and not t.isspace()]

    @classmethod
    def from_text(cls, text: str) -> "SimpleTokenizer":
        """ Build a SimpleTokenizer from raw text. """
        words = sorted(set(cls._split(text))) + ["<unk>", "<eos>"]
        return cls({word: idx for idx, word in enumerate(words)})

    def save(self, directory: str) -> None:
        """ Save the tokenizer's vocab to <directory>/vocab.json. """
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vocab.json"), "w") as f:
            json.dump(self.str_to_int, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "SimpleTokenizer":
        """ Load a SimpleTokenizer from <directory>/vocab.json. """
        with open(os.path.join(directory, "vocab.json")) as f:
            return cls(json.load(f))

    def encode(self, text: str) -> list[int]:
        """ Encode text into a list of token IDs. """
        return [self.str_to_int.get(t, self.unk_id) for t in self._split(text)] + [self.eos_id]

    def decode(self, tokens: Iterable[int]) -> str:
        """ Decode a list of token IDs into a string. """
        return " ".join(self.int_to_str.get(t, "<unk>") for t in tokens)


class BPETokenizer(BaseTokenizer):
    """BPE tokenizer used in GPT-2, via the tiktoken library."""

    def __init__(self, model_type: str) -> None:
        self.tokenizer = tiktoken.get_encoding(model_type)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(list(tokens))
