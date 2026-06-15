import re
from typing import Iterable

_PATTERN = re.compile(r"(\W+)")


def build_vocab(text: str) -> tuple[list[str], dict[str, int]]:
    """Build a vocab set {word->int} mapping"""
    tokens = [t.strip() for t in _PATTERN.split(text.lower()) if t.strip()]
    all_words: list[str] = sorted(set(tokens))
    all_words.extend(["<unk>", "<eos>"])
    vocab: dict[str, int] = {word: idx for idx, word in enumerate(all_words)}
    return all_words, vocab


class Tokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = [t.strip() for t in _PATTERN.split(text.lower()) if t.strip()]
        return [self.str_to_int.get(t, self.str_to_int["<unk>"]) for t in tokens]

    def decode(self, tokens: Iterable[int]) -> str:
        return " ".join(self.int_to_str.get(t, "<unk>") for t in tokens)
