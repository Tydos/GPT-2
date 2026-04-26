import re
from typing import Iterable

_REGEX_PATTERN = re.compile(r"(\W+)")


def _tokenize(text: str) -> list[str]:
    return [t.strip() for t in _REGEX_PATTERN.split(text.lower()) if t.strip()]


def build_vocab(text: str) -> tuple[list[str], dict[str, int]]:
    """Build a vocabulary from raw text.

    Splits on non-word characters, lowercases, deduplicates, and appends
    special tokens <unk> and <eos>.

    Returns:
        all_words: sorted list of unique tokens (including special tokens)
        vocab: mapping of word -> integer index
    """
    all_words: list[str] = sorted(set(_tokenize(text)))
    all_words.extend(["<unk>", "<eos>"])
    vocab: dict[str, int] = {word: idx for idx, word in enumerate(all_words)}
    return all_words, vocab


class Tokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.str_to_int.get(t, self.str_to_int["<unk>"]) for t in _tokenize(text)]

    def decode(self, tokens: Iterable[int]) -> str:
        # NOTE: space-joining doesn't reconstruct punctuation spacing faithfully
        return " ".join([self.int_to_str.get(t, "<unk>") for t in tokens])
