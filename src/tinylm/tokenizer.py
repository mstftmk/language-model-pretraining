"""Byte-level byte-pair encoding (BPE) tokenizer, written from scratch.

How it works
------------
1. Every string is UTF-8 bytes, so the base vocabulary is the 256 byte values.
   Any text can be encoded; there is no "unknown" token.
2. Text is first split into pre-tokens with the GPT-2 regular expression
   (words with their leading space, numbers, punctuation runs, whitespace).
   Merges never cross a pre-token boundary, so " cat" and "cat." stay apart.
3. Training repeatedly finds the most frequent adjacent pair of token ids in
   the corpus and replaces it with a new id. Each merge adds one token.
4. Encoding applies the learned merges, lowest rank (earliest learned) first.
5. Special tokens such as ``<|endoftext|>`` are cut out before pre-tokenization
   and always map to a single reserved id.

Token ids: 0-255 are bytes, 256..256+len(merges)-1 are merges, and the
special tokens come last.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import regex

# Pre-tokenization pattern used by GPT-2 (Radford et al., 2019).
GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
FORMAT = "tinylm-bpe-v1"

Pair = tuple[int, int]


def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair``, left to right."""
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _split_special(text: str, special_tokens: Iterable[str]) -> list[str]:
    """Split ``text`` so that each special token is its own piece.

    Longer special tokens are tried first so one cannot shadow another.
    """
    specials = sorted(set(special_tokens), key=len, reverse=True)
    if not specials:
        return [text]
    pattern = "(" + "|".join(regex.escape(s) for s in specials) + ")"
    return [piece for piece in regex.split(pattern, text) if piece]


def count_pretokens(
    text: str,
    special_tokens: Iterable[str] = ("<|endoftext|>",),
    pattern: str = GPT2_PATTERN,
) -> Counter[bytes]:
    """Count pre-tokens (as UTF-8 bytes) in ``text``, skipping special tokens.

    Training only needs these counts, not the text itself, which is what makes
    BPE training cheap: TinyStories has hundreds of millions of characters but
    only tens of thousands of distinct pre-tokens.
    """
    specials = set(special_tokens)
    compiled = regex.compile(pattern)
    counts: Counter[bytes] = Counter()
    for piece in _split_special(text, specials):
        if piece in specials:
            continue
        counts.update(m.encode("utf-8") for m in compiled.findall(piece))
    return counts


def learn_merges(word_counts: Counter[bytes], num_merges: int) -> list[Pair]:
    """Learn up to ``num_merges`` BPE merges from pre-token counts.

    A naive loop recounts every pair after each merge. Here we keep
    ``pair_counts`` up to date incrementally and only revisit the words that
    contain the merged pair, using ``where`` (pair -> word indices). A max-heap
    with lazy deletion returns the most frequent pair; stale entries (whose
    count changed after they were pushed) are skipped when popped.

    Ties are broken by the smaller pair of ids, which keeps training
    deterministic.
    """
    words: list[list[int]] = [list(w) for w in word_counts]
    freqs: list[int] = list(word_counts.values())

    pair_counts: dict[Pair, int] = defaultdict(int)
    where: dict[Pair, set[int]] = defaultdict(set)
    for i, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += freqs[i]
            where[pair].add(i)

    heap = [(-c, pair) for pair, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    while len(merges) < num_merges and heap:
        neg_count, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if count <= 0 or -neg_count != count:
            continue  # stale heap entry
        new_id = 256 + len(merges)
        merges.append(pair)

        touched: set[Pair] = set()
        for i in where.pop(pair, ()):
            w, f = words[i], freqs[i]
            if len(w) < 2:
                continue
            new_w = _merge(w, pair, new_id)
            if len(new_w) == len(w):
                continue  # index entry was stale, pair no longer in this word
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                touched.add(p)
            for p in zip(new_w, new_w[1:]):
                pair_counts[p] += f
                where[p].add(i)
                touched.add(p)
            words[i] = new_w

        pair_counts.pop(pair, None)
        touched.discard(pair)
        for p in touched:
            c = pair_counts[p]
            if c > 0:
                heapq.heappush(heap, (-c, p))
            else:
                pair_counts.pop(p, None)
                where.pop(p, None)
    return merges


class BPETokenizer:
    """Byte-level BPE tokenizer with special tokens."""

    def __init__(
        self,
        merges: list[Pair],
        special_tokens: Iterable[str] = ("<|endoftext|>",),
        pattern: str = GPT2_PATTERN,
    ) -> None:
        self.merges = [tuple(p) for p in merges]
        self.pattern = pattern
        self._compiled = regex.compile(pattern)
        self.special_tokens = list(dict.fromkeys(special_tokens))

        # rank of a pair = its merge index; lower ranks are applied first
        self.ranks: dict[Pair, int] = {p: r for r, p in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for r, (a, b) in enumerate(self.merges):
            self.vocab[256 + r] = self.vocab[a] + self.vocab[b]
        self.special_to_id: dict[str, int] = {}
        for s in self.special_tokens:
            idx = len(self.vocab)
            self.vocab[idx] = s.encode("utf-8")
            self.special_to_id[s] = idx
        self._cache: dict[bytes, list[int]] = {}

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        text: str | None = None,
        vocab_size: int = 4096,
        special_tokens: Iterable[str] = ("<|endoftext|>",),
        pattern: str = GPT2_PATTERN,
        word_counts: Counter[bytes] | None = None,
    ) -> "BPETokenizer":
        """Train on ``text`` or on precomputed ``word_counts``.

        ``vocab_size`` counts the 256 bytes, the merges and the special tokens.
        """
        special_tokens = list(dict.fromkeys(special_tokens))
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError("vocab_size is smaller than 256 + special tokens")
        if word_counts is None:
            if text is None:
                raise ValueError("pass text or word_counts")
            word_counts = count_pretokens(text, special_tokens, pattern)
        merges = learn_merges(word_counts, num_merges)
        return cls(merges, special_tokens, pattern)

    # ----------------------------------------------------------------- encode
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_pretoken(self, token: bytes) -> list[int]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        ids = list(token)
        inf = len(self.merges)
        while len(ids) >= 2:
            # find the adjacent pair learned earliest during training
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, inf))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            ids = _merge(ids, pair, 256 + rank)
        self._cache[token] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in _split_special(text, self.special_tokens):
            special_id = self.special_to_id.get(piece)
            if special_id is not None:
                out.append(special_id)
                continue
            for m in self._compiled.findall(piece):
                out.extend(self._encode_pretoken(m.encode("utf-8")))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        # A slice of ids can end inside a multi-byte character, so invalid
        # UTF-8 becomes U+FFFD instead of raising.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------- save / load
    def save(self, path: str | Path) -> None:
        data = {
            "format": FORMAT,
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "merges": [list(p) for p in self.merges],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"unknown tokenizer format: {data.get('format')!r}")
        return cls(
            [tuple(p) for p in data["merges"]],
            data["special_tokens"],
            data["pattern"],
        )
