"""Reading text in story-aligned chunks, and sampling training windows.

The corpus is encoded once into a flat ``uint16`` array of token ids (a
4,096-token vocabulary fits in 16 bits). Training then reads random windows
of ``context + 1`` tokens from a memory-mapped file: the first ``context``
are the input, the same window shifted by one is the target.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

EOT = "<|endoftext|>"


def read_chunks(
    path: str | Path, max_bytes: int | None = None, chunk_bytes: int = 8 << 20
) -> Iterator[str]:
    """Yield text chunks of about ``chunk_bytes`` that end on a story boundary.

    When ``max_bytes`` cuts a story in half, the partial story is dropped.
    """
    budget = float("inf") if max_bytes is None else max_bytes
    carry = ""
    with Path(path).open("r", encoding="utf-8") as f:
        while budget > 0:
            block = f.read(int(min(chunk_bytes, budget)))
            if not block:
                break
            budget -= len(block.encode("utf-8"))
            text = carry + block
            cut = text.rfind(EOT)
            if cut == -1:
                carry = text
                continue
            cut += len(EOT)
            yield text[:cut]
            carry = text[cut:]
    if carry.strip() and max_bytes is None:
        yield carry


def load_tokens(path: str | Path) -> np.ndarray:
    """Memory-map a ``uint16`` token file without reading it into RAM."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def sample_batch(
    tokens: np.ndarray, batch_size: int, context: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random windows: ``x`` is tokens[i:i+T], ``y`` is tokens[i+1:i+T+1]."""
    if len(tokens) <= context:
        raise ValueError("token array is shorter than the context length")
    starts = rng.integers(0, len(tokens) - context, size=batch_size)
    window = np.stack([tokens[s : s + context + 1] for s in starts]).astype(np.int64)
    return window[:, :-1], window[:, 1:]
