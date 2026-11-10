"""Encode a text file into a flat uint16 token file for training.

Example:
    python scripts/prepare_data.py --tokenizer assets/tokenizer_tinystories_4096.json \
        --input data/TinyStoriesV2-GPT4-valid.txt --out data/valid.bin
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tinylm.data import read_chunks
from tinylm.tokenizer import BPETokenizer

_tok: BPETokenizer | None = None


def _init(path: str) -> None:
    global _tok
    _tok = BPETokenizer.load(path)


def _encode(text: str) -> np.ndarray:
    return np.asarray(_tok.encode(text), dtype=np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-mb", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if BPETokenizer.load(args.tokenizer).vocab_size > 2**16:
        raise ValueError("vocabulary does not fit in uint16")
    max_bytes = None if args.max_mb is None else int(args.max_mb * 1e6)

    t0 = time.perf_counter()
    n_bytes = n_tokens = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f, Pool(args.workers, _init, (str(args.tokenizer),)) as pool:
        chunks = read_chunks(args.input, max_bytes, chunk_bytes=4 << 20)
        # imap (not imap_unordered) keeps the stories in file order
        for text_len, arr in pool.imap(_with_len, chunks):
            arr.tofile(f)
            n_bytes += text_len
            n_tokens += len(arr)
    dt = time.perf_counter() - t0
    print(
        f"{args.out}: {n_tokens:,} tokens from {n_bytes / 1e6:.1f} MB "
        f"({n_bytes / n_tokens:.2f} bytes/token, {dt:.1f}s)"
    )


def _with_len(text: str) -> tuple[int, np.ndarray]:
    return len(text.encode("utf-8")), _encode(text)


if __name__ == "__main__":
    main()
