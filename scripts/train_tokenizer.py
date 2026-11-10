"""Train the byte-level BPE tokenizer on a text file.

Example:
    python scripts/train_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt \
        --max-mb 50 --vocab-size 4096 --out tokenizer.json

Pre-token counting is the slow part, so the input is cut into chunks at
``<|endoftext|>`` boundaries and counted in parallel. Merging then runs on
the counts only.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from tinylm.data import read_chunks
from tinylm.tokenizer import BPETokenizer, count_pretokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--max-mb", type=float, default=None, help="read only the first N MB")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    max_bytes = None if args.max_mb is None else int(args.max_mb * 1e6)
    t0 = time.perf_counter()
    counts: Counter[bytes] = Counter()
    with Pool(args.workers) as pool:
        for c in pool.imap_unordered(count_pretokens, read_chunks(args.input, max_bytes)):
            counts.update(c)
    t1 = time.perf_counter()
    print(f"pre-tokens: {sum(counts.values()):,} total, {len(counts):,} distinct ({t1 - t0:.1f}s)")

    tok = BPETokenizer.train(vocab_size=args.vocab_size, word_counts=counts)
    t2 = time.perf_counter()
    print(f"merges: {len(tok.merges):,}, vocab size {tok.vocab_size:,} ({t2 - t1:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
