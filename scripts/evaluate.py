"""Loss on a whole token file, in non-overlapping windows of the full context.

Training logs use a fixed random sample of windows to stay fast; this script
scores every token once. It also reports bits per byte, which (unlike
per-token loss) can be compared across tokenizers.

Example:
    python scripts/evaluate.py --ckpt runs/base/ckpt.pt --tokens data/valid.bin \
        --text data/TinyStoriesV2-GPT4-valid.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tinylm.data import load_tokens
from tinylm.model import GPT


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--text", type=Path, help="source text, for bits per byte")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    model = GPT.from_checkpoint(args.ckpt, args.device)
    T = model.cfg.context
    tokens = load_tokens(args.tokens)
    n_windows = (len(tokens) - 1) // T
    total_nll, total_count = 0.0, 0
    for start in range(0, n_windows, args.batch_size):
        idx = np.arange(start, min(start + args.batch_size, n_windows)) * T
        window = np.stack([tokens[i : i + T + 1] for i in idx]).astype(np.int64)
        x = torch.from_numpy(window[:, :-1]).to(args.device)
        y = torch.from_numpy(window[:, 1:]).to(args.device)
        logits, _ = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_count += y.numel()

    loss = total_nll / total_count
    result = {"tokens_scored": total_count, "loss_nats_per_token": round(loss, 4),
              "perplexity": round(math.exp(loss), 3)}
    if args.text:
        bytes_per_token = args.text.stat().st_size / len(tokens)
        result["bytes_per_token"] = round(bytes_per_token, 3)
        result["bits_per_byte"] = round(loss / math.log(2) / bytes_per_token, 4)
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
