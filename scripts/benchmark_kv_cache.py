"""Time greedy generation with and without the KV cache.

For each target length the model generates from an 8-token prompt up to that
many tokens, once with the cache and once recomputing the full sequence every
step. Both runs must produce identical tokens; the script checks this.

Example:
    python scripts/benchmark_kv_cache.py --ckpt runs/base/ckpt.pt --out results/kv_cache.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from tinylm.model import GPT


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timed_generate(model, prompt, new_tokens, use_cache, device):
    sync(device)
    t = time.perf_counter()
    out = model.generate(prompt, new_tokens, temperature=0, use_cache=use_cache)
    sync(device)
    return time.perf_counter() - t, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--lengths", default="32,64,96,128,160,192,224,256")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = GPT.from_checkpoint(args.ckpt, device)
    g = torch.Generator().manual_seed(args.seed)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 8), generator=g).to(device)

    # warm-up: the first calls compile kernels and allocate memory
    for use_cache in (True, False):
        timed_generate(model, prompt, 32, use_cache, device)

    rows = []
    for total in map(int, args.lengths.split(",")):
        new = total - prompt.shape[1]
        times = {True: [], False: []}
        for _ in range(args.repeats):
            for use_cache in (True, False):
                dt, out = timed_generate(model, prompt, new, use_cache, device)
                times[use_cache].append(dt)
                if use_cache:
                    cached_out = out
                else:
                    assert torch.equal(out, cached_out), "cache changed the output"
        with_c, without = statistics.median(times[True]), statistics.median(times[False])
        rows.append({"total_tokens": total, "new_tokens": new,
                     "seconds_with_cache": f"{with_c:.4f}", "seconds_without_cache": f"{without:.4f}",
                     "speedup": f"{without / with_c:.2f}"})
        print(f"{total:4d} tokens: cache {with_c * 1000:7.1f} ms | no cache {without * 1000:7.1f} ms "
              f"| {without / with_c:.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"device {device}; outputs identical with and without cache; saved {args.out}")


if __name__ == "__main__":
    main()
