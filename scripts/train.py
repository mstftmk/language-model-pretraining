"""Pre-train the GPT on token files made by prepare_data.py.

Example:
    python scripts/train.py --train data/train.bin --valid data/valid.bin \
        --tokenizer assets/tokenizer_tinystories_4096.json --out runs/base \
        --max-steps 5000

Writes ``metrics.csv`` (one row per evaluation), ``ckpt.pt`` (best
validation loss) and ``samples.txt`` to the output folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tinylm.data import load_tokens, sample_batch
from tinylm.model import GPT, GPTConfig
from tinylm.tokenizer import BPETokenizer


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Linear warm-up, then cosine decay from ``max_lr`` to ``min_lr``."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, lr: float, weight_decay: float) -> torch.optim.AdamW:
    # weight decay on matrices (linear layers, embedding) only, not on norm gains
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model, tokens, batches, batch_size, context, device, seed=1234) -> float:
    """Mean loss on a fixed set of validation windows (same windows every time)."""
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(batches):
        x, y = sample_batch(tokens, batch_size, context, rng)
        _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2, help="micro-batches per optimiser step")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--max-minutes", type=float, default=None, help="stop early after this long")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    tok = BPETokenizer.load(args.tokenizer)
    train_tokens, valid_tokens = load_tokens(args.train), load_tokens(args.valid)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size, context=args.context, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
    )
    model = GPT(cfg).to(device)
    opt = make_optimizer(model, args.lr, args.weight_decay)
    tokens_per_step = args.batch_size * args.grad_accum * args.context
    print(
        f"device {device}, {model.num_params() / 1e6:.2f}M params, "
        f"{len(train_tokens) / 1e6:.1f}M train tokens, {tokens_per_step:,} tokens/step"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps({"model": cfg.to_dict(), "train": {k: str(v) for k, v in vars(args).items()}}, indent=2)
    )
    log = (args.out / "metrics.csv").open("w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "tokens", "train_loss", "val_loss", "lr", "minutes", "tokens_per_sec"])

    best = float("inf")
    t0 = time.perf_counter()
    t_last, step_last = t0, 0
    running = []
    step = 0
    for step in range(args.max_steps + 1):
        lr = lr_at(step, args.lr, args.min_lr, args.warmup, args.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        minutes = (time.perf_counter() - t0) / 60
        out_of_time = args.max_minutes is not None and minutes >= args.max_minutes
        if step % args.eval_every == 0 or step == args.max_steps or out_of_time:
            val = evaluate(model, valid_tokens, args.eval_batches, args.batch_size, args.context, device)
            now = time.perf_counter()
            tps = (step - step_last) * tokens_per_step / max(now - t_last, 1e-9)
            t_last, step_last = now, step
            train_loss = float(np.mean(running)) if running else float("nan")
            running = []
            writer.writerow([step, step * tokens_per_step, f"{train_loss:.4f}", f"{val:.4f}",
                             f"{lr:.2e}", f"{minutes:.2f}", f"{tps:.0f}"])
            log.flush()
            print(f"step {step:6d} | train {train_loss:.3f} | val {val:.3f} | "
                  f"lr {lr:.1e} | {tps:,.0f} tok/s | {minutes:.1f} min")
            if val < best:
                best = val
                torch.save({"config": cfg.to_dict(), "model": model.state_dict(),
                            "step": step, "val_loss": val}, args.out / "ckpt.pt")
        if step == args.max_steps or out_of_time:
            break

        # gradient accumulation: several small batches per optimiser step keep
        # memory low while giving the gradient of one larger batch
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = sample_batch(train_tokens, args.batch_size, args.context, rng)
            _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            (loss / args.grad_accum).backward()
            step_loss += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running.append(step_loss)
    log.close()

    # samples from the best checkpoint
    state = torch.load(args.out / "ckpt.pt", map_location=device)
    model.load_state_dict(state["model"])
    gen = torch.Generator(device=device).manual_seed(0)
    eot = tok.special_to_id["<|endoftext|>"]
    lines = []
    for prompt in ["Once upon a time", "Lily and her dog", "The little robot"]:
        idx = torch.tensor([tok.encode(prompt)], device=device)
        out = model.generate(idx, 200, temperature=0.8, top_k=50, stop_token=eot, generator=gen)
        lines.append(tok.decode(t for t in out[0].tolist() if t != eot))
    (args.out / "samples.txt").write_text("\n\n---\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"best val loss {best:.3f} (perplexity {math.exp(best):.1f}); samples in {args.out / 'samples.txt'}")


if __name__ == "__main__":
    main()
