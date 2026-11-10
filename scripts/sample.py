"""Generate stories from a trained checkpoint.

Example:
    python scripts/sample.py --ckpt runs/base/ckpt.pt \
        --tokenizer assets/tokenizer_tinystories_4096.json --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse

import torch

from tinylm.model import GPT
from tinylm.tokenizer import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--num", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    model = GPT.from_checkpoint(args.ckpt, args.device)
    eot = tok.special_to_id["<|endoftext|>"]
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    for i in range(args.num):
        idx = torch.tensor([tok.encode(args.prompt)], device=args.device)
        out = model.generate(idx, args.max_new, args.temperature, args.top_k,
                             stop_token=eot, generator=gen)
        if i:
            print("\n---\n")
        print(tok.decode(t for t in out[0].tolist() if t != eot).strip())


if __name__ == "__main__":
    main()
