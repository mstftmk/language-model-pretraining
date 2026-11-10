"""Draw docs/pipeline.png: text -> tokens -> model -> next-token probabilities -> loss.

The numbers in the figure come from the trained model, not from a mock-up:

    python scripts/make_pipeline_diagram.py --ckpt runs/base/ckpt.pt \
        --tokenizer assets/tokenizer_tinystories_4096.json
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from tinylm.model import GPT  # noqa: E402
from tinylm.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, LINE, FILL = "#0b0b0b", "#52514e", "#c9c7c0", "#f4f3ef"
LQ, RQ, MINUS, DOT = "“", "”", "−", "·"


def show(token: str) -> str:
    return token.replace(" ", DOT).replace("\n", "\\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="The cat was very")
    ap.add_argument("--target", default=" hungry", help="the continuation used to show the loss")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/pipeline.png")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    model = GPT.from_checkpoint(args.ckpt)
    ids = tok.encode(args.prompt)
    (target_id,) = tok.encode(args.target)
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]))
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    top = torch.topk(probs, 5)
    top_tokens = [show(tok.decode([int(i)])) for i in top.indices]
    top_probs = top.values.tolist()
    p_target = probs[target_id].item()
    loss = -math.log(p_target)

    fig = plt.figure(figsize=(10, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 39)
    ax.axis("off")

    def box(x, w, title):
        ax.add_patch(FancyBboxPatch((x, 12), w, 21, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=FILL, ec=LINE, lw=1))
        ax.text(x + 1.5, 30.5, title, fontsize=9.5, color=MUTED, va="center", fontweight="bold")

    def arrow(x0, x1, label=""):
        ax.add_patch(FancyArrowPatch((x0, 22.5), (x1, 22.5), arrowstyle="-|>", mutation_scale=12,
                                     color=MUTED, lw=1.2))
        ax.text((x0 + x1) / 2, 25.2, label, ha="center", fontsize=8.5, color=MUTED)

    ax.text(1, 36.2, "One prediction step of the trained model", fontsize=10.5, color=INK)

    # 1. text
    box(1, 18, "1. Text")
    ax.text(10, 21, f"{LQ}{args.prompt}{RQ}", ha="center", va="center", fontsize=10.5, color=INK)

    # 2. token ids
    arrow(19.5, 25.5, "tokenizer")
    box(26, 19, "2. Token ids")
    step = min(4.6, 15 / max(1, len(ids)))
    for k, i in enumerate(ids):
        y = 26 - k * step
        ax.text(28, y, show(tok.decode([i])), fontsize=10, color=INK, va="center", family="monospace")
        ax.text(43, y, str(i), fontsize=10, color=BLUE, va="center", ha="right", family="monospace")

    # 3. model
    arrow(45.5, 51.5)
    box(52, 18, "3. Transformer")
    ax.text(61, 21, "6 blocks of\nattention +\nfeed-forward", ha="center", va="center",
            fontsize=9, color=INK, linespacing=1.4)

    # 4. next-token probabilities, as a small bar chart inside the last box
    arrow(70.5, 76.5)
    box(77, 22, "4. Next token")
    bars = fig.add_axes([0.865, 0.335, 0.085, 0.38])
    ypos = list(range(len(top_tokens)))[::-1]
    colors = [ORANGE if t == show(args.target) else BLUE for t in top_tokens]
    bars.barh(ypos, top_probs, color=colors, height=0.62)
    bars.set_yticks(ypos, top_tokens, fontsize=8.5, family="monospace", color=INK)
    bars.set_xlim(0, max(top_probs) * 1.55)
    bars.set_xticks([])
    for spine in bars.spines.values():
        spine.set_visible(False)
    bars.tick_params(length=0)
    bars.set_facecolor(FILL)
    for y, p in zip(ypos, top_probs):
        bars.text(p + max(top_probs) * 0.05, y, f"{p:.2f}", va="center", fontsize=8, color=MUTED)

    def caption(y, label, text):
        lab = ax.text(1, y, label, fontsize=9.5, color=INK, fontweight="bold")
        fig.canvas.draw()
        right = ax.transData.inverted().transform(lab.get_window_extent().get_points())[1, 0]
        ax.text(right + 0.8, y, text, fontsize=9.5, color=INK)

    caption(7.2, "Training:",
            f"if the story really continues with {LQ}{show(args.target)}{RQ}, the loss is "
            f"{MINUS}ln p = {MINUS}ln {p_target:.3f} = {loss:.2f}. Training pushes that probability up.")
    caption(2.8, "Generation:", "pick one token from these probabilities, append it to the text, and repeat.")

    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"wrote {args.out}: p({args.target!r}) = {p_target:.3f}, loss {loss:.3f}; "
          f"top-5 {list(zip(top_tokens, [round(p, 3) for p in top_probs]))}")


if __name__ == "__main__":
    main()
