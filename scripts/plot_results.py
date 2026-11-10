"""Draw the README figures from the CSV files in results/.

    python scripts/plot_results.py  ->  docs/loss_curve.png, docs/kv_cache.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e4df"


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def loss_curve(out: Path) -> None:
    rows = [r for r in read(ROOT / "results/base-5000/metrics.csv") if r["train_loss"] != "nan"]
    tokens = [int(r["tokens"]) / 1e6 for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(tokens, train, color=BLUE, lw=2, label="Train (mean of the last 250 steps)")
    ax.plot(tokens, val, color=ORANGE, lw=2, label="Validation")
    ax.scatter([tokens[-1]], [val[-1]], s=40, color=ORANGE, edgecolor="white", lw=1.5, zorder=3)
    ax.annotate(f"{val[-1]:.2f}", (tokens[-1], val[-1]), xytext=(6, 6),
                textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Cross-entropy loss (nats per token)")
    ax.set_ylim(1.4, max(train[0], val[0]) + 0.1)
    ax.set_xlim(0, tokens[-1] * 1.06)
    style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("12M-parameter GPT on TinyStories, 58 minutes on an M2 laptop",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def kv_cache(out: Path) -> None:
    rows = read(ROOT / "results/kv_cache.csv")
    n = [int(r["total_tokens"]) for r in rows]
    with_c = [float(r["seconds_with_cache"]) * 1000 for r in rows]
    without = [float(r["seconds_without_cache"]) * 1000 for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(n, without, color=ORANGE, lw=2, marker="o", ms=5, label="Without cache: recompute the prefix")
    ax.plot(n, with_c, color=BLUE, lw=2, marker="o", ms=5, label="With KV cache: one new token per step")
    speedup = float(rows[-1]["speedup"])
    # vertical bracket between the two end points, labelled with the speed-up
    ax.annotate("", xy=(n[-1] + 6, with_c[-1]), xytext=(n[-1] + 6, without[-1]),
                arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 1})
    ax.text(n[-1] + 1, (with_c[-1] + without[-1]) / 2, f"{speedup:.1f}x\nfaster",
            ha="right", va="center", color=INK, fontsize=9)
    ax.set_xlim(n[0] - 10, n[-1] + 14)
    ax.set_xlabel("Sequence length (tokens, 8-token prompt)")
    ax.set_ylabel("Generation time (ms)")
    ax.set_ylim(0, max(without) * 1.1)
    style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_title("Greedy generation on Apple M2 (MPS), median of 5 runs",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "docs").mkdir(exist_ok=True)
    loss_curve(ROOT / "docs/loss_curve.png")
    kv_cache(ROOT / "docs/kv_cache.png")
    print("wrote docs/loss_curve.png, docs/kv_cache.png")
