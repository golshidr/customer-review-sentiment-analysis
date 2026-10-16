"""Shared figure style: one palette, readable fonts, colour-blind-safe categorical order."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, used in this fixed order (validated for colour-vision deficiencies on
# adjacent pairs; the first three are distinguishable even when all pairs are compared).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"
NEUTRAL = "#f0efec"
POSITIVE = "#2a78d6"  # diverging pole: pushes towards "positive"
NEGATIVE = "#e34948"  # diverging pole: pushes towards "negative"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("div", [NEGATIVE, NEUTRAL, POSITIVE])

MODEL_LABELS = {
    "tfidf_lr_original": "TF-IDF + LR (original)",
    "tfidf_lr_tuned": "TF-IDF + LR (tuned)",
    "distilbert": "DistilBERT",
    "roberta": "RoBERTa-base",
    "roberta_head_tail": "RoBERTa-base (head+tail)",
    "deberta": "DeBERTa-v3-base",
    "deberta_large": "DeBERTa-v3-large",
}
FAMILY_COLOR = {"classical": PALETTE[1], "transformer": PALETTE[0]}


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def family(model: str) -> str:
    return "classical" if model.startswith("tfidf") else "transformer"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "text.color": INK,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        }
    )


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def word_heatmap(
    ax,
    words: list[str],
    scores,
    max_abs: float | None = None,
    width: float = 1.0,
    fontsize: float = 10.0,
    line_height: float = 0.13,
) -> float:
    """Draw ``words`` as a wrapped paragraph with a diverging background per word.

    Returns the y coordinate (axes units) below the last line.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    max_abs = max_abs or (np.abs(scores).max() or 1.0)
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    ax.set_axis_off()
    x, y = 0.0, 1.0 - line_height / 2
    for w, s in zip(words, scores, strict=True):
        color = DIVERGING(0.5 + 0.5 * float(np.clip(s / max_abs, -1, 1)))
        t = ax.text(
            x,
            y,
            w,
            fontsize=fontsize,
            va="center",
            ha="left",
            transform=ax.transAxes,
            bbox={"boxstyle": "square,pad=0.15", "fc": color, "ec": "none"},
            color=INK,
        )
        bb = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        if bb.x1 > width and x > 0:
            t.remove()
            x, y = 0.0, y - line_height
            t = ax.text(
                x,
                y,
                w,
                fontsize=fontsize,
                va="center",
                ha="left",
                transform=ax.transAxes,
                color=INK,
                bbox={"boxstyle": "square,pad=0.15", "fc": color, "ec": "none"},
            )
            bb = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        x = bb.x1 + 0.012
    return y - line_height / 2
