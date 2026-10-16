import json

import numpy as np
import pandas as pd

from review_sentiment import figures
from review_sentiment import plotting as P
from review_sentiment.cli import COMMANDS, build_parser


def test_parser_knows_every_command():
    parser = build_parser()
    for cmd in COMMANDS:
        extra = ["--config", "x.yaml"] if cmd.startswith("train") else []
        extra += ["--models", "a"] if cmd == "predict-domains" else []
        args = parser.parse_args([cmd, *extra])
        assert args.command == cmd
    assert parser.parse_args(["compare"]).protocol == "official"


def test_leakage_figure_renders(tmp_path):
    res = tmp_path / "results"
    (res / "leakage").mkdir(parents=True)
    pd.DataFrame(
        {
            "protocol": ["legacy_random"] * 3 + ["grouped_pooled"] * 3 + ["official"],
            "test_accuracy": [0.9, 0.89, 0.895, 0.88, 0.885, 0.87, 0.883],
        }
    ).to_csv(res / "leakage" / "leakage_runs.csv", index=False)
    P.apply_style()
    out = figures.fig_leakage(res, tmp_path / "fig")
    assert out.exists() and out.stat().st_size > 10_000


def test_word_heatmap_draws_every_word(tmp_path):
    P.apply_style()
    fig, ax = P.plt.subplots(figsize=(4, 1))
    words = ["a", "very", "good", "film"] * 5
    P.word_heatmap(ax, words, np.linspace(-1, 1, len(words)))
    assert len(ax.texts) == len(words)
    P.save(fig, tmp_path / "w.png")


def test_robustness_figure_renders(tmp_path):
    res = tmp_path / "results"
    (res / "robustness").mkdir(parents=True)
    rows = [{"model": m, "test": t, "type": "MFT", "failure_rate": r}
            for m, r0 in (("tfidf_lr_tuned", 0.5), ("deberta", 0.1))
            for t, r in (("a", r0), ("b", r0 / 2))]  # fmt: skip
    pd.DataFrame(rows).to_csv(res / "robustness" / "robustness_summary.csv", index=False)
    P.apply_style()
    assert figures.fig_robustness(res, tmp_path).exists()
    json.dumps(rows)
