"""Render the README result tables (Markdown) from the CSV/JSON files in ``results/``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .plotting import label


def _fmt_pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}"


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def comparison_md(results: Path) -> str:
    comp = pd.read_csv(results / "comparison.csv")
    rows = []
    for r in comp.itertuples():
        seeds = (
            f"{_fmt_pct(r.seed_acc_mean)} +/- {_fmt_pct(r.seed_acc_std)} (n={r.n_seeds})"
            if r.n_seeds > 1
            else "-"
        )
        params = f"{r.n_parameters / 1e6:.0f}M" if pd.notna(r.n_parameters) else "-"
        rows.append(
            [
                label(r.model),
                f"{_fmt_pct(r.accuracy)} [{_fmt_pct(r.acc_ci_low)}, {_fmt_pct(r.acc_ci_high)}]",
                seeds,
                f"{r.macro_f1:.4f}",
                f"{r.roc_auc:.4f}",
                f"{r.ece:.3f}",
                params,
                f"{r.train_seconds:.0f} s",
                f"{r.throughput:,.0f}/s ({r.hardware})" if pd.notna(r.throughput) else "-",
            ]
        )
    header = ["Model", "Accuracy % [95% CI]", "Accuracy % over seeds (mean +/- sd)", "Macro-F1",
              "ROC-AUC", "ECE", "Params", "Train time", "Inference"]  # fmt: skip
    return _table(rows, header)


def significance_md(results: Path, reference: str = "tfidf_lr_tuned") -> str:
    sig = pd.read_csv(results / "significance.csv")
    sig = sig[sig.reference == reference]
    rows = [
        [
            label(r.model),
            f"{100 * r.diff:+.2f} [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}]",
            f"{r.a_only_correct:,} / {r.b_only_correct:,}",
            f"{r.p_value:.2g}" if r.p_value > 0 else "< 1e-300",
        ]
        for r in sig.itertuples()
    ]
    header = [f"Model vs {label(reference)}", "Accuracy difference, points [95% CI]",
              "Only this model right / only reference right", "McNemar p"]  # fmt: skip
    return _table(rows, header)


def cross_domain_md(results: Path) -> str:
    cross = pd.read_csv(results / "cross_domain.csv")
    piv = cross.pivot(index="model", columns="dataset", values="accuracy")
    order = list(dict.fromkeys(cross.model))
    rows = []
    for m in order:
        r = piv.loc[m]
        rows.append(
            [
                label(m),
                _fmt_pct(r["imdb_test"]),
                f"{_fmt_pct(r['amazon'])} ({100 * (r['amazon'] - r['imdb_test']):+.1f})",
                f"{_fmt_pct(r['yelp'])} ({100 * (r['yelp'] - r['imdb_test']):+.1f})",
            ]
        )
    header = ["Model (trained on IMDb only)", "IMDb test %", "Amazon % (change)", "Yelp % (change)"]
    return _table(rows, header)


def calibration_md(results: Path) -> str:
    c = pd.read_csv(results / "calibration" / "calibration_summary.csv")
    rows = []
    for m in list(dict.fromkeys(c.model)):
        part = c[c.model == m]
        cells = [label(m), f"{part.temperature.iloc[0]:.2f}"]
        for ds in ("imdb_test", "amazon", "yelp"):
            b = part[(part.dataset == ds) & (part.stage == "before")].ece
            a = part[(part.dataset == ds) & (part.stage == "after")].ece
            cells.append(f"{b.iloc[0]:.3f} -> {a.iloc[0]:.3f}" if len(b) else "-")
        rows.append(cells)
    header = ["Model", "Temperature (fit on IMDb val)", "ECE IMDb test", "ECE Amazon", "ECE Yelp"]
    return _table(rows, header)


def robustness_md(results: Path) -> str:
    t = pd.read_csv(results / "robustness" / "robustness_summary.csv")
    models = list(dict.fromkeys(t.model))
    rows = []
    for test in dict.fromkeys(t.test):
        part = t[t.test == test].set_index("model")
        kind = part.type.iloc[0]
        lo, hi = int(part.n.min()), int(part.n.max())  # DIR edits: n depends on the model
        n = f"{lo}" if lo == hi else f"{lo}-{hi}"
        rows.append(
            [f"{test.replace('_', ' ')} ({kind}, n={n})"]
            + [f"{100 * part.loc[m, 'failure_rate']:.1f}" for m in models]
        )
    return _table(rows, ["Test (failure rate %)"] + [label(m) for m in models])


def faithfulness_md(results: Path) -> str:
    meta = json.loads((results / "explain" / "faithfulness.json").read_text())
    rows = [[label(r["model"]), r["method"], f"{r['aopc']:.3f}"] for r in meta["results"]]
    return _table(rows, ["Model", "Attribution method", "AOPC (higher = more faithful)"])


def write_all(results: Path) -> Path:
    results = Path(results)
    parts = []
    for title, fn in (
        ("Model comparison (IMDb test)", comparison_md),
        ("Paired significance vs tuned TF-IDF", significance_md),
        ("Paired significance vs DeBERTa-v3-base", lambda r: significance_md(r, "deberta")),
        ("Cross-domain accuracy", cross_domain_md),
        ("Calibration", calibration_md),
        ("Behavioural tests", robustness_md),
        ("Explanation faithfulness", faithfulness_md),
    ):
        try:
            parts.append(f"## {title}\n\n{fn(results)}")
        except (FileNotFoundError, KeyError) as e:
            parts.append(f"## {title}\n\n(not available: {e!r})\n")
    out = results / "tables.md"
    out.write_text("# Result tables (generated by `python -m review_sentiment tables`)\n\n"
                   + "\n".join(parts))  # fmt: skip
    return out
