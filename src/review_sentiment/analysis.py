"""Aggregate saved predictions into comparison, calibration and error-analysis tables.

Everything here runs on the CPU from ``results/predictions`` and ``results/metrics``, so results can
be re-tabulated without retraining.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .calibration import apply_temperature, fit_temperature, reliability_table
from .evaluation import (
    accuracy_from_margin,
    bootstrap_ci,
    classification_metrics,
    mcnemar_test,
    paired_bootstrap_diff,
)
from .io import load_predictions, prediction_path
from .paths import RESULTS_DIR

SEED_RE = re.compile(r"_s\d+$")
MODEL_ORDER = [
    "tfidf_lr_original",
    "tfidf_lr_tuned",
    "distilbert",
    "roberta",
    "roberta_head_tail",
    "deberta",
    "deberta_large",
]
DATASETS = ["imdb_test", "amazon", "yelp"]


def discover_models(results_dir: Path = RESULTS_DIR) -> list[str]:
    names = [p.stem for p in (Path(results_dir) / "metrics").glob("*.json")]
    main = [n for n in names if not SEED_RE.search(n)]
    return sorted(main, key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else 99, n))


def _metrics_json(model: str, results_dir: Path) -> dict:
    return json.loads((Path(results_dir) / "metrics" / f"{model}.json").read_text())


def seed_runs(model: str, results_dir: Path = RESULTS_DIR) -> list[float]:
    """Test accuracies of the main run plus any ``<model>_s<seed>`` runs."""
    accs = []
    for p in sorted((Path(results_dir) / "metrics").glob(f"{model}*.json")):
        if p.stem == model or (SEED_RE.search(p.stem) and SEED_RE.sub("", p.stem) == model):
            accs.append(json.loads(p.read_text())["test"]["accuracy"])
    return accs


def comparison_table(models: list[str], results_dir: Path = RESULTS_DIR, n_boot: int = 1000):
    """Main results table. Throughput comes from ``results/benchmark.csv`` (``benchmark`` command),
    which times every model on the same reviews with nothing else running on the GPU."""
    bench_path = Path(results_dir) / "benchmark.csv"
    bench = {}
    if bench_path.exists():
        bench = pd.read_csv(bench_path).set_index("model").to_dict(orient="index")
    rows = []
    for m in models:
        meta = _metrics_json(m, results_dir)
        pred = load_predictions(m, "imdb_test")
        y, z = pred["label"].to_numpy(), pred["margin"].to_numpy()
        acc, lo, hi = bootstrap_ci(y, z, accuracy_from_margin, n_boot=n_boot)
        met = classification_metrics(y, z)
        seeds = seed_runs(m, results_dir)
        rows.append(
            {
                "model": m,
                "accuracy": acc,
                "acc_ci_low": lo,
                "acc_ci_high": hi,
                "macro_f1": met["macro_f1"],
                "roc_auc": met["roc_auc"],
                "log_loss": met["log_loss"],
                "ece": met["ece"],
                "n_seeds": len(seeds),
                "seed_acc_mean": float(np.mean(seeds)),
                "seed_acc_std": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else np.nan,
                "n_parameters": meta.get("n_parameters"),
                "train_seconds": meta.get("train_seconds"),
                "throughput": bench.get(m, {}).get("reviews_per_second"),
                "hardware": "CPU" if m.startswith("tfidf") else "A100 GPU",
                "n_test": met["n"],
            }
        )
    return pd.DataFrame(rows)


def significance_table(models: list[str], reference: str, n_boot: int = 2000) -> pd.DataFrame:
    ref = load_predictions(reference, "imdb_test")
    rows = []
    for m in models:
        if m == reference:
            continue
        other = load_predictions(m, "imdb_test")
        assert (other["review_id"].to_numpy() == ref["review_id"].to_numpy()).all()
        y = ref["label"].to_numpy()
        pa = (other["margin"].to_numpy() > 0).astype(int)
        pb = (ref["margin"].to_numpy() > 0).astype(int)
        rows.append({"model": m, "reference": reference, **paired_bootstrap_diff(y, pa, pb, n_boot),
                     **mcnemar_test(y, pa, pb)})  # fmt: skip
    return pd.DataFrame(rows)


def cross_domain_table(models: list[str], n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for m in models:
        for ds in DATASETS:
            if not prediction_path(m, ds).exists():
                continue
            pred = load_predictions(m, ds)
            y, z = pred["label"].to_numpy(), pred["margin"].to_numpy()
            acc, lo, hi = bootstrap_ci(y, z, accuracy_from_margin, n_boot=n_boot)
            met = classification_metrics(y, z)
            rows.append(
                {
                    "model": m,
                    "dataset": ds,
                    "accuracy": acc,
                    "acc_ci_low": lo,
                    "acc_ci_high": hi,
                    "macro_f1": met["macro_f1"],
                    "roc_auc": met["roc_auc"],
                    "ece": met["ece"],
                    "positive_rate": float((z > 0).mean()),
                    "n": met["n"],
                }
            )
    return pd.DataFrame(rows)


def calibration_tables(models: list[str], n_bins: int = 10):
    """Fit a temperature on IMDb validation margins; evaluate before/after on every test set.

    ``n_bins`` only affects the reliability-diagram table; ECE always uses 15 bins.
    """
    summary, bins = [], []
    for m in models:
        val = load_predictions(m, "imdb_val")
        t = fit_temperature(val["margin"], val["label"])
        for ds in DATASETS:
            if not prediction_path(m, ds).exists():
                continue
            pred = load_predictions(m, ds)
            y = pred["label"].to_numpy()
            for stage, z in (
                ("before", pred["margin"].to_numpy()),
                ("after", apply_temperature(pred["margin"], t)),
            ):
                met = classification_metrics(y, z)
                summary.append(
                    {
                        "model": m,
                        "dataset": ds,
                        "stage": stage,
                        "temperature": t,
                        "ece": met["ece"],
                        "log_loss": met["log_loss"],
                        "brier": met["brier"],
                        "accuracy": met["accuracy"],
                    }
                )
                for r in reliability_table(y, expit(z), n_bins):
                    bins.append({"model": m, "dataset": ds, "stage": stage, **r})
    return pd.DataFrame(summary), pd.DataFrame(bins)  # fmt: skip


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float | np.floating):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
