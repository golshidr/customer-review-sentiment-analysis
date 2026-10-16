"""Metrics, bootstrap confidence intervals and paired significance tests.

Every model writes its scores as a *margin* (log-odds of the positive class). Probabilities are
``sigmoid(margin)`` and hard predictions are ``margin > 0``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .calibration import expected_calibration_error


def classification_metrics(y_true, margin) -> dict[str, float]:
    """Standard binary metrics computed from log-odds margins."""
    y = np.asarray(y_true).astype(int)
    margin = np.asarray(margin, dtype=float)
    prob = expit(margin)
    pred = (margin > 0).astype(int)
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "log_loss": float(log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
        "ece": float(expected_calibration_error(y, prob)),
    }


def bootstrap_ci(
    y_true,
    y_score,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI of ``metric(y_true, y_score)``."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric(y[idx], s[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, s)), float(lo), float(hi)


def accuracy_from_margin(y, margin) -> float:
    return float(np.mean((np.asarray(margin) > 0).astype(int) == np.asarray(y)))


def paired_bootstrap_diff(
    y_true, pred_a, pred_b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Bootstrap CI for accuracy(A) - accuracy(B) on the same examples."""
    y = np.asarray(y_true)
    ca = (np.asarray(pred_a) == y).astype(float)
    cb = (np.asarray(pred_b) == y).astype(float)
    d = ca - cb
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def mcnemar_test(y_true, pred_a, pred_b) -> dict[str, float]:
    """Exact McNemar test on the discordant pairs of two classifiers.

    ``b`` = A correct and B wrong, ``c`` = A wrong and B correct.
    """
    y = np.asarray(y_true)
    ca = np.asarray(pred_a) == y
    cb = np.asarray(pred_b) == y
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}
