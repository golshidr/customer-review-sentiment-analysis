"""Calibration diagnostics and temperature scaling for binary classifiers."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


def _confidence(prob_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prob_pos = np.asarray(prob_pos, dtype=float)
    pred = (prob_pos > 0.5).astype(int)
    conf = np.where(pred == 1, prob_pos, 1.0 - prob_pos)
    return pred, conf


def reliability_table(y_true, prob_pos, n_bins: int = 15) -> list[dict]:
    """Confidence-binned accuracy (top-label reliability diagram data)."""
    y = np.asarray(y_true).astype(int)
    pred, conf = _confidence(prob_pos)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        mask = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        if mask.any():
            rows.append(
                {
                    "bin_low": float(lo),
                    "bin_high": float(hi),
                    "count": int(mask.sum()),
                    "confidence": float(conf[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return rows


def expected_calibration_error(y_true, prob_pos, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins on [0.5, 1]."""
    rows = reliability_table(y_true, prob_pos, n_bins)
    n = len(np.asarray(y_true))
    return float(sum(r["count"] / n * abs(r["accuracy"] - r["confidence"]) for r in rows))


def fit_temperature(margin_val, y_val) -> float:
    """Temperature ``T`` minimising validation NLL of ``sigmoid(margin / T)``."""
    z = np.asarray(margin_val, dtype=float)
    y = np.asarray(y_val).astype(float)

    def nll(log_t: float) -> float:
        p = np.clip(expit(z / np.exp(log_t)), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(margin, temperature: float) -> np.ndarray:
    return np.asarray(margin, dtype=float) / float(temperature)
