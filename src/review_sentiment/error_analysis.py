"""Slice-based error analysis on the IMDb test set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .preprocessing import has_negation, word_count


def add_slices(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate test reviews with the attributes used for slicing."""
    out = df.copy()
    out["n_words"] = out["text"].map(word_count)
    q = out["n_words"].quantile([0.25, 0.5, 0.75]).round().astype(int).tolist()
    bins = [0, q[0], q[1], q[2], np.inf]
    labels = [f"<= {q[0]}", f"{q[0] + 1}-{q[1]}", f"{q[1] + 1}-{q[2]}", f"> {q[2]}"]
    out["length_bin"] = pd.cut(out["n_words"], bins=bins, labels=labels, include_lowest=True)
    out["negation"] = np.where(out["text"].map(has_negation), "has negation", "no negation")
    out["contrast"] = np.where(
        out["text"].str.contains(r"\b(?:but|however|although|though)\b", case=False, regex=True),
        "has but/however", "no but/however",
    )  # fmt: skip
    if "rating" in out and out["rating"].notna().all():
        out["rating_group"] = out["rating"].map(
            lambda r: "1 star" if r == 1 else ("10 stars" if r == 10 else f"{int(r)} stars")
        )
        out["strength"] = np.where(
            out["rating"].isin([4, 7]), "mild (4 or 7)", "strong (1-3, 8-10)"
        )
    return out


def slice_accuracy(df: pd.DataFrame, correct_cols: dict[str, str], slice_col: str) -> pd.DataFrame:
    """Accuracy of each model (column of booleans) within each value of ``slice_col``."""
    rows = []
    for value, part in df.groupby(slice_col, observed=True, sort=True):
        row = {"slice": slice_col, "value": str(value), "n": len(part)}
        for model, col in correct_cols.items():
            row[model] = float(part[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def hardest_errors(df: pd.DataFrame, margin_col: str, k: int = 10) -> pd.DataFrame:
    """Misclassified reviews on which the model was most confident."""
    p = expit(df[margin_col].to_numpy())
    pred = (p > 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    wrong = df.assign(pred=pred, confidence=conf)[pred != df["label"].to_numpy()]
    return wrong.sort_values("confidence", ascending=False).head(k)


def truncation_ablation(
    y: np.ndarray, margin_head: np.ndarray, margin_head_tail: np.ndarray, truncated: np.ndarray
) -> pd.DataFrame:
    """Compare head-only and head+tail truncation, separately on the reviews that exceed the
    token limit (where the two inputs differ) and on the rest (where they are identical)."""
    from .evaluation import mcnemar_test

    rows = []
    for name, mask in (("longer than limit", truncated), ("within limit", ~truncated),
                       ("all", np.ones_like(truncated, dtype=bool))):  # fmt: skip
        yh, ph = y[mask], (margin_head[mask] > 0).astype(int)
        pht = (margin_head_tail[mask] > 0).astype(int)
        test = mcnemar_test(yh, pht, ph)
        rows.append(
            {
                "subset": name,
                "n": int(mask.sum()),
                "acc_head": float((ph == yh).mean()),
                "acc_head_tail": float((pht == yh).mean()),
                "only_head_tail_correct": test["a_only_correct"],
                "only_head_correct": test["b_only_correct"],
                "mcnemar_p": test["p_value"],
            }
        )
    return pd.DataFrame(rows)
