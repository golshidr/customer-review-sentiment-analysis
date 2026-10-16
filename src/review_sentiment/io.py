"""Reading and writing predictions and metric files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PREDICTIONS_DIR


def prediction_path(model: str, dataset: str, root: Path = PREDICTIONS_DIR) -> Path:
    return Path(root) / f"{model}__{dataset}.csv.gz"


def save_predictions(
    model: str, dataset: str, ids, labels, margin, root: Path = PREDICTIONS_DIR
) -> Path:
    """Store per-example log-odds so every downstream analysis can run without a GPU."""
    path = prediction_path(model, dataset, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "review_id": list(ids),
            "label": np.asarray(labels).astype(int),
            "margin": np.round(np.asarray(margin, dtype=float), 5),
        }
    )
    df.to_csv(path, index=False, compression="gzip")
    return path


def load_predictions(model: str, dataset: str, root: Path = PREDICTIONS_DIR) -> pd.DataFrame:
    return pd.read_csv(prediction_path(model, dataset, root))


def write_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")
    return path


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")
