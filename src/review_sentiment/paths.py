"""Project-relative default locations. All of them can be overridden from the CLI."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("REVIEW_SENTIMENT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"  # trained models (git-ignored)
RESULTS_DIR = PROJECT_ROOT / "results"  # small metric/prediction files (committed)
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = PROJECT_ROOT / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"
