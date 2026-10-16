"""Leakage study: how much does the original pooled random split inflate test accuracy?

The original notebook pooled the official IMDb train and test sets and re-split them at random.
The dataset authors built the official split so that train and test movies are disjoint; a random
re-split puts reviews of the same movie (and a few hundred exact duplicates) on both sides.

This module trains the *original* TF-IDF + LR configuration under three protocols:

* ``legacy_random`` - the notebook's protocol (several seeds),
* ``grouped_pooled`` - same pool and proportions, but split by movie (several seeds),
* ``official`` - the authors' split with duplicates removed (the protocol used from now on).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import make_splits, overlap_stats
from .models import classical


def _run(protocol: str, seed: int, train_df, test_df, cfg) -> dict:
    s = make_splits(train_df, test_df, protocol=protocol, seed=seed)
    train = pd.concat([s.train, s.val])  # the notebook trained on the full 80 %
    pipe = classical.fit(cfg, train["text"], train["label"])
    acc = float((pipe.predict(list(s.test["text"])) == s.test["label"].to_numpy()).mean())
    return {
        "protocol": protocol,
        "seed": seed,
        "n_train": len(train),
        "n_test": len(s.test),
        "test_accuracy": acc,
        **overlap_stats(train, s.test),
    }


def run_leakage_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seeds=(0, 1, 2, 3, 4),
    n_jobs: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = classical.TfidfLRConfig()  # the notebook's settings
    jobs = [(p, s) for p in ("legacy_random", "grouped_pooled") for s in seeds]
    jobs.append(("official", 0))
    rows = Parallel(n_jobs=n_jobs)(delayed(_run)(p, s, train_df, test_df, cfg) for p, s in jobs)
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby("protocol", sort=False)
        .agg(
            runs=("test_accuracy", "size"),
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0),
            n_test=("n_test", "mean"),
            share_test_movie_in_train=("test_share_with_movie_in_train", "mean"),
            exact_duplicates=("test_exact_duplicates_of_train", "mean"),
        )
        .reset_index()
    )
    return runs, summary
