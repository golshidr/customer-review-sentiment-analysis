"""TF-IDF + logistic regression baseline (the model from the original notebook)."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..preprocessing import clean_text

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TfidfLRConfig:
    """Hyper-parameters of the TF-IDF + logistic regression pipeline.

    The defaults are exactly the settings used in the original notebook.
    """

    max_features: int | None = 10000
    ngram_min: int = 1
    ngram_max: int = 2
    stop_words: str | None = "english"
    sublinear_tf: bool = False
    min_df: int = 1
    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> TfidfLRConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


def build_pipeline(cfg: TfidfLRConfig | None = None) -> Pipeline:
    """Create an unfitted pipeline that takes raw review text as input."""
    cfg = cfg or TfidfLRConfig()
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=cfg.max_features,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        stop_words=cfg.stop_words,
        sublinear_tf=cfg.sublinear_tf,
        min_df=cfg.min_df,
    )
    clf = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, random_state=cfg.random_state)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def fit(cfg: TfidfLRConfig, texts, labels) -> Pipeline:
    pipe = build_pipeline(cfg)
    pipe.fit(list(texts), np.asarray(labels))
    return pipe


def predict_margin(pipe: Pipeline, texts) -> np.ndarray:
    """Log-odds of the positive class (``decision_function``)."""
    return pipe.decision_function(list(texts))


def _score_config(cfg: TfidfLRConfig, train_x, train_y, val_x, val_y) -> dict:
    pipe = fit(cfg, train_x, train_y)
    acc = float((pipe.predict(val_x) == np.asarray(val_y)).mean())
    return {**cfg.to_dict(), "val_accuracy": acc}


def grid_search(
    base: TfidfLRConfig,
    grid: dict[str, list],
    train_x,
    train_y,
    val_x,
    val_y,
    n_jobs: int = -1,
) -> tuple[TfidfLRConfig, pd.DataFrame]:
    """Exhaustive search over ``grid`` scored by accuracy on a fixed validation set."""
    keys = list(grid)
    candidates = [
        replace(base, **dict(zip(keys, values, strict=True)))
        for values in itertools.product(*grid.values())
    ]
    log.info("Grid search over %d configurations", len(candidates))
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_score_config)(c, list(train_x), train_y, list(val_x), val_y) for c in candidates
    )
    results = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False, kind="stable")
    best = TfidfLRConfig.from_dict(results.iloc[0].to_dict())
    # pandas turns None into NaN; restore proper types
    best = replace(
        best,
        max_features=None if pd.isna(best.max_features) else int(best.max_features),
        stop_words=None if not isinstance(best.stop_words, str) else best.stop_words,
        ngram_min=int(best.ngram_min),
        ngram_max=int(best.ngram_max),
        min_df=int(best.min_df),
        sublinear_tf=bool(best.sublinear_tf),
    )
    return best, results.reset_index(drop=True)


def top_coefficients(pipe: Pipeline, k: int = 15) -> pd.DataFrame:
    """The ``k`` most positive and most negative n-gram weights."""
    names = np.asarray(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    neg, pos = order[:k], order[::-1][:k]
    return pd.concat(
        [
            pd.DataFrame({"term": names[pos], "weight": coef[pos], "direction": "positive"}),
            pd.DataFrame({"term": names[neg], "weight": coef[neg], "direction": "negative"}),
        ],
        ignore_index=True,
    )


def word_contributions(pipe: Pipeline, words: list[str]) -> np.ndarray:
    """Per-word contribution to the positive-class log-odds for a review split into words.

    ``words`` should come from ``light_clean(text).split()`` (HTML already removed). Each active
    n-gram feature contributes ``tfidf_value * coefficient``. That amount is shared evenly across
    the feature's occurrences and, for bigrams, across the words of each occurrence. Words removed
    by cleaning or stop-word filtering get zero. By construction the scores sum to
    ``decision_function - intercept`` (a completeness property tested in ``tests/``).
    """
    vec = pipe.named_steps["tfidf"]
    coef = pipe.named_steps["clf"].coef_[0]
    vocab = vec.vocabulary_
    stop = vec.get_stop_words() or frozenset()
    x = vec.transform([" ".join(words)])
    value_of = dict(zip(x.indices, x.data, strict=True))
    # cleaned analyzer tokens, remembering which original word each one came from
    tokens: list[tuple[int, str]] = []
    for wi, w in enumerate(words):
        for tok in clean_text(w).split():
            if len(tok) >= 2 and tok not in stop:
                tokens.append((wi, tok))
    occurrences: dict[int, list[list[int]]] = {}
    ngram_min, ngram_max = vec.ngram_range
    for n in range(ngram_min, ngram_max + 1):
        for i in range(len(tokens) - n + 1):
            span = tokens[i : i + n]
            j = vocab.get(" ".join(t for _, t in span))
            if j is not None and j in value_of:
                occurrences.setdefault(j, []).append([wi for wi, _ in span])
    scores = np.zeros(len(words))
    for j, spans in occurrences.items():
        per_occurrence = value_of[j] * coef[j] / len(spans)
        for span in spans:
            for wi in span:
                scores[wi] += per_occurrence / len(span)
    return scores


def save(pipe: Pipeline, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
