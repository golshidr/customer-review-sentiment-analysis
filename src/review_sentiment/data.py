"""Dataset download, parsing and train/validation/test splitting.

The IMDb Large Movie Review Dataset (Maas et al., 2011) is read from the original Stanford archive
rather than from a pre-processed copy, because the archive also ships the star rating and the IMDb
movie URL of every review. The movie identifier is needed to build leakage-free splits.
"""

from __future__ import annotations

import logging
import re
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .paths import PROCESSED_DIR, RAW_DIR
from .preprocessing import normalize_for_dedup

log = logging.getLogger(__name__)

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE = "aclImdb_v1.tar.gz"
COLUMNS = ["review_id", "split", "text", "label", "rating", "movie_id"]

_REVIEW_RE = re.compile(r"aclImdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt$")
_URLS_RE = re.compile(r"aclImdb/(train|test)/urls_(pos|neg)\.txt$")
_MOVIE_RE = re.compile(r"(tt\d+)")


def download_file(url: str, dest: Path, overwrite: bool = False) -> Path:
    """Download ``url`` to ``dest`` (no-op if the file already exists)."""
    dest = Path(dest)
    if dest.exists() and not overwrite:
        log.info("Found %s, skipping download", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    return dest


def parse_imdb_archive(archive: Path) -> pd.DataFrame:
    """Parse the aclImdb tarball into one DataFrame (labelled train and test reviews only).

    Columns: ``review_id`` (e.g. ``train/pos/200``), ``split``, ``text``, ``label`` (1 = positive),
    ``rating`` (1-10 stars), ``movie_id`` (IMDb ``tt`` identifier taken from the URL files).
    """
    reviews: list[dict] = []
    urls: dict[tuple[str, str], list[str]] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            m = _REVIEW_RE.search(member.name)
            if m:
                split, polarity, idx, rating = m.groups()
                text = tar.extractfile(member).read().decode("utf-8")
                reviews.append(
                    {
                        "split": split,
                        "polarity": polarity,
                        "idx": int(idx),
                        "rating": int(rating),
                        "text": text,
                    }
                )
                continue
            m = _URLS_RE.search(member.name)
            if m:
                lines = tar.extractfile(member).read().decode("utf-8").splitlines()
                urls[m.groups()] = lines
    df = pd.DataFrame(reviews)
    if df.empty:
        raise ValueError(f"No reviews found in {archive}")

    def movie_of(row) -> str | None:
        lines = urls.get((row.split, row.polarity))
        if lines is None or row.idx >= len(lines):
            return None
        m = _MOVIE_RE.search(lines[row.idx])
        return m.group(1) if m else None

    df["movie_id"] = [movie_of(r) for r in df.itertuples(index=False)]
    df["label"] = (df["polarity"] == "pos").astype(int)
    df["review_id"] = df["split"] + "/" + df["polarity"] + "/" + df["idx"].astype(str)
    df = df.sort_values(["split", "polarity", "idx"], ascending=[False, True, True])
    return df[COLUMNS].reset_index(drop=True)


def prepare_imdb(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    """Download the archive (if needed) and write ``imdb_train.parquet`` / ``imdb_test.parquet``."""
    archive = download_file(IMDB_URL, Path(raw_dir) / IMDB_ARCHIVE)
    df = parse_imdb_archive(archive)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "test"):
        path = processed_dir / f"imdb_{split}.parquet"
        df[df["split"] == split].reset_index(drop=True).to_parquet(path, index=False)
        out[split] = path
        log.info("Wrote %s (%d rows)", path, int((df["split"] == split).sum()))
    return out


def load_imdb(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed official IMDb train and test tables."""
    processed_dir = Path(processed_dir)
    paths = [processed_dir / f"imdb_{s}.parquet" for s in ("train", "test")]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run `python -m review_sentiment download` (or `make data`) first."
        )
    return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])


def load_imdb_from_hub() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback loader using the Hugging Face copy (no ratings or movie ids available)."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    frames = []
    for split in ("train", "test"):
        part = ds[split].to_pandas()
        frames.append(
            pd.DataFrame(
                {
                    "review_id": [f"{split}/hf/{i}" for i in range(len(part))],
                    "split": split,
                    "text": part["text"],
                    "label": part["label"].astype(int),
                    "rating": np.nan,
                    "movie_id": None,
                }
            )
        )
    return frames[0], frames[1]


@dataclass
class Splits:
    """Train/validation/test frames plus bookkeeping about how they were built."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    protocol: str
    info: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "protocol": self.protocol,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            **self.info,
        }


PROTOCOLS = ("official", "legacy_random", "grouped_pooled")


def _group_holdout(
    df: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-stratified hold-out of roughly ``fraction`` of rows with no movie in both parts."""
    if df["movie_id"].isna().any():
        log.warning("movie_id missing; falling back to a random (non-grouped) hold-out")
        return train_test_split(df, test_size=fraction, random_state=seed, stratify=df["label"])
    n_splits = max(2, round(1 / fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(sgkf.split(df, df["label"], groups=df["movie_id"]))
    return df.iloc[keep_idx], df.iloc[hold_idx]


def overlap_stats(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """How much of ``test`` is already 'seen' in ``train`` (same movie or identical text)."""
    seen_text = set(train["text"].map(normalize_for_dedup))
    test_norm = test["text"].map(normalize_for_dedup)
    stats = {"test_exact_duplicates_of_train": int(test_norm.isin(seen_text).sum())}
    if not test["movie_id"].isna().any():
        seen_movies = set(train["movie_id"])
        stats["test_reviews_with_movie_in_train"] = int(test["movie_id"].isin(seen_movies).sum())
        stats["test_share_with_movie_in_train"] = round(
            stats["test_reviews_with_movie_in_train"] / len(test), 4
        )
    return stats


def drop_duplicates_of(df: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows of ``df`` whose normalised text also occurs in ``reference``."""
    seen = set(reference["text"].map(normalize_for_dedup))
    mask = df["text"].map(normalize_for_dedup).isin(seen)
    return df[~mask], int(mask.sum())


def make_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protocol: str = "official",
    val_fraction: float = 0.1,
    seed: int = 42,
    dedupe: bool = True,
) -> Splits:
    """Build train/validation/test splits.

    * ``official`` (default): the dataset authors' train/test split, whose movies are disjoint.
      The validation set is a movie-grouped hold-out of the training set. With ``dedupe=True``,
      duplicate reviews inside the training data and test reviews whose text also appears in the
      training data are removed.
    * ``legacy_random``: reproduces the original notebook - official train and test are pooled and
      re-split at random 80/20 (stratified by label). Reviews of the same movie, and even identical
      reviews, end up on both sides, which inflates test accuracy. Kept only for the leakage study.
    * ``grouped_pooled``: same pool and 80/20 proportions as ``legacy_random`` but split by movie.
      Isolates the effect of movie overlap in the leakage study.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; choose from {PROTOCOLS}")
    info: dict = {}
    if protocol == "legacy_random":
        pooled = pd.concat([train_df, test_df], ignore_index=True)
        train, test = train_test_split(
            pooled, test_size=0.2, random_state=seed, stratify=pooled["label"]
        )
        train, val = train_test_split(
            train, test_size=val_fraction, random_state=seed, stratify=train["label"]
        )
    else:
        if protocol == "official":
            train, test = train_df, test_df
        else:  # grouped_pooled
            pooled = pd.concat([train_df, test_df], ignore_index=True)
            train, test = _group_holdout(pooled, 0.2, seed)
        if dedupe:
            before = len(train)
            train = train[~train["text"].map(normalize_for_dedup).duplicated()]
            info["removed_train_duplicates"] = before - len(train)
            test, info["removed_test_duplicates_of_train"] = drop_duplicates_of(test, train)
        train, val = _group_holdout(train, val_fraction, seed)
    info.update(overlap_stats(pd.concat([train, val]), test))
    return Splits(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        protocol,
        info,
    )


# ------------------------------------------------------------------------------- cross-domain sets
# Test splits of two customer-review corpora (Zhang et al., 2015), read straight from the Hub.
# Labels in both parquet files: 0 = negative, 1 = positive.
CROSS_DOMAIN = {
    "amazon": {
        "repo": "fancyzhx/amazon_polarity",
        "file": "amazon_polarity/test-00000-of-00001.parquet",
    },
    "yelp": {
        "repo": "fancyzhx/yelp_polarity",
        "file": "plain_text/test-00000-of-00001.parquet",
    },
}


def _cross_domain_text(name: str, df: pd.DataFrame) -> pd.Series:
    if name == "amazon":  # review title + body, as written by the customer
        return df["title"].fillna("").str.strip() + ". " + df["content"].fillna("").str.strip()
    # Yelp texts store line breaks as the two characters "\n"
    return df["text"].str.replace("\\n", " ", regex=False)


def prepare_cross_domain(
    n_per_dataset: int = 10000, seed: int = 42, processed_dir: Path = PROCESSED_DIR
) -> dict[str, Path]:
    """Download the Amazon and Yelp polarity test splits and store a label-balanced sample."""
    from huggingface_hub import hf_hub_download

    out = {}
    for name, spec in CROSS_DOMAIN.items():
        local = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset")
        raw = pd.read_parquet(local)
        per_class = n_per_dataset // 2
        parts = [raw[raw["label"] == y].sample(n=per_class, random_state=seed) for y in (0, 1)]
        sample = pd.concat(parts).sample(frac=1.0, random_state=seed)
        df = pd.DataFrame(
            {
                "review_id": [f"{name}/{i}" for i in sample.index],
                "text": _cross_domain_text(name, sample).to_numpy(),
                "label": sample["label"].astype(int).to_numpy(),
            }
        )
        path = Path(processed_dir) / f"{name}_test_sample.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        out[name] = path
        log.info("Wrote %s (%d rows from %d)", path, len(df), len(raw))
    return out


def load_cross_domain(name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = Path(processed_dir) / f"{name}_test_sample.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run `python -m review_sentiment download --cross-domain`"
        )
    return pd.read_parquet(path)
