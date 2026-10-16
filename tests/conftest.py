"""Shared fixtures: a small synthetic review corpus (no downloads needed)."""

from __future__ import annotations

import io
import random
import tarfile

import pandas as pd
import pytest

POS = ["great", "excellent", "wonderful", "loved", "brilliant", "enjoyable", "superb"]
NEG = ["awful", "boring", "terrible", "waste", "poor", "dull", "worst"]
FILLER = ["the", "plot", "acting", "story", "movie", "film", "scenes", "cast", "music", "ending"]


def make_review(rng: random.Random, label: int) -> str:
    words = rng.sample(FILLER, 5) + rng.sample(POS if label else NEG, 3)
    rng.shuffle(words)
    return (
        " ".join(words).capitalize()
        + ".<br /><br />Really "
        + rng.choice(POS if label else NEG)
        + "!"
    )


def make_frame(n: int, split: str, n_movies: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = i % 2
        rows.append(
            {
                "review_id": f"{split}/{'pos' if label else 'neg'}/{i}",
                "split": split,
                "text": make_review(rng, label),
                "label": label,
                "rating": rng.choice([8, 9, 10]) if label else rng.choice([1, 2, 3]),
                "movie_id": f"tt{split[:2]}{rng.randrange(n_movies):04d}",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def imdb_frames():
    train = make_frame(400, "train", 60, 0)
    test = make_frame(200, "test", 30, 1)
    return train, test


@pytest.fixture()
def toy_archive(tmp_path):
    """A miniature aclImdb_v1.tar.gz with the same layout as the real archive."""
    path = tmp_path / "aclImdb_v1.tar.gz"
    reviews = {
        ("train", "pos"): ["A wonderful film.", "Loved it."],
        ("train", "neg"): ["Terrible.", "Boring and dull."],
        ("test", "pos"): ["Great cast."],
        ("test", "neg"): ["Worst ever."],
    }
    with tarfile.open(path, "w:gz") as tar:

        def add(name: str, content: str):
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        for (split, pol), texts in reviews.items():
            urls = []
            for i, t in enumerate(texts):
                rating = 9 if pol == "pos" else 2
                add(f"aclImdb/{split}/{pol}/{i}_{rating}.txt", t)
                code = 1 if split == "train" else 2
                urls.append(f"http://www.imdb.com/title/tt00{code}{i}/usercomments")
            add(f"aclImdb/{split}/urls_{pol}.txt", "\n".join(urls) + "\n")
        add("aclImdb/README", "toy")
    return path


class KeywordPredictor:
    """Deterministic stand-in for a model: +1 per positive word, -1 per negative word."""

    kind = "fake"

    def margin(self, texts):
        import numpy as np

        out = []
        for t in texts:
            words = t.lower().replace(".", " ").replace("!", " ").split()
            out.append(sum(w in POS for w in words) - sum(w in NEG for w in words) + 0.01)
        return np.asarray(out, dtype=float)


@pytest.fixture()
def keyword_predictor():
    return KeywordPredictor()
