import pandas as pd
import pytest

from review_sentiment.data import (
    drop_duplicates_of,
    make_splits,
    overlap_stats,
    parse_imdb_archive,
)


def test_parse_archive_reads_labels_ratings_and_movies(toy_archive):
    df = parse_imdb_archive(toy_archive)
    assert len(df) == 6
    assert set(df.split) == {"train", "test"}
    row = df[df.review_id == "train/pos/1"].iloc[0]
    assert row.label == 1 and row.rating == 9 and row.text == "Loved it."
    assert row.movie_id == "tt0011"
    assert df[df.review_id == "test/neg/0"].iloc[0].label == 0


def test_official_split_has_no_movie_overlap_between_train_and_val(imdb_frames):
    train, test = imdb_frames
    s = make_splits(train, test, protocol="official", val_fraction=0.2, seed=0)
    assert set(s.train.movie_id).isdisjoint(set(s.val.movie_id))
    assert len(s.test) <= len(test)
    assert s.info["test_exact_duplicates_of_train"] == 0
    # labels stay roughly balanced in the validation fold
    assert 0.3 < s.val.label.mean() < 0.7


def test_official_split_removes_test_duplicates_of_train(imdb_frames):
    train, test = imdb_frames
    test = pd.concat([test, train.iloc[:5].assign(split="test")], ignore_index=True)
    s = make_splits(train, test, protocol="official", seed=0)
    assert s.info["removed_test_duplicates_of_train"] >= 5
    assert s.info["test_exact_duplicates_of_train"] == 0


def test_legacy_random_split_reproduces_notebook_proportions(imdb_frames):
    train, test = imdb_frames
    s = make_splits(train, test, protocol="legacy_random", val_fraction=0.1, seed=42)
    total = len(train) + len(test)
    assert len(s.test) == int(0.2 * total)
    assert len(s.train) + len(s.val) == total - len(s.test)


def test_grouped_pooled_split_is_movie_disjoint(imdb_frames):
    train, test = imdb_frames
    s = make_splits(train, test, protocol="grouped_pooled", seed=0)
    seen = set(s.train.movie_id) | set(s.val.movie_id)
    assert seen.isdisjoint(set(s.test.movie_id))
    assert (
        overlap_stats(pd.concat([s.train, s.val]), s.test)["test_reviews_with_movie_in_train"] == 0
    )


def test_unknown_protocol_raises(imdb_frames):
    with pytest.raises(ValueError):
        make_splits(*imdb_frames, protocol="nope")


def test_drop_duplicates_of():
    a = pd.DataFrame({"text": ["Hello  world", "unique"]})
    b = pd.DataFrame({"text": ["hello world"]})
    kept, n = drop_duplicates_of(a, b)
    assert n == 1 and kept.text.tolist() == ["unique"]
