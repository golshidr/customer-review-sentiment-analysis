import numpy as np
import pytest

from review_sentiment.explain import split_words
from review_sentiment.models import classical


@pytest.fixture()
def fitted(imdb_frames):
    train, _ = imdb_frames
    cfg = classical.TfidfLRConfig(max_features=500, stop_words=None)
    return classical.fit(cfg, train.text, train.label)


def test_default_config_matches_original_notebook():
    pipe = classical.build_pipeline()
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    assert vec.max_features == 10000
    assert vec.ngram_range == (1, 2)
    assert vec.stop_words == "english"
    assert clf.C == 1.0 and clf.max_iter == 1000


def test_pipeline_learns_toy_sentiment(fitted, imdb_frames):
    _, test = imdb_frames
    acc = (fitted.predict(list(test.text)) == test.label.to_numpy()).mean()
    assert acc > 0.95


def test_word_contributions_sum_to_logit(fitted):
    text = "The acting was wonderful but the ending was awful.<br />Loved the music."
    _, words, _ = split_words(text)
    scores = classical.word_contributions(fitted, words)
    margin = classical.predict_margin(fitted, [" ".join(words)])[0]
    intercept = fitted.named_steps["clf"].intercept_[0]
    assert scores.shape == (len(words),)
    assert np.isclose(scores.sum(), margin - intercept, atol=1e-8)
    assert scores[words.index("wonderful")] > 0
    assert scores[words.index("awful.")] < 0


def test_top_coefficients(fitted):
    top = classical.top_coefficients(fitted, k=3)
    assert len(top) == 6
    assert (top[top.direction == "positive"].weight > 0).all()
    assert (top[top.direction == "negative"].weight < 0).all()


def test_grid_search_returns_best_config(imdb_frames):
    train, test = imdb_frames
    base = classical.TfidfLRConfig(max_features=None, stop_words=None)
    best, table = classical.grid_search(
        base,
        {"C": [0.01, 10.0], "sublinear_tf": [False, True]},
        train.text,
        train.label,
        test.text,
        test.label,
        n_jobs=1,
    )
    assert len(table) == 4
    assert table.val_accuracy.iloc[0] == table.val_accuracy.max()
    assert isinstance(best.sublinear_tf, bool) and best.max_features is None


def test_save_and_load_roundtrip(fitted, tmp_path):
    path = tmp_path / "m" / "model.joblib"
    classical.save(fitted, path)
    again = classical.load(path)
    texts = ["great film", "awful film"]
    assert np.allclose(again.decision_function(texts), fitted.decision_function(texts))
