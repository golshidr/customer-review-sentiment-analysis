import numpy as np

from review_sentiment.explain import (
    comprehensiveness,
    deletion_curves,
    split_words,
    tokens_to_words,
)
from review_sentiment.robustness import (
    add_typos,
    append_sentence,
    mft_cases,
    run_invariance,
    run_mft,
    run_suite,
)


def test_split_words_offsets():
    clean, words, starts = split_words("Great  film<br />really")
    assert words == ["Great", "film", "really"]
    assert [clean[s : s + len(w)] for s, w in zip(starts, words, strict=True)] == words


def test_tokens_to_words_sums_subwords_and_skips_specials():
    _, words, starts = split_words("unbelievably good")
    offsets = [(0, 0), (0, 2), (2, 12), (13, 17), (0, 0)]  # [CLS] un ##believably good [SEP]
    scores = tokens_to_words(offsets, [9.0, 1.0, 2.0, 0.5, 9.0], starts, len(words))
    assert np.allclose(scores, [3.0, 0.5])


def test_deletion_of_top_words_hurts_more_than_random(keyword_predictor):
    texts = ["great excellent plot acting story music film cast scenes the ending"] * 20
    attrs = []
    for t in texts:
        words = split_words(t)[1]
        attrs.append(np.array([1.0 if w in ("great", "excellent") else 0.0 for w in words]))
    curves = deletion_curves(keyword_predictor, texts, attrs, fractions=(0.0, 0.2), seed=0)
    comp = comprehensiveness(curves)
    assert comp["aopc_top"] > comp["aopc_random"]
    assert np.allclose(curves["top"][:, 0], curves["original"])


def test_typos_are_deterministic_and_keep_length():
    t = "This wonderful movie had brilliant acting throughout"
    a, b = add_typos(t, 0.5, seed=3), add_typos(t, 0.5, seed=3)
    assert a == b and a != t
    assert len(a.split()) == len(t.split())
    assert sorted(a.replace(" ", "")) == sorted(t.replace(" ", ""))


def test_append_sentence_adds_period():
    assert append_sentence("Nice", "Bye.") == "Nice. Bye."
    assert append_sentence("Nice!", "Bye.") == "Nice! Bye."


def test_mft_templates_have_expected_labels():
    cases = mft_cases()
    assert cases["simple_positive"][2] == 1 and cases["negated_positive"][2] == 0
    assert all(len(sents) >= 64 for _, sents, _ in cases.values())


def test_keyword_model_fails_negation_but_passes_simple(keyword_predictor):
    res = {r.name: r for r in run_mft(keyword_predictor)}
    assert res["negated_positive"].failure_rate == 1.0  # bag-of-words ignores "not"


def test_full_suite_runs(keyword_predictor):
    texts = ["the film was great and the cast superb", "boring plot and awful acting"] * 5
    results = run_suite(keyword_predictor, texts)
    names = {r.name for r in results}
    assert {"typos_10pct", "neutral_sentence", "intensifier", "add_negative_sentence"} <= names
    inv = {r.name: r for r in run_invariance(keyword_predictor, texts)}
    assert inv["neutral_sentence"].failures == 0
    for r in results:
        d = r.to_dict()
        assert 0 <= d["failure_rate"] <= 1 or np.isnan(d["failure_rate"])
