import numpy as np

from review_sentiment.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_table,
)
from review_sentiment.evaluation import (
    accuracy_from_margin,
    bootstrap_ci,
    classification_metrics,
    mcnemar_test,
    paired_bootstrap_diff,
)


def test_metrics_perfect_classifier():
    y = np.array([0, 1, 0, 1])
    m = classification_metrics(y, np.array([-5.0, 5.0, -3.0, 4.0]))
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0 and m["roc_auc"] == 1.0
    assert m["n"] == 4


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    z = np.where(rng.random(500) < 0.8, 2 * y - 1, 1 - 2 * y).astype(float)
    point, lo, hi = bootstrap_ci(y, z, accuracy_from_margin, n_boot=300)
    assert lo <= point <= hi
    assert 0.74 < point < 0.86


def test_mcnemar_counts_discordant_pairs():
    y = np.array([1, 1, 1, 1, 0, 0])
    a = np.array([1, 1, 1, 0, 0, 0])
    b = np.array([1, 0, 0, 0, 0, 1])
    r = mcnemar_test(y, a, b)
    assert r["a_only_correct"] == 3 and r["b_only_correct"] == 0
    assert 0 < r["p_value"] <= 1


def test_paired_bootstrap_diff_sign():
    y = np.ones(100, dtype=int)
    a = np.ones(100, dtype=int)
    b = np.r_[np.ones(80), np.zeros(20)].astype(int)
    d = paired_bootstrap_diff(y, a, b, n_boot=200)
    assert np.isclose(d["diff"], 0.2) and d["ci_low"] > 0


def test_ece_zero_when_confidence_equals_accuracy():
    # 100 predictions at confidence 0.8 of which exactly 80 are right
    p = np.full(100, 0.8)
    y = np.r_[np.ones(80), np.zeros(20)].astype(int)
    assert expected_calibration_error(y, p) < 1e-9
    rows = reliability_table(y, p)
    assert len(rows) == 1 and rows[0]["count"] == 100


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(1)
    true_logit = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    overconfident = 3.0 * true_logit
    t = fit_temperature(overconfident, y)
    assert 2.7 < t < 3.3
    assert np.allclose(apply_temperature(overconfident, t), overconfident / t)


def test_truncation_ablation_splits_long_and_short_reviews():
    from review_sentiment.error_analysis import truncation_ablation

    y = np.array([1, 1, 0, 0, 1, 0])
    head = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0])  # wrong on three long reviews
    head_tail = np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0])  # right everywhere
    long = np.array([True, True, True, True, False, False])
    t = truncation_ablation(y, head, head_tail, long).set_index("subset")
    assert t.loc["longer than limit", "n"] == 4
    assert t.loc["longer than limit", "acc_head"] == 0.25
    assert t.loc["longer than limit", "acc_head_tail"] == 1.0
    assert t.loc["longer than limit", "only_head_tail_correct"] == 3
    assert t.loc["within limit", "only_head_correct"] == 0
    assert t.loc["all", "n"] == 6
