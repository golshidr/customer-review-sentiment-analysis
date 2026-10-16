"""CheckList-style behavioural tests (Ribeiro et al., 2020), implemented without extra libraries.

Three test types:

* **MFT** (minimum functionality): templated sentences with a known label.
* **INV** (invariance): a label-preserving edit (typos, an irrelevant sentence) should not flip the
  prediction.
* **DIR** (directional expectation): an edit with a known direction (an intensifier, an added
  complaint) should not move the score the wrong way by more than a tolerance.

All tests work with any object exposing ``margin(texts) -> log-odds``.
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .preprocessing import light_clean

NOUNS = ["movie", "film", "product", "service", "food", "staff", "hotel", "phone"]
POS_ADJ = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "enjoyable",
           "impressive"]  # fmt: skip
NEG_ADJ = ["bad", "terrible", "awful", "boring", "disappointing", "horrible", "poor", "mediocre"]
INTENSIFIERS = ["very", "really", "extremely", "incredibly"]
NEUTRAL_SUFFIXES = [
    "I went there on a Tuesday.",
    "My friend recommended it to me.",
    "I read about it online last week.",
    "It was the first time I tried it.",
]
NEGATIVE_SUFFIXES = [
    "Overall, it was a waste of money.",
    "I would not recommend it to anyone.",
    "Honestly, it was a big disappointment.",
]
POSITIVE_SUFFIXES = [
    "Overall, I loved it.",
    "I would definitely recommend it.",
    "Honestly, it was a great experience.",
]


@dataclass
class TestResult:
    name: str
    kind: str
    description: str
    n: int
    failures: int
    examples: list[dict]

    @property
    def failure_rate(self) -> float:
        return self.failures / self.n if self.n else float("nan")

    def to_dict(self) -> dict:
        return {
            "test": self.name,
            "type": self.kind,
            "description": self.description,
            "n": self.n,
            "failures": self.failures,
            "failure_rate": round(self.failure_rate, 4),
            "examples": self.examples,
        }


# --------------------------------------------------------------------------------- templates
def _fill(template: str, adjectives: list[str]) -> list[str]:
    return [template.format(noun=n, adj=a) for n, a in itertools.product(NOUNS, adjectives)]


def _contrast(first: list[str], second: list[str]) -> list[str]:
    return [
        f"I expected the {n} to be {a}, but it was {b}."
        for n in NOUNS
        for a, b in zip(first, second, strict=True)
    ]


def mft_cases() -> dict[str, tuple[str, list[str], int]]:
    """name -> (description, sentences, expected label)."""
    return {
        "simple_positive": ("'The {noun} was {pos}.' -> positive",
                            _fill("The {noun} was {adj}.", POS_ADJ), 1),
        "simple_negative": ("'The {noun} was {neg}.' -> negative",
                            _fill("The {noun} was {adj}.", NEG_ADJ), 0),
        "negated_positive": ("'The {noun} was not {pos}.' -> negative",
                             _fill("The {noun} was not {adj}.", POS_ADJ)
                             + _fill("I didn't think the {noun} was {adj}.", POS_ADJ), 0),
        "negated_negative": ("'The {noun} was not {neg}.' -> positive (lenient: may be neutral)",
                             _fill("The {noun} was not {adj}.", NEG_ADJ)
                             + _fill("I didn't think the {noun} was {adj}.", NEG_ADJ), 1),
        "contrast_neg_then_pos": ("'I expected the {noun} to be {neg}, but it was {pos}.' -> pos",
                                  _contrast(NEG_ADJ, POS_ADJ), 1),
        "contrast_pos_then_neg": ("'I expected the {noun} to be {pos}, but it was {neg}.' -> neg",
                                  _contrast(POS_ADJ, NEG_ADJ), 0),
    }  # fmt: skip


# ----------------------------------------------------------------------------- perturbations
def add_typos(text: str, rate: float = 0.1, seed: int = 0) -> str:
    """Swap two adjacent inner characters in a ``rate`` share of words with >= 4 letters."""
    rng = random.Random(seed)
    words = light_clean(text).split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    k = max(1, round(rate * len(candidates))) if candidates else 0
    for i in rng.sample(candidates, min(k, len(candidates))):
        w = words[i]
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def append_sentence(text: str, sentence: str) -> str:
    text = light_clean(text)
    sep = "" if re.search(r"[.!?]$", text) else "."
    return f"{text}{sep} {sentence}"


# -------------------------------------------------------------------------------------- runner
def _examples(texts, margins, k=3, extra=None):
    out = []
    for i in range(min(k, len(texts))):
        row = {"text": texts[i][:300], "p_positive": round(float(expit(margins[i])), 4)}
        if extra is not None:
            row.update(extra[i])
        out.append(row)
    return out


def run_mft(predictor) -> list[TestResult]:
    results = []
    for name, (desc, sentences, label) in mft_cases().items():
        m = predictor.margin(sentences)
        wrong = (m > 0).astype(int) != label
        idx = np.flatnonzero(wrong)
        results.append(
            TestResult(
                name,
                "MFT",
                desc,
                len(sentences),
                int(wrong.sum()),
                _examples([sentences[i] for i in idx], m[idx]),
            )
        )
    return results


def run_invariance(predictor, texts: list[str], seed: int = 0) -> list[TestResult]:
    base = predictor.margin([light_clean(t) for t in texts])
    rng = random.Random(seed)
    tests = {
        "typos_10pct": ("swap adjacent letters in 10% of words; prediction should not flip",
                        [add_typos(t, 0.10, seed + i) for i, t in enumerate(texts)]),
        "neutral_sentence": ("append an irrelevant sentence; prediction should not flip",
                             [append_sentence(t, rng.choice(NEUTRAL_SUFFIXES)) for t in texts]),
    }  # fmt: skip
    results = []
    for name, (desc, perturbed) in tests.items():
        m = predictor.margin(perturbed)
        flipped = (m > 0) != (base > 0)
        idx = np.flatnonzero(flipped)
        results.append(
            TestResult(name, "INV", desc, len(texts), int(flipped.sum()),
                       _examples([perturbed[i] for i in idx], m[idx],
                                 extra=[{"p_positive_before": round(float(expit(base[i])), 4)}
                                        for i in idx]))
        )  # fmt: skip
    return results


def run_directional(
    predictor, texts: list[str], tol: float = 0.1, seed: int = 0
) -> list[TestResult]:
    """DIR tests. A failure is a move of more than ``tol`` in probability in the wrong direction."""
    rng = random.Random(seed)
    results = []
    # 1) intensifiers on templated sentences: confidence in the expected class must not drop
    base_s, inten_s, labels = [], [], []
    for adjs, y in ((POS_ADJ, 1), (NEG_ADJ, 0)):
        for n, a in itertools.product(NOUNS, adjs):
            base_s.append(f"The {n} was {a}.")
            inten_s.append(f"The {n} was {rng.choice(INTENSIFIERS)} {a}.")
            labels.append(y)
    y = np.asarray(labels)
    p0 = expit(predictor.margin(base_s))
    p1 = expit(predictor.margin(inten_s))
    conf0 = np.where(y == 1, p0, 1 - p0)
    conf1 = np.where(y == 1, p1, 1 - p1)
    fail = conf1 < conf0 - tol
    idx = np.flatnonzero(fail)
    results.append(
        TestResult(
            "intensifier",
            "DIR",
            f"add very/really/...; expected-class prob must not drop > {tol}",
            len(base_s),
            int(fail.sum()),
            _examples([inten_s[i] for i in idx], np.log(p1[idx] / (1 - p1[idx]))),
        )
    )
    # 2) append a complaint to reviews the model calls positive: P(pos) must not rise
    # 3) append praise to reviews the model calls negative: P(pos) must not fall
    base = predictor.margin([light_clean(t) for t in texts])
    for name, keep, suffixes, sign in (
        ("add_negative_sentence", base > 0, NEGATIVE_SUFFIXES, 1),
        ("add_positive_sentence", base <= 0, POSITIVE_SUFFIXES, -1),
    ):
        sel = [t for t, k in zip(texts, keep, strict=True) if k]
        before = expit(base[keep])
        pert = [append_sentence(t, rng.choice(suffixes)) for t in sel]
        after = expit(predictor.margin(pert))
        fail = sign * (after - before) > tol
        idx = np.flatnonzero(fail)
        direction = "rise" if sign == 1 else "fall"
        results.append(
            TestResult(
                name,
                "DIR",
                f"P(positive) must not {direction} by more than {tol}",
                len(sel),
                int(fail.sum()),
                _examples([pert[i] for i in idx], np.log(after[idx] / (1 - after[idx]))),
            )
        )
    return results


def run_suite(predictor, texts: list[str], seed: int = 0) -> list[TestResult]:
    return (
        run_mft(predictor)
        + run_invariance(predictor, texts, seed)
        + run_directional(predictor, texts, seed=seed)
    )
