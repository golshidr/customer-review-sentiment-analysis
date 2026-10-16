"""Text cleaning helpers.

``clean_text`` is the cleaning function from the original notebook and is kept unchanged so that
the TF-IDF baseline can be reproduced exactly. ``light_clean`` is used for transformer models,
which work better on near-raw text (case and punctuation carry sentiment signal).
"""

from __future__ import annotations

import re

_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_WS_RE = re.compile(r"\s+")

# Words and contractions that flip or weaken polarity. Note that scikit-learn's built-in English
# stop-word list contains several of these ("not", "no", "nor", "never", ...), so a vectorizer
# configured with ``stop_words="english"`` cannot see negation at all.
NEGATION_TERMS = frozenset(
    {
        "not", "no", "never", "nor", "none", "nothing", "nobody", "nowhere", "neither",
        "cannot", "without", "hardly", "barely", "scarcely",
        "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't",
        "wouldn't", "couldn't", "shouldn't", "can't", "haven't", "hasn't", "hadn't", "ain't",
    }
)  # fmt: skip
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_text(text: str) -> str:
    """Original notebook cleaning: lowercase, drop HTML tags, keep letters only, squeeze spaces."""
    text = str(text).lower()
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def light_clean(text: str) -> str:
    """Minimal normalisation for transformer models: replace HTML tags (``<br />``) by spaces."""
    return _WS_RE.sub(" ", _HTML_RE.sub(" ", str(text))).strip()


def normalize_for_dedup(text: str) -> str:
    """Canonical form used to detect exact duplicates (case, markup and spacing insensitive)."""
    return _WS_RE.sub(" ", _HTML_RE.sub(" ", str(text).lower())).strip()


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokens that keep internal apostrophes (``didn't`` stays one token)."""
    return _TOKEN_RE.findall(light_clean(text).lower().replace("’", "'"))


def has_negation(text: str) -> bool:
    """True if the text contains at least one negation term (see :data:`NEGATION_TERMS`)."""
    return any(tok in NEGATION_TERMS for tok in tokenize_words(text))


def word_count(text: str) -> int:
    """Number of whitespace-separated words after HTML removal."""
    return len(light_clean(text).split())
