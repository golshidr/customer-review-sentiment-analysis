"""Tests for transformer helpers that need no model download."""

import numpy as np
import pytest

from review_sentiment.models.transformer import (
    TransformerConfig,
    _batches,
    encode_texts,
    margin_from_logits,
)


class StubTokenizer:
    """Whitespace tokenizer with [CLS]=1 / [SEP]=2 special tokens; word ids start at 10."""

    def __call__(self, texts, truncation=False, max_length=None, add_special_tokens=True):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        out = []
        for t in texts:
            ids = [10 + i for i, _ in enumerate(t.split())]
            if add_special_tokens:
                if truncation and max_length:
                    ids = ids[: max_length - 2]
                ids = [1, *ids, 2]
            out.append(ids)
        return {"input_ids": out[0] if single else out}


def test_head_truncation_keeps_start():
    enc = encode_texts(StubTokenizer(), ["a " * 50], max_length=12, truncation="head")
    assert len(enc[0]) == 12 and enc[0][0] == 1 and enc[0][-1] == 2
    assert enc[0][1:-1] == list(range(10, 20))


def test_head_tail_truncation_keeps_start_and_end():
    enc = encode_texts(
        StubTokenizer(), ["w " * 50], max_length=12, truncation="head_tail", head_tokens=4
    )[0]
    assert len(enc) == 12 and enc[0] == 1 and enc[-1] == 2
    body = enc[1:-1]
    assert body[:4] == [10, 11, 12, 13]  # first four tokens
    assert body[4:] == list(range(10 + 50 - 6, 10 + 50))  # last six tokens


def test_short_texts_are_not_truncated():
    enc = encode_texts(
        StubTokenizer(), ["two words"], max_length=12, truncation="head_tail", head_tokens=4
    )
    assert enc[0] == [1, 10, 11, 2]


def test_invalid_truncation():
    with pytest.raises(ValueError):
        encode_texts(StubTokenizer(), ["x"], truncation="middle")


def test_batches_cover_all_indices_once():
    lengths = np.random.default_rng(0).integers(1, 100, 103)
    for shuffle in (False, True):
        batches = _batches(lengths, 10, shuffle=shuffle, seed=1)
        flat = np.concatenate(batches)
        assert sorted(flat.tolist()) == list(range(103))


def test_margin_from_logits():
    assert np.allclose(margin_from_logits(np.array([[1.0, 3.0], [2.0, 0.5]])), [2.0, -1.5])


def test_config_from_dict_ignores_unknown_keys():
    cfg = TransformerConfig.from_dict({"name": "x", "model_name": "y", "epochs": 3, "foo": 1})
    assert cfg.epochs == 3 and cfg.to_dict()["model_name"] == "y"
