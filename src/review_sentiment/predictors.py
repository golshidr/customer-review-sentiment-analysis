"""A common ``texts -> log-odds`` interface over the classical and transformer models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import ARTIFACTS_DIR, RESULTS_DIR


@dataclass
class ClassicalPredictor:
    name: str
    pipeline: object
    kind: str = "classical"

    def margin(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.decision_function(list(texts))


@dataclass
class TransformerPredictor:
    name: str
    model: object
    tokenizer: object
    max_length: int = 512
    truncation: str = "head"
    head_tokens: int = 128
    batch_size: int = 128
    kind: str = "transformer"

    def encode(self, texts: list[str]) -> list[list[int]]:
        from .models.transformer import encode_texts

        return encode_texts(
            self.tokenizer, texts, self.max_length, self.truncation, self.head_tokens
        )

    def margin(self, texts: list[str]) -> np.ndarray:
        from .models.transformer import margin_from_logits, predict_logits

        enc = self.encode(list(texts))
        logits = predict_logits(self.model, enc, self.tokenizer.pad_token_id, self.batch_size)
        return margin_from_logits(logits)


def load_predictor(
    name: str, artifacts_dir: Path = ARTIFACTS_DIR, results_dir: Path = RESULTS_DIR, device=None
):
    """Load a trained model by its config ``name`` (e.g. ``tfidf_lr_tuned`` or ``roberta``)."""
    model_dir = Path(artifacts_dir) / name
    if (model_dir / "model.joblib").exists():
        from .models import classical

        return ClassicalPredictor(name, classical.load(model_dir / "model.joblib"))
    if (model_dir / "config.json").exists():
        import torch

        from .models import transformer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, tok = transformer.load(model_dir, device)
        cfg = {}
        metrics = Path(results_dir) / "metrics" / f"{name}.json"
        if metrics.exists():
            cfg = json.loads(metrics.read_text()).get("config", {})
        return TransformerPredictor(
            name,
            model,
            tok,
            max_length=cfg.get("max_length", 512),
            truncation=cfg.get("truncation", "head"),
            head_tokens=cfg.get("head_tokens", 128),
        )
    raise FileNotFoundError(f"No trained model named {name!r} in {artifacts_dir}")
