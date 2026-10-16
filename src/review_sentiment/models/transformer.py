"""Fine-tuning pretrained transformer encoders for binary sentiment classification.

A small, explicit training loop (AdamW, linear warm-up/decay, bf16 autocast, length-bucketed
batches) is used instead of ``transformers.Trainer`` so that every step is visible and the code is
stable across library versions. The checkpoint with the best validation accuracy is kept.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..preprocessing import light_clean

log = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    name: str
    model_name: str
    max_length: int = 512
    truncation: str = "head"  # "head" (first tokens) or "head_tail" (first + last tokens)
    head_tokens: int = 128  # tokens kept from the start when truncation == "head_tail"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 2
    batch_size: int = 16
    eval_batch_size: int = 128
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    seed: int = 42
    bf16: bool = True
    max_train_samples: int | None = None
    log_every: int = 200

    @classmethod
    def from_dict(cls, d: dict) -> TransformerConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (seeds third-party code that uses the global RNG)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_texts(
    tokenizer, texts, max_length: int = 512, truncation: str = "head", head_tokens: int = 128
) -> list[list[int]]:
    """Tokenise reviews, truncating long ones either at the end or in the middle."""
    texts = [light_clean(t) for t in texts]
    if truncation == "head":
        return tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]
    if truncation != "head_tail":
        raise ValueError(f"unknown truncation {truncation!r}")
    # special tokens added around a single sequence, e.g. [CLS] ... [SEP] or <s> ... </s>
    empty = tokenizer("", add_special_tokens=True)["input_ids"]
    prefix, suffix = empty[:1], empty[1:]
    budget = max_length - len(empty)
    if not 0 < head_tokens < budget:
        raise ValueError("head_tokens must be between 0 and the token budget")
    raw = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    out = []
    for ids in raw:
        if len(ids) > budget:
            ids = ids[:head_tokens] + ids[len(ids) - (budget - head_tokens) :]
        out.append(prefix + ids + suffix)
    return out


def _batches(lengths, batch_size: int, shuffle: bool, seed: int = 0):
    """Yield index batches of similar length (less padding = faster)."""
    idx = np.arange(len(lengths))
    if not shuffle:
        order = idx[np.argsort(lengths, kind="stable")]
        return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for start in range(0, len(idx), chunk):
        part = idx[start : start + chunk]
        part = part[np.argsort(np.asarray(lengths)[part], kind="stable")]
        batches.extend(part[i : i + batch_size] for i in range(0, len(part), batch_size))
    rng.shuffle(batches)
    return batches


def _collate(seqs: list[list[int]], pad_id: int, device):
    import torch

    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def predict_logits(model, encoded, pad_id: int, batch_size: int = 128, bf16: bool = True):
    """Return an ``(n, 2)`` float array of logits for pre-encoded inputs."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    out = np.zeros((len(encoded), model.config.num_labels), dtype=np.float32)
    lengths = [len(s) for s in encoded]
    with torch.inference_mode():
        for b in _batches(lengths, batch_size, shuffle=False):
            ids, mask = _collate([encoded[i] for i in b], pad_id, device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"
            ):
                logits = model(input_ids=ids, attention_mask=mask).logits
            out[b] = logits.float().cpu().numpy()
    return out


def margin_from_logits(logits: np.ndarray) -> np.ndarray:
    """Positive-class log-odds from two-class logits."""
    return logits[:, 1] - logits[:, 0]


def load(model_dir: Path, device: str = "cuda"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    return model, tok


def train(cfg: TransformerConfig, train_df, val_df, out_dir: Path, device: str = "cuda") -> dict:
    """Fine-tune ``cfg.model_name``; save the best-validation checkpoint to ``out_dir``."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    # Load master weights in float32: recent transformers versions keep the checkpoint's dtype by
    # default, and e.g. DeBERTa-v3 is stored in float16, which makes AdamW training diverge (NaN).
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2, dtype=torch.float32
    )
    model.config.id2label = {0: "negative", 1: "positive"}
    model.config.label2id = {"negative": 0, "positive": 1}
    model.to(device)

    if cfg.max_train_samples:
        train_df = train_df.sample(n=cfg.max_train_samples, random_state=cfg.seed)
    enc_kw = {
        "max_length": cfg.max_length,
        "truncation": cfg.truncation,
        "head_tokens": cfg.head_tokens,
    }
    t0 = time.perf_counter()
    x_train = encode_texts(tok, train_df["text"], **enc_kw)
    x_val = encode_texts(tok, val_df["text"], **enc_kw)
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    log.info(
        "Tokenised %d + %d reviews in %.1fs", len(x_train), len(x_val), time.perf_counter() - t0
    )

    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(groups, lr=cfg.learning_rate)
    steps_per_epoch = math.ceil(len(x_train) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_ratio * total), total)
    pad_id = tok.pad_token_id
    lengths = [len(s) for s in x_train]

    history, best_acc, step = [], -1.0, 0
    t_train = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for bi, b in enumerate(_batches(lengths, cfg.batch_size, True, cfg.seed + epoch)):
            ids, mask = _collate([x_train[i] for i in b], pad_id, device)
            labels = torch.as_tensor(y_train[b], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item()
            step += 1
            if (bi + 1) % cfg.log_every == 0:
                log.info(
                    "epoch %d step %d/%d loss %.4f",
                    epoch + 1,
                    bi + 1,
                    steps_per_epoch,
                    running / cfg.log_every,
                )
                running = 0.0
        val_logits = predict_logits(model, x_val, pad_id, cfg.eval_batch_size, cfg.bf16)
        val_acc = float((val_logits.argmax(1) == y_val).mean())
        history.append(
            {
                "epoch": epoch + 1,
                "val_accuracy": val_acc,
                "elapsed_s": round(time.perf_counter() - t_train, 1),
            }
        )
        log.info("epoch %d validation accuracy %.4f", epoch + 1, val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    return {
        "history": history,
        "best_val_accuracy": best_acc,
        "train_seconds": round(time.perf_counter() - t_train, 1),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "n_train": len(x_train),
    }
