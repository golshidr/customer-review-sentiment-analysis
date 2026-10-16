"""Word-level explanations and a deletion-based faithfulness test.

Attribution methods
-------------------
* ``lr_attributions`` - exact per-word contributions of the TF-IDF + LR model
  (``tfidf value x coefficient``; they sum to the log-odds minus the intercept).
* ``integrated_gradients`` - Integrated Gradients (Sundararajan et al., 2017) on the input
  embeddings of a fine-tuned transformer, computed with Captum. The attributed quantity is the
  positive-class log-odds, so the sign convention matches the LR contributions. The baseline
  replaces every non-special token by the padding token.
* ``gradient_x_input`` - the cheaper gradient-times-embedding approximation.

Sub-word scores are summed into whitespace-separated words so all methods (and both model
families) are compared on the same units.

Faithfulness
------------
``deletion_curves`` removes the words an explanation ranks as most supportive of the predicted
class and records how much the predicted-class probability drops (*comprehensiveness*, DeYoung
et al., 2020). A faithful explanation should cause a much larger drop than deleting the same number
of random words.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
from scipy.special import expit

from .preprocessing import light_clean


# ------------------------------------------------------------------------------------ utilities
def split_words(text: str) -> tuple[str, list[str], list[int]]:
    """Return the cleaned text, its whitespace words and each word's start offset."""
    clean = light_clean(text)
    words, starts, pos = [], [], 0
    for w in clean.split(" "):
        if w:
            words.append(w)
            starts.append(pos)
        pos += len(w) + 1
    return clean, words, starts


def tokens_to_words(offsets, token_scores, starts: list[int], n_words: int) -> np.ndarray:
    """Sum sub-word scores into words using the tokenizer's character offsets."""
    scores = np.zeros(n_words)
    for (a, b), s in zip(offsets, token_scores, strict=True):
        if b <= a:  # special tokens have empty spans
            continue
        wi = bisect.bisect_right(starts, a) - 1
        if 0 <= wi < n_words:
            scores[wi] += s
    return scores


# ---------------------------------------------------------------------------------- LR model
def lr_attributions(pipeline, text: str) -> tuple[list[str], np.ndarray]:
    from .models.classical import word_contributions

    _, words, _ = split_words(text)
    return words, word_contributions(pipeline, words)


# ------------------------------------------------------------------------------- transformers
def _encode_with_offsets(predictor, clean: str):
    import torch

    tok = predictor.tokenizer
    enc = tok(
        clean,
        truncation=True,
        max_length=predictor.max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    device = next(predictor.model.parameters()).device
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    special = torch.tensor(sorted(set(tok.all_special_ids)), device=device)
    is_special = torch.isin(ids, special)
    return ids, mask, offsets, is_special


def _margin_forward(model):
    def forward(input_ids=None, attention_mask=None, inputs_embeds=None):
        out = model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        logits = out.logits.float()
        return logits[:, 1] - logits[:, 0]

    return forward


def integrated_gradients(
    predictor, text: str, n_steps: int = 50, internal_batch_size: int = 25
) -> tuple[list[str], np.ndarray, float]:
    """Word attributions from Integrated Gradients; also returns the convergence delta."""
    import torch
    from captum.attr import LayerIntegratedGradients

    model = predictor.model
    model.eval()
    clean, words, starts = split_words(text)
    ids, mask, offsets, is_special = _encode_with_offsets(predictor, clean)
    baseline = torch.where(is_special, ids, torch.full_like(ids, predictor.tokenizer.pad_token_id))
    fwd = _margin_forward(model)
    lig = LayerIntegratedGradients(
        lambda i, m: fwd(input_ids=i, attention_mask=m), model.get_input_embeddings()
    )
    attr, delta = lig.attribute(
        ids,
        baselines=baseline,
        additional_forward_args=(mask,),
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    token_scores = attr.sum(-1)[0].detach().float().cpu().numpy()
    return words, tokens_to_words(offsets, token_scores, starts, len(words)), float(delta.item())


def gradient_x_input(predictor, text: str) -> tuple[list[str], np.ndarray]:
    """Word attributions from gradient x input embedding."""
    model = predictor.model
    model.eval()
    clean, words, starts = split_words(text)
    ids, mask, offsets, _ = _encode_with_offsets(predictor, clean)
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    out = _margin_forward(model)(attention_mask=mask, inputs_embeds=emb)
    out.sum().backward()
    token_scores = (emb.grad * emb).sum(-1)[0].detach().float().cpu().numpy()
    model.zero_grad(set_to_none=True)
    return words, tokens_to_words(offsets, token_scores, starts, len(words))


# ---------------------------------------------------------------------------------- faithfulness
def deletion_curves(
    predictor,
    texts: list[str],
    attributions: list[np.ndarray],
    fractions=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Predicted-class probability after deleting the top-``k`` (or ``k`` random) words.

    Returns arrays of shape ``(n_texts, n_fractions)`` for keys ``"top"`` and ``"random"``,
    plus ``"original"`` (probability of the predicted class before deletion).
    """
    rng = np.random.default_rng(seed)
    batch, index = [], []
    word_lists = [split_words(t)[1] for t in texts]
    orig = predictor.margin([" ".join(w) for w in word_lists])
    sign = np.where(orig > 0, 1.0, -1.0)
    for i, (words, scores) in enumerate(zip(word_lists, attributions, strict=True)):
        n = len(words)
        top_order = np.argsort(-sign[i] * np.asarray(scores), kind="stable")
        rand_order = rng.permutation(n)
        for fi, frac in enumerate(fractions):
            k = math.ceil(frac * n)
            for mode, order in (("top", top_order), ("random", rand_order)):
                drop = set(order[:k].tolist())
                batch.append(" ".join(w for j, w in enumerate(words) if j not in drop))
                index.append((i, fi, mode))
    margins = predictor.margin(batch)
    out = {m: np.zeros((len(texts), len(fractions))) for m in ("top", "random")}
    for (i, fi, mode), z in zip(index, margins, strict=True):
        out[mode][i, fi] = expit(sign[i] * z)
    out["original"] = expit(sign * orig)
    out["fractions"] = np.asarray(fractions)
    return out


def comprehensiveness(curves: dict) -> dict[str, float]:
    """Mean drop in predicted-class probability, averaged over the non-zero deletion fractions."""
    base = curves["original"][:, None]
    nz = curves["fractions"] > 0
    return {f"aopc_{m}": float((base - curves[m])[:, nz].mean()) for m in ("top", "random")}
