"""Render every README figure from the files in ``results/`` (no model or GPU needed)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.ticker
import numpy as np
import pandas as pd

from . import plotting as P
from .analysis import DATASETS
from .io import load_predictions

log = logging.getLogger(__name__)
plt = P.plt

DATASET_LABELS = {"imdb_test": "IMDb (movies, in-domain)", "amazon": "Amazon (products)",
                  "yelp": "Yelp (businesses)"}  # fmt: skip


def _bar_end_labels(ax, ys, values, fmt="{:.1%}", offset=0.002):
    for y, v in zip(ys, values, strict=True):
        ax.text(v + offset, y, fmt.format(v), va="center", ha="left", fontsize=9.5, color=P.INK_2)


# --------------------------------------------------------------------------------- figures
def fig_leakage(results: Path, out: Path):
    runs = pd.read_csv(results / "leakage" / "leakage_runs.csv")
    order = [("legacy_random", "Original protocol:\nrandom re-split of pooled data"),
             ("grouped_pooled", "Same pool, split by movie\n(no shared movies or duplicates)"),
             ("official", "Official movie-disjoint split\n(de-duplicated)")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (proto, _) in enumerate(order):
        acc = runs.loc[runs.protocol == proto, "test_accuracy"].to_numpy()
        color = P.PALETTE[1] if proto == "legacy_random" else P.PALETTE[0]
        jitter = np.linspace(-0.12, 0.12, len(acc)) if len(acc) > 1 else [0]
        ax.scatter(
            acc,
            i + np.asarray(jitter),
            s=42,
            color=color,
            edgecolor=P.SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        ax.plot([acc.mean()] * 2, [i - 0.28, i + 0.28], color=P.INK, lw=2, zorder=4)
        ax.text(
            acc.mean(),
            i - 0.33,
            f"mean {acc.mean():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=P.INK,
        )
    ax.set_yticks(range(len(order)), [o[1] for o in order])
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlabel("Test accuracy of the original TF-IDF + LR model")
    ax.grid(axis="y", visible=False)
    ax.set_title("Random re-splitting inflates accuracy by ~1.3 points")
    ax.text(
        0,
        -0.3,
        "Dots: individual random seeds (5 per protocol); bar: mean.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=P.MUTED,
    )
    return P.save(fig, out / "leakage_effect.png")


def fig_model_comparison(results: Path, out: Path):
    comp = pd.read_csv(results / "comparison.csv")
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(comp) + 1.2))
    ys = np.arange(len(comp))
    colors = [P.FAMILY_COLOR[P.family(m)] for m in comp.model]
    ax.barh(ys, comp.accuracy, height=0.55, color=colors, zorder=2)
    err = np.vstack([comp.accuracy - comp.acc_ci_low, comp.acc_ci_high - comp.accuracy])
    ax.errorbar(
        comp.accuracy, ys, xerr=err, fmt="none", ecolor=P.INK, elinewidth=1.2, capsize=3, zorder=3
    )
    _bar_end_labels(ax, ys, comp.acc_ci_high.to_numpy(), fmt="{:.4f}")
    for y, a in zip(ys, comp.accuracy, strict=True):
        ax.text(a + (comp.acc_ci_high.max() - comp.accuracy.min()) * 0 + 0.0, y, "", va="center")
    ax.set_yticks(ys, [P.label(m) for m in comp.model])
    ax.invert_yaxis()
    lo = np.floor((comp.acc_ci_low.min() - 0.01) * 50) / 50
    ax.set_xlim(lo, min(1.0, comp.acc_ci_high.max() + 0.015))
    ax.set_xlabel(
        f"Accuracy on the official IMDb test set (n = {int(comp.n_test.iloc[0]):,}), 95% CI"
    )
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=P.FAMILY_COLOR[f]) for f in ("classical", "transformer")
    ]
    ax.legend(
        handles,
        ["Classical (TF-IDF + LR)", "Fine-tuned transformer"],
        loc="upper center",
        bbox_to_anchor=(0.4, -0.2),
        ncol=2,
    )
    ax.set_title("Fine-tuned transformers vs the TF-IDF baseline (same test split)")
    # end labels show the point estimate next to the CI cap
    for t in list(ax.texts):
        t.remove()
    for y, a, hi in zip(ys, comp.accuracy, comp.acc_ci_high, strict=True):
        ax.text(hi + 0.001, y, f"{a:.4f}", va="center", ha="left", fontsize=9.5, color=P.INK_2)
    return P.save(fig, out / "model_comparison.png")


def fig_accuracy_vs_cost(results: Path, out: Path):
    comp = pd.read_csv(results / "comparison.csv")
    comp = comp[comp.model != "roberta_head_tail"].reset_index(drop=True)  # ablation, same cost
    if comp["throughput"].isna().any():
        raise FileNotFoundError("throughput missing: run `python -m review_sentiment benchmark`")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, col, xlabel in (
        (axes[0], "train_seconds", "Training time (seconds, log scale)"),
        (axes[1], "throughput", "Inference throughput (reviews / second, log scale)"),
    ):
        for i, r in comp.iterrows():
            c = P.FAMILY_COLOR[P.family(r.model)]
            # put the label below the point if a more accurate model sits just above it
            near = comp[(comp.accuracy > r.accuracy) & (comp.accuracy - r.accuracy < 0.008)]
            offset = (6, -12) if len(near) and i < comp.index.max() else (6, 4)
            ax.scatter(
                r[col], r.accuracy, s=64, color=c, edgecolor=P.SURFACE, linewidth=2, zorder=3
            )
            ax.annotate(
                P.label(r.model),
                (r[col], r.accuracy),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5,
                color=P.INK_2,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 3.0)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("IMDb test accuracy")
    axes[0].set_title("Accuracy vs compute")
    fig.text(
        0.01,
        -0.04,
        "Classical models: CPU (64-core server). Transformers: one NVIDIA A100 "
        "80GB, bf16. Training time excludes tokenisation.",
        fontsize=8.5,
        color=P.MUTED,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=P.FAMILY_COLOR[f], markersize=8)
        for f in ("classical", "transformer")
    ]
    axes[1].legend(handles, ["Classical", "Transformer"], loc="lower left")
    fig.tight_layout()
    return P.save(fig, out / "accuracy_vs_compute.png")


def fig_cross_domain(results: Path, out: Path):
    cross = pd.read_csv(results / "cross_domain.csv")
    models = list(dict.fromkeys(cross.model))
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    width = 0.26
    x = np.arange(len(models))
    for i, ds in enumerate(DATASETS):
        part = cross[cross.dataset == ds].set_index("model").reindex(models)
        pos = x + (i - 1) * width
        ax.bar(
            pos,
            part.accuracy,
            width=width - 0.03,
            color=P.PALETTE[i],
            label=DATASET_LABELS[ds],
            zorder=2,
        )
        err = np.vstack([part.accuracy - part.acc_ci_low, part.acc_ci_high - part.accuracy])
        ax.errorbar(
            pos,
            part.accuracy,
            yerr=err,
            fmt="none",
            ecolor=P.INK,
            elinewidth=1,
            capsize=2,
            zorder=3,
        )
    names = [P.label(m).replace(" (", "\n(").replace("-base", "\nbase").replace("-large", "\nlarge")
             for m in models]  # fmt: skip
    ax.set_xticks(x, names, fontsize=9)
    lo = np.floor((cross.acc_ci_low.min() - 0.02) * 20) / 20
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.02))
    ax.set_title("Trained on IMDb movie reviews, tested on customer reviews", pad=26)
    ax.text(
        0,
        -0.25,
        "Amazon and Yelp: 10,000-review label-balanced samples of each test split; "
        "no training on these domains. Error bars: 95% bootstrap CI.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=P.MUTED,
    )
    return P.save(fig, out / "cross_domain.png")


def fig_calibration(results: Path, out: Path, models=None):
    bins = pd.read_csv(results / "calibration" / "reliability_bins.csv")
    summ = pd.read_csv(results / "calibration" / "calibration_summary.csv")
    bins = bins[bins.dataset == "imdb_test"]
    models = models or list(dict.fromkeys(bins.model))
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n, 2.9), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models, strict=True):
        ax.plot([0.5, 1], [0.5, 1], color=P.BASELINE, lw=1, ls="-", zorder=1)
        for i, stage in enumerate(("before", "after")):
            b = bins[(bins.model == m) & (bins.stage == stage) & (bins["count"] >= 50)]
            s = summ[(summ.model == m) & (summ.dataset == "imdb_test") & (summ.stage == stage)]
            ece = float(s.ece.iloc[0])
            name = "raw" if stage == "before" else f"T = {float(s.temperature.iloc[0]):.2f}"
            ax.plot(
                b.confidence,
                b.accuracy,
                marker="o",
                ms=4,
                color=P.PALETTE[i + 1 if i else 0],
                label=f"{name} (ECE {ece:.3f})",
                zorder=2 + i,
            )
        ax.set_title(P.label(m), fontsize=10)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0.5, 1.0)
        # underconfident models (T < 1) sit above the diagonal: keep the legend off the curve
        t = float(summ[(summ.model == m) & (summ.dataset == "imdb_test")].temperature.iloc[0])
        ax.legend(loc="lower right" if t < 1 else "upper left", fontsize=7.8, handlelength=1.2)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy")
    fig.text(
        0.01,
        -0.04,
        "Points: 10 equal-width confidence bins (bins with fewer than 50 reviews omitted). "
        "T fitted on the validation split; ECE (15 bins) on the test set.",
        fontsize=8.5,
        color=P.MUTED,
    )
    fig.suptitle(
        "Reliability on the IMDb test set before and after temperature scaling",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    return P.save(fig, out / "calibration.png")


def fig_robustness(results: Path, out: Path):
    table = pd.read_csv(results / "robustness" / "robustness_summary.csv")
    tests = list(dict.fromkeys(table.test))
    models = list(dict.fromkeys(table.model))
    mat = table.pivot(index="test", columns="model", values="failure_rate").loc[tests, models]
    kinds = table.drop_duplicates("test").set_index("test").loc[tests, "type"]
    fig, ax = plt.subplots(figsize=(1.35 * len(models) + 3.2, 0.42 * len(tests) + 1.4))
    im = ax.imshow(mat.to_numpy(), cmap=P.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(len(tests)):
        for j in range(len(models)):
            v = mat.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > 0.55 else P.INK,
            )
    ax.set_xticks(
        range(len(models)), [P.label(m).replace(" (", "\n(") for m in models], fontsize=8.8
    )
    ax.set_yticks(
        range(len(tests)),
        [f"{t.replace('_', ' ')}  [{k}]" for t, k in zip(tests, kinds, strict=True)],
    )
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Failure rate")
    cb.outline.set_visible(False)
    ax.set_title("Behavioural tests: share of cases each model gets wrong (lower is better)")
    return P.save(fig, out / "robustness.png")


def fig_slices(results: Path, out: Path, models=None):
    sl = pd.read_csv(results / "error_analysis" / "slice_accuracy.csv")
    model_cols = [c for c in sl.columns if c not in ("slice", "value", "n")]
    models = models or model_cols
    fig = plt.figure(figsize=(11, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[4, 2, 2])
    panels = [
        (fig.add_subplot(gs[0, 0]), "length_bin", "Review length (words)"),
        (fig.add_subplot(gs[0, 1]), "contrast", "Contains 'but' or 'however'"),
        (fig.add_subplot(gs[0, 2]), "strength", "Rating strength"),
        (
            fig.add_subplot(gs[1, :]),
            "rating_group",
            "Star rating (1-4 = negative, 7-10 = positive)",
        ),
    ]
    lo = np.floor((sl[models].min().min() - 0.02) * 20) / 20
    for ax, col, title in panels:
        part = sl[sl.slice == col]
        if col == "rating_group":
            part = part.assign(k=part.value.str.extract(r"(\d+)").astype(int)).sort_values("k")
        x = np.arange(len(part))
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            ax.bar(
                x + (i - (len(models) - 1) / 2) * w,
                part[m],
                width=w * 0.9,
                color=P.PALETTE[i],
                label=P.label(m),
                zorder=2,
            )
        names = [{"has but/however": "yes", "no but/however": "no"}.get(v, v) for v in part.value]
        names = [v.replace(" (", "\n(") for v in names]
        ax.set_xticks(
            x, [f"{v}\nn={n:,}" for v, n in zip(names, part.n, strict=True)], fontsize=8.5
        )
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(lo, 1.0)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Accuracy")
    handles, labels = panels[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(models))
    fig.suptitle(
        "Where the models fail: accuracy by slice of the IMDb test set",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    return P.save(fig, out / "error_slices.png")


def fig_confusion(results: Path, out: Path, models):
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.2))
    for ax, m in zip(np.atleast_1d(axes), models, strict=True):
        pred = load_predictions(m, "imdb_test")
        cm = confusion_matrix(pred.label, (pred.margin > 0).astype(int))
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=P.SEQUENTIAL, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n({share[i, j]:.1%})",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color="white" if share[i, j] > 0.55 else P.INK,
                )
        ax.set_xticks([0, 1], ["negative", "positive"])
        ax.set_yticks([0, 1], ["negative", "positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(P.label(m), fontsize=10.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    return P.save(fig, out / "confusion_matrices.png")


def fig_top_terms(results: Path, out: Path, model="tfidf_lr_tuned", k=15):
    from .models import classical
    from .paths import ARTIFACTS_DIR

    path = ARTIFACTS_DIR / model / "model.joblib"
    csv = results / f"top_terms_{model}.csv"
    if path.exists():
        top = classical.top_coefficients(classical.load(path), k)
        top.to_csv(csv, index=False)
    elif csv.exists():
        top = pd.read_csv(csv)
    else:
        log.warning("skip top terms: no model or csv")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, direction, color in (
        (axes[0], "positive", P.POSITIVE),
        (axes[1], "negative", P.NEGATIVE),
    ):
        part = top[top.direction == direction].copy()
        part = part.reindex(part.weight.abs().sort_values().index)
        ax.barh(part.term, part.weight.abs(), color=color, height=0.6, zorder=2)
        ax.set_title(f"Strongest {direction} terms", fontsize=10.5)
        ax.set_xlabel("|coefficient|")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        f"{P.label(model)}: most influential n-grams",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    return P.save(fig, out / "top_terms.png")


def fig_faithfulness(results: Path, out: Path):
    curves = pd.read_csv(results / "explain" / "deletion_curves.csv")
    meta = json.loads((results / "explain" / "faithfulness.json").read_text())
    models = list(dict.fromkeys(curves.model))
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.5), sharey=True)
    method_color = {"LR contribution": P.PALETTE[1], "Integrated Gradients": P.PALETTE[0],
                    "Gradient x Input": P.PALETTE[2], "random": P.MUTED}  # fmt: skip
    aopc = {(r["model"], r["method"]): r["aopc"] for r in meta["results"]}
    for ax, m in zip(np.atleast_1d(axes), models, strict=True):
        for method, part in curves[curves.model == m].groupby("method", sort=False):
            ax.plot(
                part.fraction * 100,
                part.mean_drop,
                marker="o",
                ms=4,
                color=method_color.get(method, P.INK),
                label=f"{method} (AOPC {aopc[(m, method)]:.3f})",
                ls="--" if method == "random" else "-",
            )
            ax.fill_between(
                part.fraction * 100,
                part.mean_drop - 1.96 * part["sem"],
                part.mean_drop + 1.96 * part["sem"],
                color=method_color.get(method, P.INK),
                alpha=0.1,
                lw=0,
            )
        ax.set_title(P.label(m), fontsize=10.5)
        ax.set_xlabel("Words deleted (% of review, most important first)")
        ax.legend(loc="upper left", fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel("Drop in predicted-class probability")
    fig.suptitle(
        f"Faithfulness: deleting the words an explanation ranks highest "
        f"({meta['n_reviews']} test reviews)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    return P.save(fig, out / "explanation_faithfulness.png")


def fig_explanation_examples(results: Path, out: Path):
    from scipy.special import expit

    ex = json.loads((results / "explain" / "examples.json").read_text())
    meta = json.loads((results / "explain" / "faithfulness.json").read_text())
    rows = len(ex)
    fig, axes = plt.subplots(rows * 2, 1, figsize=(10, 1.25 * rows * 2 + 0.6))
    for r, e in enumerate(ex):
        for k, (key, name, z) in enumerate(
            (
                ("lr", P.label(meta["classical"]) + " - word contribution", e["lr_margin"]),
                ("ig", P.label(meta["transformer"]) + " - Integrated Gradients", e["tf_margin"]),
            )
        ):
            ax = axes[2 * r + k]
            s = np.asarray(e[key])
            P.word_heatmap(
                ax, e["words"], s, max_abs=np.abs(s).max() or 1, fontsize=9.2, line_height=0.3
            )
            truth = "positive" if e["label"] == 1 else "negative"
            ax.set_title(
                f"Review {r + 1} (true: {truth})  |  {name}:  P(positive) = {expit(z):.2f}",
                fontsize=9.5,
                loc="left",
                fontweight="normal",
                color=P.INK_2,
            )
    fig.suptitle(
        "Token attributions (blue: pushes towards positive, red: towards negative)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    return P.save(fig, out / "explanation_examples.png")


def make_all(results: Path, out: Path) -> list[Path]:
    P.apply_style()
    results, out = Path(results), Path(out)
    made = []
    steps = [
        ("leakage", lambda: fig_leakage(results, out)),
        ("comparison", lambda: fig_model_comparison(results, out)),
        ("cost", lambda: fig_accuracy_vs_cost(results, out)),
        ("cross-domain", lambda: fig_cross_domain(results, out)),
        ("calibration", lambda: fig_calibration(
            results, out, ["tfidf_lr_tuned", "distilbert", "roberta", "deberta", "deberta_large"])),
        ("robustness", lambda: fig_robustness(results, out)),
        ("slices", lambda: fig_slices(
            results, out, ["tfidf_lr_tuned", "distilbert", "deberta", "deberta_large"])),
        ("confusion", lambda: fig_confusion(results, out, ["tfidf_lr_tuned", "deberta_large"])),
        ("top terms", lambda: fig_top_terms(results, out)),
        ("faithfulness", lambda: fig_faithfulness(results, out)),
        ("explanation examples", lambda: fig_explanation_examples(results, out)),
    ]  # fmt: skip
    for name, fn in steps:
        try:
            path = fn()
            if path:
                made.append(path)
                log.info("figure: %s", path)
        except (FileNotFoundError, KeyError, IndexError) as e:
            # a model or analysis has not been run yet; render the remaining figures
            log.warning("skip %s figure (missing input: %r)", name, e)
    return made
