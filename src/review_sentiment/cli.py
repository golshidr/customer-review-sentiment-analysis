"""Command-line interface: ``python -m review_sentiment <command> [options]``."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from . import paths

log = logging.getLogger("review_sentiment")


# --------------------------------------------------------------------------------------- helpers
def _splits(args):
    from .data import load_imdb, make_splits

    train_df, test_df = load_imdb(args.data_dir)
    return make_splits(train_df, test_df, protocol=args.protocol, seed=args.seed)


# -------------------------------------------------------------------------------------- commands
def cmd_download(args) -> None:
    from .data import prepare_cross_domain, prepare_imdb

    prepare_imdb(args.raw_dir, args.data_dir)
    if args.cross_domain:
        prepare_cross_domain(args.cross_domain_size, args.seed, args.data_dir)


def cmd_train_classical(args) -> None:
    from .config import load_config
    from .evaluation import classification_metrics
    from .io import save_predictions, write_json
    from .models import classical

    cfg_dict = load_config(args.config)
    name = cfg_dict["name"]
    splits = _splits(args)
    base = classical.TfidfLRConfig.from_dict(cfg_dict.get("params", {}))
    record: dict = {"model": name, "split": splits.summary()}
    if cfg_dict.get("grid"):
        best, table = classical.grid_search(
            base,
            cfg_dict["grid"],
            splits.train["text"],
            splits.train["label"],
            splits.val["text"],
            splits.val["label"],
            n_jobs=args.n_jobs,
        )
        table.to_csv(paths.RESULTS_DIR / f"grid_search_{name}.csv", index=False)
        record["grid_size"] = len(table)
        base = best
    t0 = time.perf_counter()
    pipe = classical.fit(base, splits.train["text"], splits.train["label"])
    record["train_seconds"] = round(time.perf_counter() - t0, 1)
    record["params"] = base.to_dict()
    classical.save(pipe, paths.ARTIFACTS_DIR / name / "model.joblib")
    for part in ("val", "test"):
        df = getattr(splits, part)
        t0 = time.perf_counter()
        margin = classical.predict_margin(pipe, df["text"])
        elapsed = time.perf_counter() - t0
        save_predictions(name, f"imdb_{part}", df["review_id"], df["label"], margin)
        record[part] = classification_metrics(df["label"], margin)
        if part == "test":
            record["test_examples_per_second_cpu"] = round(len(df) / elapsed, 1)
    write_json(record, paths.RESULTS_DIR / "metrics" / f"{name}.json")
    log.info("%s test accuracy = %.4f", name, record["test"]["accuracy"])


def cmd_train_transformer(args) -> None:
    import torch

    from .config import load_config
    from .evaluation import classification_metrics
    from .io import save_predictions, write_json
    from .models import transformer as tr

    cfg = tr.TransformerConfig.from_dict(load_config(args.config))
    if args.model_seed is not None:  # extra runs for seed variance; the data split stays fixed
        cfg.seed = args.model_seed
        cfg.name = f"{cfg.name}_s{args.model_seed}"
    splits = _splits(args)
    out_dir = paths.ARTIFACTS_DIR / cfg.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    record = {"model": cfg.name, "split": splits.summary(), "config": cfg.to_dict()}
    record["gpu"] = torch.cuda.get_device_name(0) if device == "cuda" else None
    record.update(tr.train(cfg, splits.train, splits.val, out_dir, device=device))
    model, tok = tr.load(out_dir, device)
    for part in ("val", "test"):
        df = getattr(splits, part)
        enc = tr.encode_texts(tok, df["text"], cfg.max_length, cfg.truncation, cfg.head_tokens)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = tr.predict_logits(model, enc, tok.pad_token_id, cfg.eval_batch_size, cfg.bf16)
        elapsed = time.perf_counter() - t0
        margin = tr.margin_from_logits(logits)
        save_predictions(cfg.name, f"imdb_{part}", df["review_id"], df["label"], margin)
        record[part] = classification_metrics(df["label"], margin)
        if part == "test":
            record["test_examples_per_second_gpu"] = round(len(df) / elapsed, 1)
    write_json(record, paths.RESULTS_DIR / "metrics" / f"{cfg.name}.json")
    log.info("%s test accuracy = %.4f", cfg.name, record["test"]["accuracy"])


def cmd_predict_domains(args) -> None:
    """Score IMDb-trained models on customer-review corpora (Amazon, Yelp)."""
    from .data import CROSS_DOMAIN, load_cross_domain
    from .io import save_predictions
    from .predictors import load_predictor

    for name in args.models:
        predictor = load_predictor(name)
        for dataset in CROSS_DOMAIN:
            df = load_cross_domain(dataset, args.data_dir)
            margin = predictor.margin(df["text"].tolist())
            save_predictions(name, dataset, df["review_id"], df["label"], margin)
            acc = float(((margin > 0).astype(int) == df["label"].to_numpy()).mean())
            log.info("%s on %s: accuracy %.4f", name, dataset, acc)


def _models(args) -> list[str]:
    from .analysis import discover_models

    return args.models or discover_models()


def cmd_compare(args) -> None:
    from . import analysis as A

    models = _models(args)
    out = paths.RESULTS_DIR
    comp = A.comparison_table(models)
    comp.to_csv(out / "comparison.csv", index=False)
    sig = pd_concat(
        [A.significance_table(models, "tfidf_lr_tuned")]
        + ([A.significance_table(models, args.best)] if args.best in models else [])
    )
    sig.to_csv(out / "significance.csv", index=False)
    cross = A.cross_domain_table(models)
    cross.to_csv(out / "cross_domain.csv", index=False)
    log.info("\n%s", comp[["model", "accuracy", "acc_ci_low", "acc_ci_high", "seed_acc_mean",
                           "seed_acc_std", "ece"]].to_string(index=False))  # fmt: skip
    log.info("\n%s", sig.to_string(index=False))
    if not cross.empty:
        log.info("\n%s", cross.pivot(index="model", columns="dataset", values="accuracy"))


def pd_concat(frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def cmd_calibration(args) -> None:
    from .analysis import calibration_tables

    summary, bins = calibration_tables(_models(args))
    out = paths.RESULTS_DIR / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "calibration_summary.csv", index=False)
    bins.to_csv(out / "reliability_bins.csv", index=False)
    log.info("\n%s", summary[summary.dataset == "imdb_test"].to_string(index=False))


def cmd_errors(args) -> None:
    import pandas as pd

    from .error_analysis import add_slices, hardest_errors, slice_accuracy
    from .io import load_predictions

    splits = _splits(args)
    test = add_slices(splits.test)
    models = _models(args)
    correct = {}
    for m in models:
        pred = load_predictions(m, "imdb_test").set_index("review_id")
        test[f"margin_{m}"] = pred.loc[test["review_id"], "margin"].to_numpy()
        test[f"correct_{m}"] = (test[f"margin_{m}"] > 0).astype(int) == test["label"]
        correct[m] = f"correct_{m}"
    out = paths.RESULTS_DIR / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    slices = pd.concat(
        [slice_accuracy(test, correct, c) for c in
         ("length_bin", "negation", "contrast", "rating_group", "strength")],
        ignore_index=True,
    )  # fmt: skip
    slices.to_csv(out / "slice_accuracy.csv", index=False)
    best = args.best if args.best in models else models[-1]
    hard = hardest_errors(test, f"margin_{best}", k=args.k)
    hard = hard.assign(text=hard["text"].str.slice(0, 400))
    hard[["review_id", "label", "rating", "pred", "confidence", "n_words", "text"]].to_csv(
        out / f"hardest_errors_{best}.csv", index=False
    )
    if "tfidf_lr_tuned" in correct:
        a, b = test[correct["tfidf_lr_tuned"]], test[correct[best]]
        agreement = pd.crosstab(a.rename("tfidf_lr_tuned_correct"), b.rename(f"{best}_correct"))
        agreement.to_csv(out / f"agreement_tfidf_lr_tuned_vs_{best}.csv")
        log.info("\n%s", agreement)
    log.info("\n%s", slices.to_string(index=False))
    if {"roberta", "roberta_head_tail"} <= set(models):
        _truncation_ablation(test)


def _truncation_ablation(test) -> None:
    """Head-only vs head+tail RoBERTa on the reviews that exceed 512 tokens."""
    import numpy as np
    from transformers import AutoTokenizer

    from .error_analysis import truncation_ablation

    cfg = json.loads((paths.RESULTS_DIR / "metrics" / "roberta.json").read_text())["config"]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    n_tokens = np.array([len(ids) for ids in tok(test["text"].tolist())["input_ids"]])
    table = truncation_ablation(
        test["label"].to_numpy(),
        test["margin_roberta"].to_numpy(),
        test["margin_roberta_head_tail"].to_numpy(),
        n_tokens > cfg["max_length"],
    )
    table.to_csv(paths.RESULTS_DIR / "error_analysis" / "truncation_ablation.csv", index=False)
    log.info("\n%s", table.to_string(index=False))


def cmd_explain(args) -> None:
    import numpy as np
    from scipy.stats import spearmanr

    from . import explain as E
    from .io import write_json
    from .predictors import load_predictor

    splits = _splits(args)
    test = splits.test.copy()
    test["n_words"] = test["text"].map(lambda t: len(E.split_words(t)[1]))
    sample = test[test["n_words"] <= args.max_words].sample(n=args.n, random_state=0)
    texts = sample["text"].tolist()
    out = paths.RESULTS_DIR / "explain"
    out.mkdir(parents=True, exist_ok=True)

    lr = load_predictor(args.classical)
    tf = load_predictor(args.model)
    t0 = time.perf_counter()
    lr_attr = [E.lr_attributions(lr.pipeline, t)[1] for t in texts]
    ig, deltas = [], []
    for t in texts:
        _, s, d = E.integrated_gradients(tf, t, n_steps=args.steps)
        ig.append(s)
        deltas.append(d)
    t_ig = time.perf_counter() - t0
    gxi = [E.gradient_x_input(tf, t)[1] for t in texts]
    rng = np.random.default_rng(0)
    rand = [rng.normal(size=len(a)) for a in lr_attr]

    record = {"n_reviews": len(texts), "max_words": args.max_words, "ig_steps": args.steps,
              "transformer": args.model, "classical": args.classical,
              "ig_seconds_total": round(t_ig, 1),
              "ig_abs_convergence_delta_median": float(np.median(np.abs(deltas))),
        # |delta| relative to F(x) - F(baseline) = sum(attributions) - delta
        "ig_relative_convergence_delta_median": float(
            np.median([abs(d) / max(abs(a.sum() - d), 1e-6) for a, d in zip(ig, deltas)])  # noqa: B905
        ),
              "results": []}  # fmt: skip
    curves_rows = []
    for model_name, predictor, methods in (
        (args.classical, lr, {"LR contribution": lr_attr, "random": rand}),
        (args.model, tf, {"Integrated Gradients": ig, "Gradient x Input": gxi, "random": rand}),
    ):
        for method, attrs in methods.items():
            curves = E.deletion_curves(predictor, texts, attrs)
            comp = E.comprehensiveness(curves)
            record["results"].append(
                {"model": model_name, "method": method, "aopc": comp["aopc_top"]}
            )
            drop = curves["original"][:, None] - curves["top"]
            for fi, frac in enumerate(curves["fractions"]):
                curves_rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "fraction": frac,
                        "mean_drop": float(drop[:, fi].mean()),
                        "sem": float(drop[:, fi].std(ddof=1) / np.sqrt(len(texts))),
                    }
                )
    # agreement between the two models' explanations (same words, same reviews)
    rhos = [spearmanr(a, b).statistic for a, b in zip(lr_attr, ig, strict=True) if len(a) > 2]
    record["spearman_lr_vs_ig_median"] = float(np.nanmedian(rhos))
    top_overlap = []
    for a, b in zip(lr_attr, ig, strict=True):
        k = min(5, len(a))
        top_overlap.append(
            len(set(np.argsort(-np.abs(a))[:k]) & set(np.argsort(-np.abs(b))[:k])) / k
        )
    record["top5_overlap_lr_vs_ig_mean"] = float(np.mean(top_overlap))
    write_json(record, out / "faithfulness.json")
    import pandas as pd

    pd.DataFrame(curves_rows).to_csv(out / "deletion_curves.csv", index=False)

    # a few short reviews for the qualitative figure
    examples = []
    pool = test[(test["n_words"] >= 18) & (test["n_words"] <= 45)].sample(n=200, random_state=1)
    picks = [pool[pool.label == 1].iloc[0], pool[pool.label == 0].iloc[0]]
    neg = pool[pool["text"].str.contains(r"\b(?:not|n't|never)\b", regex=True)]
    if len(neg):
        picks.append(neg.iloc[0])
    for row in picks:
        words, s_lr = E.lr_attributions(lr.pipeline, row["text"])
        _, s_ig, _ = E.integrated_gradients(tf, row["text"], n_steps=args.steps)
        examples.append({"review_id": row["review_id"], "label": int(row["label"]),
                         "words": words, "lr": s_lr.tolist(), "ig": s_ig.tolist(),
                         "lr_margin": float(lr.margin([row["text"]])[0]),
                         "tf_margin": float(tf.margin([row["text"]])[0])})  # fmt: skip
    write_json(examples, out / "examples.json")
    log.info("%s", json.dumps({k: v for k, v in record.items() if k != "results"}, indent=1))
    for r in record["results"]:
        log.info("%s / %s: AOPC %.4f", r["model"], r["method"], r["aopc"])


def cmd_robustness(args) -> None:
    import pandas as pd

    from .io import write_json
    from .predictors import load_predictor
    from .robustness import run_suite

    splits = _splits(args)
    texts = splits.test.sample(n=args.n, random_state=0)["text"].tolist()
    rows, details = [], {}
    for m in _models(args):
        res = run_suite(load_predictor(m), texts, seed=0)
        details[m] = [r.to_dict() for r in res]
        rows += [{"model": m, **{k: v for k, v in r.to_dict().items() if k != "examples"}}
                 for r in res]  # fmt: skip
        log.info("%s done", m)
    out = paths.RESULTS_DIR / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "robustness_summary.csv", index=False)
    write_json(details, out / "robustness_details.json")
    log.info("\n%s", table.pivot(index="test", columns="model", values="failure_rate"))


def cmd_benchmark(args) -> None:
    """End-to-end inference throughput (including tokenisation), one model at a time."""
    import pandas as pd
    import torch

    from .predictors import load_predictor

    splits = _splits(args)
    texts = splits.test.sample(n=args.n, random_state=0)["text"].tolist()
    rows = []
    for m in _models(args):
        predictor = load_predictor(m)
        predictor.margin(texts[:256])  # warm-up
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.margin(texts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        hw = "CPU" if predictor.kind == "classical" else torch.cuda.get_device_name(0)
        rows.append(
            {
                "model": m,
                "n": len(texts),
                "seconds": round(elapsed, 2),
                "reviews_per_second": round(len(texts) / elapsed, 1),
                "hardware": hw,
            }
        )
        log.info("%s: %.1f reviews/s", m, len(texts) / elapsed)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(paths.RESULTS_DIR / "benchmark.csv", index=False)


def cmd_figures(args) -> None:
    from . import figures

    figures.make_all(paths.RESULTS_DIR, paths.FIGURES_DIR)


def cmd_tables(args) -> None:
    from .tables import write_all

    log.info("wrote %s", write_all(paths.RESULTS_DIR))


def cmd_leakage(args) -> None:
    import os

    from .data import load_imdb
    from .io import write_json
    from .leakage import run_leakage_study

    os.environ.setdefault("OMP_NUM_THREADS", "8")
    train_df, test_df = load_imdb(args.data_dir)
    runs, summary = run_leakage_study(train_df, test_df, n_jobs=args.n_jobs)
    out = paths.RESULTS_DIR / "leakage"
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "leakage_runs.csv", index=False)
    summary.to_csv(out / "leakage_summary.csv", index=False)
    write_json(summary.to_dict(orient="records"), out / "leakage_summary.json")
    log.info("\n%s", summary.to_string(index=False))


# ------------------------------------------------------------------------------------ main/parse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="review_sentiment", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=paths.PROCESSED_DIR)
    common.add_argument("--raw-dir", type=Path, default=paths.RAW_DIR)
    common.add_argument("--protocol", default="official")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--n-jobs", type=int, default=-1)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", parents=[common], help="download and prepare datasets")
    p.add_argument("--cross-domain", action="store_true", help="also Amazon/Yelp test samples")
    p.add_argument("--cross-domain-size", type=int, default=10000)
    p = sub.add_parser("train-classical", parents=[common], help="train TF-IDF + LR")
    p.add_argument("--config", type=Path, required=True)
    p = sub.add_parser("train-transformer", parents=[common], help="fine-tune a transformer")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-seed", type=int, default=None, help="override the training seed")
    p = sub.add_parser("predict-domains", parents=[common], help="score Amazon/Yelp samples")
    p.add_argument("--models", nargs="+", required=True)
    sub.add_parser("leakage", parents=[common], help="random vs movie-disjoint split study")

    analysis = argparse.ArgumentParser(add_help=False, parents=[common])
    analysis.add_argument("--models", nargs="+", default=None, help="default: all trained models")
    analysis.add_argument("--best", default="deberta", help="model highlighted in comparisons")
    sub.add_parser("compare", parents=[analysis], help="accuracy table, CIs, significance")
    sub.add_parser("calibration", parents=[analysis], help="ECE and temperature scaling")
    p = sub.add_parser("errors", parents=[analysis], help="slice-based error analysis")
    p.add_argument("--k", type=int, default=15, help="number of hardest errors to keep")
    p = sub.add_parser("explain", parents=[analysis], help="attributions + faithfulness test")
    p.add_argument("--model", default="deberta")
    p.add_argument("--classical", default="tfidf_lr_tuned")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--max-words", type=int, default=200)
    p.add_argument("--steps", type=int, default=100)
    p = sub.add_parser("robustness", parents=[analysis], help="CheckList-style tests")
    p.add_argument("--n", type=int, default=1000)
    p = sub.add_parser("benchmark", parents=[analysis], help="inference throughput per model")
    p.add_argument("--n", type=int, default=5000)
    sub.add_parser("figures", parents=[analysis], help="render all figures")
    sub.add_parser("tables", parents=[analysis], help="write results/tables.md")
    return parser


COMMANDS = {
    "download": cmd_download,
    "train-classical": cmd_train_classical,
    "train-transformer": cmd_train_transformer,
    "predict-domains": cmd_predict_domains,
    "leakage": cmd_leakage,
    "compare": cmd_compare,
    "calibration": cmd_calibration,
    "errors": cmd_errors,
    "explain": cmd_explain,
    "robustness": cmd_robustness,
    "benchmark": cmd_benchmark,
    "figures": cmd_figures,
    "tables": cmd_tables,
}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
