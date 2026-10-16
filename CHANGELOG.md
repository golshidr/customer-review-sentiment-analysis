# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-16

First tagged release. The exploratory notebook is now an installable, tested package with a full
evaluation of classical and transformer sentiment models.

### Added

- The `review_sentiment` package (`src/review_sentiment/`) with a command-line interface
  (`python -m review_sentiment <command>`), one YAML config per model and a Makefile target for
  every stage of the pipeline.
- A leakage-free evaluation protocol: the official movie-disjoint IMDb split, exact duplicates
  removed and a movie-grouped validation set.
- A TF-IDF + logistic regression baseline tuned on the validation set (36 configurations) next to
  the notebook's original model. Tuning adds 2.2 points; the biggest single change is keeping stop
  words, because scikit-learn's stop-word list deletes "not", "no", "never", "but" and "however".
- Fine-tuned DistilBERT, RoBERTa-base (with a head+tail truncation ablation), DeBERTa-v3-base and
  DeBERTa-v3-large, with three training seeds for DistilBERT, RoBERTa-base and DeBERTa-v3-base.
- Bootstrap confidence intervals, paired bootstrap differences and exact McNemar tests.
- Cross-domain evaluation of the IMDb-trained models on Amazon and Yelp customer reviews.
- Token-level explanations (exact word contributions for the linear model, Integrated Gradients
  and gradient x input for transformers) with a deletion test of their faithfulness.
- CheckList-style behavioural tests: negation, contrast, typos, irrelevant sentences, intensifiers
  and appended praise or complaints.
- Calibration analysis: reliability diagrams, expected calibration error and temperature scaling.
- Slice-based error analysis and a review of the most confident errors.
- An inference throughput benchmark.
- A Gradio demo that highlights the words behind each prediction (`app/app.py`).
- A model card (`MODEL_CARD.md`), a detailed results document (`docs/RESULTS.md`), the per-review
  predictions of every model (`results/predictions/`) and all figures (`figures/`).
- 45 unit tests on synthetic data, ruff linting and formatting, and GitHub Actions CI on Python
  3.11, 3.12 and 3.13.

### Changed

- The original notebook is reordered so that it runs top to bottom; its figures are kept in
  `figures/original/`.
- `requirements.txt` pins the exact versions used to produce the results. The CPU-only set for
  development and CI is `requirements-dev.txt` (previously `requirements-ci.txt`).
- Python 3.11 or newer is required, as the pinned numpy, pandas, scipy, scikit-learn and
  matplotlib versions do not support Python 3.10.

### Fixed

- Train/test leakage in the original evaluation. The notebook pooled the official IMDb train and
  test sets and re-split them at random 80/20, which put a review of the same movie in training for
  about 96% of test reviews and left 126-149 exact duplicates of training reviews in the test set.
  On movie-disjoint, de-duplicated splits the same model scores 88.3%, not 89.6%. All reported
  results now use the official split.

[1.0.0]: https://github.com/golshidr/customer-review-sentiment-analysis/releases/tag/v1.0.0
