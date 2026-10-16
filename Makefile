# Reproduce every number and figure in the README.
#
#   make setup      create the virtual environment (Python 3.11) and install pinned dependencies
#   make all        data -> classical -> transformers -> analyses -> figures and tables (one GPU)
#   make cpu        everything that does not need a GPU (classical models, leakage study)
#   make figures    re-render figures from the committed results/ files only
#   make test lint  quality checks (same as CI)
#
# Override the interpreter or GPU, e.g.:  make all PYTHON=python3 GPU=1

PYTHON ?= .venv/bin/python
GPU ?= 0
RUN = $(PYTHON) -m review_sentiment
GPU_RUN = CUDA_VISIBLE_DEVICES=$(GPU) TOKENIZERS_PARALLELISM=false $(RUN)
TRANSFORMERS = distilbert roberta roberta_head_tail deberta deberta_large
SEEDS = 43 44
ALL_MODELS = tfidf_lr_original tfidf_lr_tuned $(TRANSFORMERS)

.PHONY: setup data classical leakage transformers seeds domains compare calibration errors \
        explain robustness benchmark figures tables cpu all test lint format app clean

setup:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

data:
	$(RUN) download --cross-domain

classical:
	$(RUN) train-classical --config configs/tfidf_lr_original.yaml
	$(RUN) train-classical --config configs/tfidf_lr_tuned.yaml --n-jobs 12

leakage:
	$(RUN) leakage --n-jobs 11

transformers:
	for m in $(TRANSFORMERS); do $(GPU_RUN) train-transformer --config configs/$$m.yaml; done

seeds:
	for s in $(SEEDS); do for m in distilbert roberta deberta; do \
		$(GPU_RUN) train-transformer --config configs/$$m.yaml --model-seed $$s; done; done

domains:
	$(GPU_RUN) predict-domains --models $(ALL_MODELS)

compare:
	$(RUN) compare

calibration:
	$(RUN) calibration

errors:
	$(RUN) errors

explain:
	$(GPU_RUN) explain --model deberta --classical tfidf_lr_tuned

robustness:
	$(GPU_RUN) robustness --models $(ALL_MODELS)

benchmark:
	$(GPU_RUN) benchmark --models $(ALL_MODELS)

figures:
	$(RUN) figures

tables:
	$(RUN) tables

cpu: data classical leakage

all: data classical leakage transformers seeds domains benchmark compare calibration errors \
     explain robustness figures tables

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check . --fix

app:
	$(PYTHON) app/app.py

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
