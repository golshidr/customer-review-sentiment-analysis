# Model card: review sentiment classifiers

This card covers the models trained in this repository. Two are recommended:

- **DeBERTa-v3-large, fine-tuned** (`configs/deberta_large.yaml`): the most accurate model.
- **TF-IDF + logistic regression, tuned** (`configs/tfidf_lr_tuned.yaml`): a CPU-only baseline that
  is fast, small and fully interpretable.

DistilBERT, RoBERTa-base and DeBERTa-v3-base were trained with the same procedure and are reported
for comparison. All numbers come from `results/` and are explained in
[docs/RESULTS.md](docs/RESULTS.md).

## Model details

| | DeBERTa-v3-large (fine-tuned) | TF-IDF + LR (tuned) |
|---|---|---|
| Base model | `microsoft/deberta-v3-large` (MIT license) | none (scikit-learn) |
| Parameters | 435M | 50,000 features x 1 coefficient + intercept |
| Input | up to 512 tokens (longer reviews keep the first 510) | full text, unigrams and bigrams |
| Output | log-odds of "positive"; P(positive) = sigmoid(log-odds) | same |
| Training | 2 epochs, AdamW, learning rate 1e-5, batch size 16, 10% linear warm-up, bf16; checkpoint with the best validation accuracy kept | grid search over 36 settings on the validation split, refit on the training split |
| Hardware | one NVIDIA A100 80GB | CPU |
| Weights | not distributed; retrain with `python -m review_sentiment train-transformer --config configs/deberta_large.yaml` | not distributed; retrain in about 12 s |

The other base models are `distilbert-base-uncased` (Apache 2.0), `roberta-base` (MIT) and
`microsoft/deberta-v3-base` (MIT), according to their Hugging Face model cards.

## Intended use

- Classifying English review text as positive or negative, for example to monitor the overall tone
  of customer feedback or to triage reviews for a human to read.
- Teaching and research on evaluation practice: leakage-free splits, significance testing,
  calibration, explanation faithfulness and behavioural testing.

## Out-of-scope use

- Decisions about individual people (employees, customers, reviewers) based on predicted sentiment.
- Neutral or mixed feedback: the models have no neutral class and will force such text into
  positive or negative.
- Non-English text, star-rating prediction, aspect-level sentiment ("good food, slow service"),
  and detecting sarcasm.
- Any use where a wrong label has serious consequences without human review.

## Training data

The Large Movie Review Dataset v1.0 (Maas et al., 2011), official split: 22,416 training and 2,486
validation reviews after removing 98 exact duplicates; the validation reviews come from movies that
do not appear in the training set. Labels: 1-4 stars negative, 7-10 stars positive; 5-6 star
reviews are not in the dataset. All reviews are English movie reviews from IMDb.

## Evaluation data

- IMDb official test split: 24,877 reviews of movies that do not appear in training (123 test
  reviews that duplicate a training review were removed).
- Out of domain, without any training: label-balanced samples of 10,000 reviews from the test
  splits of Amazon Review Polarity and Yelp Review Polarity (Zhang et al., 2015).

## Results

| Metric | DeBERTa-v3-large | TF-IDF + LR (tuned) |
|---|---|---|
| IMDb test accuracy [95% bootstrap CI] | 96.73% [96.52, 96.95] | 90.28% [89.93, 90.63] |
| IMDb test macro-F1 / ROC-AUC | 0.967 / 0.994 | 0.903 / 0.965 |
| Amazon accuracy (no training on Amazon) | 95.81% | 84.53% |
| Yelp accuracy (no training on Yelp) | 97.20% | 85.49% |
| Accuracy on mild reviews (4 or 7 stars) | 91.00% | 79.83% |
| ECE on IMDb test, raw -> temperature-scaled | 0.026 -> 0.022 | 0.057 -> 0.012 |

The difference between the two models is 6.46 points [6.09, 6.83] (paired bootstrap; exact
McNemar p = 2.6e-268).

## Known limitations and failure modes

- **Mild and mixed reviews.** Accuracy drops to 89% and 93% on 4- and 7-star reviews
  (DeBERTa-v3-large) and to 79% and 80% (TF-IDF); about half of DeBERTa-v3-large's errors are on these reviews.
- **Negation.** The TF-IDF model fails 89% of templated negated-praise sentences ("The service was
  not great.") and all negated-criticism sentences ("The product was not terrible."). DeBERTa-v3-large
  fails 15% and 34% of them.
- **Sarcasm and label noise.** Many of the most confident errors are sarcastic praise or reviews
  whose text contradicts their star rating.
- **Long reviews.** Only the first 512 tokens are read; about 13% of IMDb test reviews are longer.
- **Calibration does not transfer.** Temperature scaling fitted on IMDb validation data improves
  ECE on IMDb and Amazon but doubles it for DeBERTa-v3-large on Yelp (0.014 -> 0.029).
- **Explanations.** Word attributions from the TF-IDF model are exact. Integrated Gradients for
  transformers is only partly faithful (deleting the top-ranked half of a review lowers the
  predicted-class probability by 0.28 on average) and should not be read as a full account of a
  decision.

## Ethical considerations

- IMDb reviewers are not representative of all customers, and the language of movie reviews
  differs from that of support tickets or survey responses.
- Bias with respect to names, dialects or demographic groups mentioned in reviews was not
  evaluated. Before deployment, test the model on reviews that mention different groups and
  compare predictions.
- Predicted sentiment is an aggregate signal. It should not be used to judge or rank the people
  who wrote the reviews.

## How to reproduce

```bash
python -m review_sentiment download --cross-domain
python -m review_sentiment train-classical --config configs/tfidf_lr_tuned.yaml
python -m review_sentiment train-transformer --config configs/deberta_large.yaml
python -m review_sentiment compare
```

## Citation of data

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. (2011). Learning Word
Vectors for Sentiment Analysis. ACL-HLT.

Zhang, X., Zhao, J., and LeCun, Y. (2015). Character-level Convolutional Networks for Text
Classification. NeurIPS.
