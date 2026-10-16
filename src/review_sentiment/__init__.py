"""Sentiment analysis of customer and movie reviews.

The package turns the original exploratory notebook into reusable, tested components:

* :mod:`review_sentiment.data` - download and split the IMDb dataset (plus cross-domain test sets)
* :mod:`review_sentiment.preprocessing` - text cleaning helpers
* :mod:`review_sentiment.models` - TF-IDF + logistic regression and fine-tuned transformers
* :mod:`review_sentiment.evaluation` - metrics, bootstrap confidence intervals, significance tests
* :mod:`review_sentiment.calibration` - reliability analysis and temperature scaling
* :mod:`review_sentiment.explain` - token attributions and faithfulness (deletion) tests
* :mod:`review_sentiment.robustness` - CheckList-style behavioural tests
* :mod:`review_sentiment.error_analysis` - slice-based error analysis
"""

__version__ = "1.0.0"
