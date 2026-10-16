"""Download the IMDb archive and build the processed parquet tables.

Usage:
    python scripts/download_data.py            # IMDb only
    python scripts/download_data.py --cross-domain   # also Amazon / Yelp polarity test samples

Equivalent to ``python -m review_sentiment download``.
"""

from review_sentiment.cli import main

if __name__ == "__main__":
    import sys

    main(["download", *sys.argv[1:]])
