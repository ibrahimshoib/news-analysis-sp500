# Financial news and S&P 500 returns

The project tests whether headline sentiment adds information to forecasts of the next S&P 500 session. The report documents a calendar-join leakage bug, rebuilds the analysis with one target per trading session, and evaluates direction and absolute-return forecasts using a conservative news cutoff.

## Reproduce

Run these commands from the project root with Python 3.11 or newer.

```text
python -m pip install -r requirements.txt
python -B scripts/news_research.py
python -B scripts/generate_pdf_report.py
python -B -m unittest discover -s tests
```

- `checkpoint_research_report.pdf` and `checkpoint_research_report.md` are the submission report.
- `datasets/daily_sentiment_market.csv` is the bundled input. Its SHA-256 is recorded in the results.
- `datasets/holdout_predictions.csv` records each test target, cutoff, eligible news dates, features, and forecasts.
- `datasets/research_results.json` contains the computed audit and forecast metrics.
- `visualizations/` contains the two figures generated from those metrics.

## Input provenance

The daily aggregate and scored-headline sample were supplied with the project. The original `scripts/cleaning.py` and `scripts/sentiment_pipeline.py` document the upstream preparation and custom VADER scoring. Rebuilding the aggregate requires the original raw news and S&P 500 files plus NLTK and its VADER lexicon. The full raw files are outside this submission. The reported analysis reproduces directly from the bundled aggregate.
