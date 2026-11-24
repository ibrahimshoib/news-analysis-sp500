"""Reproduce the session-alignment audit and news forecasting experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
SOURCE = DATA / "daily_sentiment_market.csv"
START_TEST = "2022-01-01"
END_TEST = "2023-12-31"
BLOCK_LENGTH = 10
RESAMPLES = 2000
SEED = 42

MARKET_FEATURES = ["previous_return", "mean_return_5", "mean_abs_return_5", "mean_abs_return_20"]
NEWS_FEATURES = ["available_sentiment", "available_dispersion", "available_log_article_count"]
PROBABILITY_COLUMNS = ["base_up_probability", "market_up_probability", "market_news_up_probability"]
VOLATILITY_COLUMNS = ["base_abs_return", "market_abs_return", "market_news_abs_return"]


def load_daily(path=SOURCE):
    daily = pd.read_csv(path, parse_dates=["published_date", "market_date"])
    required = ["published_date", "market_date", "market_return", "article_count",
                "sentiment_mean", "sentiment_std"]
    daily = daily[required].sort_values("published_date", ignore_index=True)
    if daily.isna().any().any() or not np.isfinite(daily[required[2:]].to_numpy()).all():
        raise ValueError("The daily source has missing or non-finite values.")
    if (daily["published_date"].duplicated().any()
        or not daily["market_date"].is_monotonic_increasing
        or not daily["published_date"].lt(daily["market_date"]).all()
        or daily["market_date"].dt.dayofweek.ge(5).any()):
        raise ValueError("The daily source has invalid publication-to-market date mappings.")
    observed_sessions = pd.Index(daily["market_date"].drop_duplicates())
    next_session = observed_sessions.take(observed_sessions.searchsorted(daily["published_date"], side="right"))
    if not next_session.equals(pd.Index(daily["market_date"])):
        raise ValueError("A publication row skips its first subsequent recorded session.")
    if (daily["article_count"].le(0).any()
        or daily["article_count"].mod(1).ne(0).any()
        or daily["sentiment_std"].lt(0).any()
        or daily.groupby("market_date")["market_return"].nunique().ne(1).any()):
        raise ValueError("The daily source has invalid aggregates or conflicting target returns.")
    return daily


def pool_sessions(daily):
    weighted = daily.assign(
        sentiment_sum=daily["article_count"] * daily["sentiment_mean"],
        sentiment_squares=daily["article_count"] * (daily["sentiment_std"] ** 2 + daily["sentiment_mean"] ** 2),
    )
    sessions = weighted.groupby("market_date").agg(
        articles=("article_count", "sum"),
        total_score=("sentiment_sum", "sum"),
        total_squared_score=("sentiment_squares", "sum"),
        first_news_date=("published_date", "min"),
        last_news_date=("published_date", "max"),
        market_return=("market_return", "first"),
    )
    sessions["sentiment"] = sessions["total_score"] / sessions["articles"]
    variance = sessions["total_squared_score"] / sessions["articles"] - sessions["sentiment"] ** 2
    sessions["dispersion"] = np.sqrt(variance.clip(lower=0))
    return sessions


def make_features(sessions):
    frame = sessions.copy()
    frame["cutoff_session"] = frame.index.to_series().shift()
    frame["available_first_news_date"] = frame["first_news_date"].shift()
    frame["available_last_news_date"] = frame["last_news_date"].shift()
    frame["available_sentiment"] = frame["sentiment"].shift()
    frame["available_dispersion"] = frame["dispersion"].shift()
    frame["available_log_article_count"] = np.log1p(frame["articles"].shift())
    frame["previous_return"] = frame["market_return"].shift()
    frame["mean_return_5"] = frame["previous_return"].rolling(5).mean()
    frame["mean_abs_return_5"] = frame["previous_return"].abs().rolling(5).mean()
    frame["mean_abs_return_20"] = frame["previous_return"].abs().rolling(20).mean()
    frame["up"] = frame["market_return"].gt(0).astype(int)
    frame["abs_return"] = frame["market_return"].abs()
    frame = frame.dropna(subset=MARKET_FEATURES + NEWS_FEATURES + ["available_last_news_date"])
    if not frame["available_last_news_date"].lt(frame["cutoff_session"]).all():
        raise ValueError("A forecast has news dated on or after its cutoff session.")
    return frame


def linear_prediction(train, test, columns, target):
    # Least squares with an intercept. Scaling uses training moments only.
    mean = train[columns].mean()
    scale = train[columns].std(ddof=0).replace(0, 1)
    x = np.column_stack([np.ones(len(train)), (train[columns] - mean) / scale])
    new_x = np.column_stack([np.ones(len(test)), (test[columns] - mean) / scale])
    coefficients = np.linalg.lstsq(x, train[target], rcond=None)[0]
    return new_x @ coefficients


def forecast(train, test):
    output = test[["cutoff_session", "available_first_news_date", "available_last_news_date",
                   "market_return", "up", "abs_return", *MARKET_FEATURES, *NEWS_FEATURES]].copy()
    output["base_up_probability"] = train["up"].mean()
    output["base_abs_return"] = train["abs_return"].median()
    for name, columns in (("market", MARKET_FEATURES), ("market_news", MARKET_FEATURES + NEWS_FEATURES)):
        output[f"{name}_up_probability"] = linear_prediction(train, test, columns, "up").clip(0, 1)
        output[f"{name}_abs_return"] = linear_prediction(train, test, columns, "abs_return").clip(min=0)
    return output


def direction_scores(actual, probability):
    rank = pd.Series(probability).rank().to_numpy()
    positive = np.count_nonzero(actual)
    negative = len(actual) - positive
    auc = (rank[actual == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    return {
        "accuracy": float(np.mean((probability >= 0.5) == actual)),
        "auc": float(auc),
        "brier": float(np.mean((probability - actual) ** 2)),
    }


def volatility_scores(actual, prediction):
    errors = prediction - actual
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.mean(errors ** 2) ** 0.5)}


def paired_intervals(brier_delta, absolute_error_delta, squared_error_market, squared_error_news):
    # Resample the same contiguous test-session blocks for all model errors.
    size = len(brier_delta)
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, size - BLOCK_LENGTH + 1,
                         size=(RESAMPLES, (size + BLOCK_LENGTH - 1) // BLOCK_LENGTH))
    blocks = (starts[..., None] + np.arange(BLOCK_LENGTH)).reshape(RESAMPLES, -1)[:, :size]
    samples = {
        "brier": brier_delta[blocks].mean(axis=1),
        "mae": absolute_error_delta[blocks].mean(axis=1),
        "rmse": np.sqrt(squared_error_news[blocks].mean(axis=1)) -
                np.sqrt(squared_error_market[blocks].mean(axis=1)),
    }
    return {name: np.percentile(values, [2.5, 97.5]).tolist() for name, values in samples.items()}


def audit_alignment(daily, sessions):
    reused = daily["market_date"].eq(daily["market_date"].shift())
    calendar = daily.assign(reused=reused, lag=daily["market_return"].shift())
    calendar = calendar[calendar["market_date"].between(START_TEST, END_TEST)].dropna(subset=["lag"])
    session = sessions.assign(lag=sessions["market_return"].shift()).loc[START_TEST:END_TEST]
    calendar_correct = calendar["market_return"].gt(0).eq(calendar["lag"].gt(0))
    return {
        "headline_count": int(daily["article_count"].sum()),
        "publication_days": len(daily),
        "sessions": len(sessions),
        "reused_targets": int(reused.sum()),
        "reuse_fraction": float(reused.mean()),
        "calendar_test_rows": len(calendar),
        "calendar_lag_accuracy": float(calendar_correct.mean()),
        "calendar_test_reused": int(calendar["reused"].sum()),
        "calendar_test_reused_correct": int(calendar_correct[calendar["reused"]].sum()),
        "session_test_rows": len(session),
        "session_lag_accuracy": float(session["market_return"].gt(0).eq(session["lag"].gt(0)).mean()),
        "example": daily.loc[daily["market_date"].eq("2023-01-09"),
                             ["published_date", "market_date", "market_return"]].astype(
                                 {"published_date": str, "market_date": str}).to_dict("records"),
    }


def summarize(daily, sessions, train, output):
    actual_up = output["up"].to_numpy()
    actual_abs = output["abs_return"].to_numpy()
    direction = {col: direction_scores(actual_up, output[col].to_numpy()) for col in PROBABILITY_COLUMNS}
    volatility = {col: volatility_scores(actual_abs, output[col].to_numpy()) for col in VOLATILITY_COLUMNS}
    market_error = output["market_abs_return"].to_numpy() - actual_abs
    news_error = output["market_news_abs_return"].to_numpy() - actual_abs
    brier_delta = (output["market_news_up_probability"].to_numpy() - actual_up) ** 2 - (
        output["market_up_probability"].to_numpy() - actual_up) ** 2
    mae_delta = np.abs(news_error) - np.abs(market_error)
    descriptive = sessions.assign(previous=sessions["market_return"].shift(),
                                  abs_return=sessions["market_return"].abs()).loc[:END_TEST]
    descriptive_test = descriptive.loc[START_TEST:END_TEST]
    return {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "publication_range": [daily["published_date"].min().date().isoformat(),
                              daily["published_date"].max().date().isoformat()],
        "session_range": [sessions.index.min().date().isoformat(), sessions.index.max().date().isoformat()],
        "audit": audit_alignment(daily, sessions),
        "training": {"sessions": len(train), "first": str(train.index.min().date()),
                     "last": str(train.index.max().date())},
        "holdout": {"sessions": len(output), "first": str(output.index.min().date()),
                    "last": str(output.index.max().date()), "up_fraction": float(actual_up.mean())},
        "associations": {
            "tone_previous": descriptive["sentiment"].corr(descriptive["previous"]),
            "tone_next": descriptive["sentiment"].corr(descriptive["market_return"]),
            "dispersion_target": descriptive["dispersion"].corr(descriptive["abs_return"]),
            "dispersion_holdout": descriptive_test["dispersion"].corr(descriptive_test["abs_return"]),
            "buffered_dispersion_holdout": output["available_dispersion"].corr(output["abs_return"]),
        },
        "direction": direction,
        "volatility": volatility,
        "bootstrap": {"block_sessions": BLOCK_LENGTH, "resamples": RESAMPLES, "seed": SEED,
                      "intervals": paired_intervals(brier_delta, mae_delta, market_error ** 2, news_error ** 2)},
        "yearly_loss_change": [
            {"year": int(year), "brier": float(brier_delta[output.index.year == year].mean()),
             "mae": float(mae_delta[output.index.year == year].mean())}
            for year in sorted(set(output.index.year))
        ],
    }


def main():
    daily = load_daily()
    sessions = pool_sessions(daily)
    features = make_features(sessions)
    train = features.loc[features.index < START_TEST]
    holdout = features.loc[START_TEST:END_TEST]
    predictions = forecast(train, holdout)
    results = summarize(daily, sessions, train, predictions)
    predictions.to_csv(DATA / "holdout_predictions.csv")
    (DATA / "research_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Audited {results['audit']['publication_days']:,} publication dates for "
          f"{results['audit']['sessions']:,} market sessions; evaluated {len(holdout):,} holdout sessions.")


if __name__ == "__main__":
    main()
