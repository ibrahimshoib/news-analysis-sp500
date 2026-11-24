"""Run the CS334 forecasting experiment using the bundled news and market data.

From the project root: python -B scripts/course_experiment.py
"""

import hashlib
import json

import numpy as np
import pandas as pd

from news_research import (
    BLOCK_LENGTH, DATA, END_TEST, MARKET_FEATURES, NEWS_FEATURES, RESAMPLES, SEED, SOURCE,
    START_TEST, direction_scores, load_daily, make_features, paired_intervals, pool_sessions,
    volatility_scores,
)


FEATURE_SETS = {
    "market": MARKET_FEATURES,
    "news": NEWS_FEATURES,
    "market_plus_news": MARKET_FEATURES + NEWS_FEATURES,
}
REGRESSION = {"ols": (0.0,), "ridge": (0.001, 0.01, 0.1, 1.0, 10.0),
              "lasso": (0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.003)}
CLASSIFICATION = {"logistic": (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0)}
FOLDS = 5


def standardized_design(train, evaluation, columns):
    mean = train[columns].mean()
    scale = train[columns].std(ddof=0).replace(0, 1)
    x = np.column_stack((np.ones(len(train)), (train[columns] - mean) / scale))
    new_x = np.column_stack((np.ones(len(evaluation)), (evaluation[columns] - mean) / scale))
    return x, new_x


def lasso_coefficients(x, y, penalty):
    # Coordinate descent for half the mean squared error plus the L1 penalty.
    # The standardized columns are centered, so the unpenalized intercept is mean(y).
    features = x[:, 1:]
    intercept = y.mean()
    coefficients = np.zeros(features.shape[1])
    residual = y - intercept
    column_squares = np.mean(features ** 2, axis=0)
    for _ in range(2000):
        largest_change = 0.0
        for column_index, (column, square) in enumerate(zip(features.T, column_squares)):
            if square == 0:
                continue
            previous = coefficients[column_index]
            cross_product = column @ residual / len(y) + square * previous
            updated = np.sign(cross_product) * max(abs(cross_product) - penalty, 0.0) / square
            change = updated - previous
            coefficients[column_index] = updated
            residual -= column * change
            largest_change = max(largest_change, abs(change))
        if largest_change < 1e-11:
            return np.r_[intercept, coefficients]
    raise RuntimeError("LASSO coordinate descent did not converge.")


def sigmoid(scores):
    return np.exp(-np.logaddexp(0, -scores))


def logistic_coefficients(x, y, penalty):
    # Newton steps with backtracking for mean logistic loss plus an L2 penalty.
    # The first coefficient is the intercept, which is excluded from the penalty.
    penalties = np.r_[0.0, np.full(x.shape[1] - 1, penalty)]
    coefficients = np.zeros(x.shape[1])

    def objective(weights):
        scores = x @ weights
        return np.mean(np.logaddexp(0, scores) - y * scores) + 0.5 * (penalties * weights) @ weights

    for _ in range(80):
        probabilities = sigmoid(x @ coefficients)
        gradient = x.T @ (probabilities - y) / len(y) + penalties * coefficients
        if np.linalg.norm(gradient, ord=np.inf) < 1e-8:
            return coefficients
        weights = probabilities * (1 - probabilities)
        hessian = (x.T * weights) @ x / len(y) + np.diag(penalties)
        step = np.linalg.solve(hessian, gradient)
        current = objective(coefficients)
        step_size = 1.0
        while step_size >= 1 / 1024:
            candidate = coefficients - step_size * step
            if objective(candidate) <= current:
                coefficients = candidate
                break
            step_size /= 2
        else:
            raise RuntimeError("Logistic line search did not decrease the objective.")
    raise RuntimeError("Logistic regression did not converge.")


def predict(x, y, new_x, family, penalty):
    if family == "ols":
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    elif family == "ridge":
        penalized = np.diag(np.r_[0.0, np.full(x.shape[1] - 1, penalty)])
        coefficients = np.linalg.solve(x.T @ x / len(y) + penalized, x.T @ y / len(y))
    elif family == "lasso":
        coefficients = lasso_coefficients(x, y, penalty)
    elif family == "logistic":
        coefficients = logistic_coefficients(x, y, penalty)
        return sigmoid(new_x @ coefficients)
    else:
        raise ValueError(f"Unknown model family: {family}")
    return np.maximum(new_x @ coefficients, 0)


def expanding_folds(train):
    # An initial training window followed by five contiguous validation blocks.
    first_validation = len(train) // 2
    for block in np.array_split(np.arange(first_validation, len(train)), FOLDS):
        yield train.iloc[:block[0]], train.iloc[block]


def evaluate(train, test, folds, target, families):
    predictions = {}
    validation = {}
    selected = {}
    for feature_name, columns in FEATURE_SETS.items():
        # Fit preprocessing within each fold and again on the full training period.
        designs = [(*standardized_design(fit, valid, columns),
                    fit[target].to_numpy(), valid[target].to_numpy()) for fit, valid in folds]
        x, new_x = standardized_design(train, test, columns)
        y = train[target].to_numpy()
        for family, penalties in families.items():
            model = f"{family}_{feature_name}"
            candidates = []
            for penalty in penalties:
                errors = [predict(fit_x, fit_y, valid_x, family, penalty) - valid_y
                          for fit_x, valid_x, fit_y, valid_y in designs]
                losses = [float(np.mean(error ** 2)) for error in errors]
                mean_loss = float(np.average(losses, weights=[len(error) for error in errors]))
                candidates.append({"penalty": penalty, "validation_mse": mean_loss, "fold_mse": losses})
            best = min(candidates, key=lambda candidate: candidate["validation_mse"])
            validation[model] = {"best": best, "candidates": candidates}
            predictions[model] = predict(x, y, new_x, family, best["penalty"])
        selected[feature_name] = min(
            (f"{family}_{feature_name}" for family in families),
            key=lambda model: validation[model]["best"]["validation_mse"],
        )
    predictions["constant"] = np.full(len(test), train[target].mean())
    return predictions, validation, selected


def classification_metrics(actual, probability):
    result = direction_scores(actual, probability)
    positive = probability >= 0.5
    true_positive = int(np.count_nonzero(positive & (actual == 1)))
    false_positive = int(np.count_nonzero(positive & (actual == 0)))
    false_negative = int(np.count_nonzero(~positive & (actual == 1)))
    true_negative = len(actual) - (true_positive + false_positive + false_negative)
    return {
        **result,
        "precision": float(true_positive / max(1, true_positive + false_positive)),
        "recall": float(true_positive / max(1, true_positive + false_negative)),
        "f1": float(2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def regression_metrics(actual, prediction):
    result = volatility_scores(actual, prediction)
    return {**result, "r_squared": float(1 - np.sum((actual - prediction) ** 2) /
                                        np.sum((actual - actual.mean()) ** 2))}


def describe(frame):
    columns = ["available_sentiment", "available_dispersion", "available_log_article_count",
               "market_return", "abs_return"]
    return {
        "sessions": len(frame),
        "first_session": str(frame.index.min().date()),
        "last_session": str(frame.index.max().date()),
        "up_fraction": float(frame["up"].mean()),
        "statistics": frame[columns].describe().to_dict(),
        "feature_target_pearson": frame[columns].corr().loc[columns[:3], columns[3:]].to_dict(),
    }


def compare_news(actual_up, actual_abs, probabilities, magnitudes, direction_selected, magnitude_selected):
    market_brier = (probabilities[direction_selected["market"]] - actual_up) ** 2
    news_brier = (probabilities[direction_selected["market_plus_news"]] - actual_up) ** 2
    market_error = magnitudes[magnitude_selected["market"]] - actual_abs
    news_error = magnitudes[magnitude_selected["market_plus_news"]] - actual_abs
    brier_delta = news_brier - market_brier
    mae_delta = np.abs(news_error) - np.abs(market_error)
    market_squared = market_error ** 2
    news_squared = news_error ** 2
    return {
        "comparison": "market-plus-news minus market-only; negative values favor adding news",
        "estimates": {"brier": float(brier_delta.mean()), "mae": float(mae_delta.mean()),
                      "rmse": float(np.sqrt(news_squared.mean()) - np.sqrt(market_squared.mean()))},
        "paired_95_percent_intervals": paired_intervals(brier_delta, mae_delta, market_squared, news_squared),
        "block_sessions": BLOCK_LENGTH,
        "resamples": RESAMPLES,
        "seed": SEED,
    }


def main():
    daily = load_daily()
    sessions = pool_sessions(daily)
    frame = make_features(sessions)
    train = frame.loc[frame.index < START_TEST]
    test = frame.loc[START_TEST:END_TEST]
    folds = list(expanding_folds(train))
    probabilities, direction_cv, direction_selected = evaluate(train, test, folds, "up", CLASSIFICATION)
    magnitudes, magnitude_cv, magnitude_selected = evaluate(train, test, folds, "abs_return", REGRESSION)
    actual_up = test["up"].to_numpy()
    actual_abs = test["abs_return"].to_numpy()
    results = {
        "source": SOURCE.name,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_counts": {"headlines": int(daily["article_count"].sum()),
                          "publication_rows": len(daily), "unique_target_sessions": len(sessions)},
        "coverage": {"training": describe(train), "test": describe(test)},
        "test_status": "retrospective chronological holdout previously examined in this project",
        "prediction_cutoff": "preceding session close; headlines dated before that session only",
        "features": FEATURE_SETS,
        "validation_folds": [
            {"fit_first": str(fit.index.min().date()), "fit_last": str(fit.index.max().date()),
             "fit_sessions": len(fit), "validation_first": str(valid.index.min().date()),
             "validation_last": str(valid.index.max().date()), "validation_sessions": len(valid)}
            for fit, valid in folds
        ],
        "validation_mse_interpretation": {"direction": "Brier score", "magnitude": "squared return error"},
        "validation": {"direction": direction_cv, "magnitude": magnitude_cv},
        "selected_using_validation_only": {"direction": direction_selected, "magnitude": magnitude_selected},
        "test_metrics": {
            "direction": {model: classification_metrics(actual_up, values)
                          for model, values in probabilities.items()},
            "magnitude": {model: regression_metrics(actual_abs, values)
                          for model, values in magnitudes.items()},
        },
        "adding_news": compare_news(actual_up, actual_abs, probabilities, magnitudes,
                                    direction_selected, magnitude_selected),
    }
    predictions = test[["cutoff_session", "available_first_news_date", "available_last_news_date",
                        "market_return", "up", "abs_return"]].assign(
        **{f"up_{model}": values for model, values in probabilities.items()},
        **{f"abs_{model}": values for model, values in magnitudes.items()},
    )
    serialized_results = json.dumps(results, indent=2, allow_nan=False)
    predictions.to_csv(DATA / "course_experiment_predictions.csv")
    (DATA / "course_experiment_results.json").write_text(serialized_results + "\n", encoding="utf-8")

    print(f"Training sessions: {len(train):,} | Test sessions: {len(test):,}")
    print("\nDirection\n" + pd.DataFrame(results["test_metrics"]["direction"]).T[
        ["accuracy", "auc", "brier", "f1"]].to_string())
    print("\nAbsolute-return magnitude\n" +
          pd.DataFrame(results["test_metrics"]["magnitude"]).T.to_string())
    print("\nSelected by training validation")
    print(json.dumps(results["selected_using_validation_only"], indent=2))
    print("\nHoldout error changes from adding news")
    print(json.dumps(results["adding_news"], indent=2))


if __name__ == "__main__":
    main()
