"""Model validation metrics and prediction tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_val_score

from modules.models import flatten_prediction


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    cv_predictions: pd.DataFrame
    cv_scores: pd.DataFrame
    warnings: list[str]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def adjusted_cv_folds(n_samples: int, requested_folds: int) -> int:
    return max(2, min(int(requested_folds), int(n_samples)))


def make_cv(n_samples: int, folds: int = 5, repeats: int = 1, random_state: int = 42):
    folds = adjusted_cv_folds(n_samples, folds)
    if repeats > 1:
        return RepeatedKFold(n_splits=folds, n_repeats=int(repeats), random_state=random_state)
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def regression_line(y_true, y_pred) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.isclose(np.std(y_true), 0.0):
        return np.nan, np.nan
    slope, intercept = np.polyfit(y_true, y_pred, deg=1)
    return float(slope), float(intercept)


def prediction_frame(index, y_true, y_pred, split: str) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return pd.DataFrame(
        {
            "sample_id": pd.Index(index).astype(str),
            "split": split,
            "observed": y_true,
            "predicted": y_pred,
            "residual": y_pred - y_true,
            "absolute_error": np.abs(y_pred - y_true),
        },
        index=index,
    )


def evaluate_fitted_model(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    cv_repeats: int = 1,
    random_state: int = 42,
) -> EvaluationResult:
    train_pred = flatten_prediction(estimator.predict(X_train))
    test_pred = flatten_prediction(estimator.predict(X_test)) if len(X_test) else np.array([])

    cv = make_cv(len(y_train), cv_folds, cv_repeats, random_state)
    cv_estimator = clone(estimator)
    cv_pred = flatten_prediction(cross_val_predict(cv_estimator, X_train, y_train, cv=cv, n_jobs=None))
    cv_r2_scores = cross_val_score(clone(estimator), X_train, y_train, cv=cv, scoring="r2", n_jobs=None)
    cv_rmse_scores = -cross_val_score(
        clone(estimator), X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=None
    )
    cv_mae_scores = -cross_val_score(
        clone(estimator), X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=None
    )

    train_slope, train_intercept = regression_line(y_train, train_pred)
    test_slope, test_intercept = regression_line(y_test, test_pred) if len(y_test) else (np.nan, np.nan)

    residuals_train = train_pred - np.asarray(y_train)
    residuals_test = test_pred - np.asarray(y_test) if len(y_test) else np.array([])

    metrics: dict[str, Any] = {
        "R2 train": float(r2_score(y_train, train_pred)),
        "R2 test": float(r2_score(y_test, test_pred)) if len(y_test) > 1 else np.nan,
        "Q2 CV": float(r2_score(y_train, cv_pred)),
        "RMSE train": rmse(y_train, train_pred),
        "RMSE test": rmse(y_test, test_pred) if len(y_test) else np.nan,
        "RMSE CV": rmse(y_train, cv_pred),
        "MAE train": float(mean_absolute_error(y_train, train_pred)),
        "MAE test": float(mean_absolute_error(y_test, test_pred)) if len(y_test) else np.nan,
        "MAE CV": float(mean_absolute_error(y_train, cv_pred)),
        "MSE train": float(mean_squared_error(y_train, train_pred)),
        "MSE test": float(mean_squared_error(y_test, test_pred)) if len(y_test) else np.nan,
        "Bias train": float(np.mean(residuals_train)),
        "Bias test": float(np.mean(residuals_test)) if len(residuals_test) else np.nan,
        "CV R2 std": float(np.std(cv_r2_scores, ddof=1)) if len(cv_r2_scores) > 1 else 0.0,
        "CV RMSE std": float(np.std(cv_rmse_scores, ddof=1)) if len(cv_rmse_scores) > 1 else 0.0,
        "Train slope": train_slope,
        "Train intercept": train_intercept,
        "Test slope": test_slope,
        "Test intercept": test_intercept,
        "Residual std train": float(np.std(residuals_train, ddof=1)) if len(residuals_train) > 1 else 0.0,
        "Residual std test": float(np.std(residuals_test, ddof=1)) if len(residuals_test) > 1 else np.nan,
    }

    warnings = diagnostic_warnings(metrics, len(y_train), X_train.shape[1])
    train_predictions = prediction_frame(X_train.index, y_train, train_pred, "train")
    test_predictions = prediction_frame(X_test.index, y_test, test_pred, "test")
    cv_predictions = prediction_frame(X_train.index, y_train, cv_pred, "cv")
    cv_scores = pd.DataFrame(
        {
            "fold": np.arange(1, len(cv_r2_scores) + 1),
            "R2": cv_r2_scores,
            "RMSE": cv_rmse_scores,
            "MAE": cv_mae_scores,
        }
    )
    return EvaluationResult(metrics, train_predictions, test_predictions, cv_predictions, cv_scores, warnings)


def diagnostic_warnings(metrics: dict[str, Any], n_train: int, n_descriptors: int) -> list[str]:
    warnings: list[str] = []
    r2_train = metrics.get("R2 train", np.nan)
    r2_test = metrics.get("R2 test", np.nan)
    q2_cv = metrics.get("Q2 CV", np.nan)
    if np.isfinite(r2_train) and np.isfinite(r2_test) and r2_train > 0.90 and r2_test < 0.50:
        warnings.append("High training R2 with weak test R2 suggests possible overfitting.")
    if np.isfinite(r2_train) and np.isfinite(q2_cv) and r2_train - q2_cv > 0.30:
        warnings.append("Training R2 is much higher than Q2 CV; descriptor/model complexity may be too high.")
    if n_descriptors >= max(1, n_train / 3):
        warnings.append("The descriptor-to-sample ratio is high; consider stronger descriptor selection.")
    return warnings


def results_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for label, payload in results.items():
        row = {"Model label": label}
        row.update(payload.get("metrics", {}))
        row["Model"] = payload.get("model_name")
        row["Candidate"] = payload.get("candidate_index")
        row["Seed"] = payload.get("random_seed")
        row["Descriptors"] = len(payload.get("selected_descriptors", []))
        row["Parameters"] = payload.get("parameters", {})
        rows.append(row)
    return pd.DataFrame(rows)


def rank_models(table: pd.DataFrame, metric: str, ascending: bool | None = None) -> pd.DataFrame:
    if table.empty or metric not in table.columns:
        return table
    if ascending is None:
        ascending = any(token in metric.upper() for token in ["RMSE", "MAE", "MSE", "BIAS"])
    return table.sort_values(metric, ascending=ascending, na_position="last")
