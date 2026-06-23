"""Model validation metrics and prediction tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold

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
    if n_samples < 2:
        raise ValueError("Cross-validation requires at least 2 training samples.")
    return max(2, min(int(requested_folds), int(n_samples)))


def make_cv(n_samples: int, folds: int = 5, repeats: int = 1, random_state: int = 42):
    folds = adjusted_cv_folds(n_samples, folds)
    if repeats > 1:
        return RepeatedKFold(n_splits=folds, n_repeats=int(repeats), random_state=random_state)
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def _r2_or_nan(y_true, y_pred) -> float:
    if len(y_true) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def cross_validated_predictions_and_scores(estimator, X: pd.DataFrame, y: pd.Series, cv) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate CV predictions and fold metrics with one set of model fits.

    The estimator is cloned and refit inside each training fold, preserving
    the no-leakage contract for scalers, PCA/PCR steps, and model fitting.
    Repeated CV produces multiple predictions per sample; those predictions
    are averaged for the aggregate Q2/RMSE/MAE statistics.
    """

    X_frame = pd.DataFrame(X)
    y_series = pd.Series(y, index=X_frame.index).astype(float)
    pred_sum = np.zeros(len(y_series), dtype=float)
    pred_count = np.zeros(len(y_series), dtype=int)
    rows: list[dict[str, float | int]] = []

    for fold, (train_idx, validation_idx) in enumerate(cv.split(X_frame, y_series), start=1):
        fold_estimator = clone(estimator)
        X_fold_train = X_frame.iloc[train_idx]
        y_fold_train = y_series.iloc[train_idx]
        X_fold_validation = X_frame.iloc[validation_idx]
        y_fold_validation = y_series.iloc[validation_idx]
        fold_estimator.fit(X_fold_train, y_fold_train)
        fold_pred = flatten_prediction(fold_estimator.predict(X_fold_validation))
        pred_sum[validation_idx] += fold_pred
        pred_count[validation_idx] += 1
        rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_validation": int(len(validation_idx)),
                "R2": _r2_or_nan(y_fold_validation, fold_pred),
                "RMSE": rmse(y_fold_validation, fold_pred),
                "MAE": float(mean_absolute_error(y_fold_validation, fold_pred)),
            }
        )

    if (pred_count == 0).any():
        raise ValueError("Cross-validation did not generate predictions for every training sample.")
    return pred_sum / pred_count, pd.DataFrame(rows)


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
    cv_pred, cv_scores = cross_validated_predictions_and_scores(estimator, X_train, y_train, cv)
    cv_r2_scores = cv_scores["R2"].dropna().to_numpy(dtype=float)
    cv_rmse_scores = cv_scores["RMSE"].dropna().to_numpy(dtype=float)
    cv_mae_scores = cv_scores["MAE"].dropna().to_numpy(dtype=float)

    train_slope, train_intercept = regression_line(y_train, train_pred)
    test_slope, test_intercept = regression_line(y_test, test_pred) if len(y_test) else (np.nan, np.nan)

    residuals_train = train_pred - np.asarray(y_train)
    residuals_test = test_pred - np.asarray(y_test) if len(y_test) else np.array([])

    metrics: dict[str, Any] = {
        "R2 train": float(r2_score(y_train, train_pred)),
        "R2 test": float(r2_score(y_test, test_pred)) if len(y_test) > 1 else np.nan,
        "Q2 CV": float(r2_score(y_train, cv_pred)) if len(y_train) > 1 else np.nan,
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
    if cv_scores["R2"].isna().any():
        warnings.append("Some CV folds had fewer than 2 validation samples, so fold-level R2 is undefined for those folds.")
    train_predictions = prediction_frame(X_train.index, y_train, train_pred, "train")
    test_predictions = prediction_frame(X_test.index, y_test, test_pred, "test")
    cv_predictions = prediction_frame(X_train.index, y_train, cv_pred, "cv")
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
