"""Applicability domain calculations for QSAR/QSPR models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def _standardize(train_X: pd.DataFrame, test_X: pd.DataFrame | None = None):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    if test_X is None:
        return train_scaled, None
    return train_scaled, scaler.transform(test_X)


def leverage_values(train_X: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    train_scaled, X_scaled = _standardize(train_X, X)
    design_train = np.column_stack([np.ones(train_scaled.shape[0]), train_scaled])
    design_X = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
    hat_inv = np.linalg.pinv(design_train.T @ design_train)
    return np.einsum("ij,jk,ik->i", design_X, hat_inv, design_X)


def williams_results(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    train_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate leverage and standardized residuals for Williams plots."""

    train_leverage = leverage_values(train_X, train_X)
    test_leverage = leverage_values(train_X, test_X) if len(test_X) else np.array([])
    p = train_X.shape[1]
    n = train_X.shape[0]
    warning_leverage = 3 * (p + 1) / max(n, 1)

    residual_std = train_predictions["residual"].std(ddof=1)
    if not np.isfinite(residual_std) or np.isclose(residual_std, 0.0):
        residual_std = 1.0

    train_frame = pd.DataFrame(
        {
            "sample_id": train_predictions["sample_id"].astype(str).values,
            "split": "train",
            "leverage": train_leverage,
            "standardized_residual": train_predictions["residual"].values / residual_std,
            "h_warning": warning_leverage,
        },
        index=train_predictions.index,
    )
    test_frame = pd.DataFrame(
        {
            "sample_id": test_predictions["sample_id"].astype(str).values,
            "split": "test",
            "leverage": test_leverage,
            "standardized_residual": test_predictions["residual"].values / residual_std if len(test_predictions) else [],
            "h_warning": warning_leverage,
        },
        index=test_predictions.index,
    )
    result = pd.concat([train_frame, test_frame], axis=0)
    result["outside_leverage"] = result["leverage"] > warning_leverage
    result["outside_residual"] = result["standardized_residual"].abs() > 3
    result["outside_ad"] = result["outside_leverage"] | result["outside_residual"]
    return result


def distance_domain_results(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """Insubria-style AD proxy based on distance to the training-set centroid."""

    train_scaled, test_scaled = _standardize(train_X, test_X)
    centroid = train_scaled.mean(axis=0, keepdims=True)
    train_distance = pairwise_distances(train_scaled, centroid).ravel()
    test_distance = pairwise_distances(test_scaled, centroid).ravel() if len(test_X) else np.array([])
    threshold = float(np.quantile(train_distance, quantile))

    train_frame = pd.DataFrame(
        {
            "sample_id": train_X.index.astype(str),
            "split": "train",
            "distance_to_train_centroid": train_distance,
            "distance_threshold": threshold,
        },
        index=train_X.index,
    )
    test_frame = pd.DataFrame(
        {
            "sample_id": test_X.index.astype(str),
            "split": "test",
            "distance_to_train_centroid": test_distance,
            "distance_threshold": threshold,
        },
        index=test_X.index,
    )
    result = pd.concat([train_frame, test_frame], axis=0)
    result["outside_ad"] = result["distance_to_train_centroid"] > threshold
    return result


def pca_domain_scores(train_X: pd.DataFrame, test_X: pd.DataFrame) -> pd.DataFrame:
    n_components = min(2, train_X.shape[1], train_X.shape[0])
    if n_components < 2:
        return pd.DataFrame()
    train_scaled, test_scaled = _standardize(train_X, test_X)
    pca = PCA(n_components=2)
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled) if len(test_X) else np.empty((0, 2))
    train_frame = pd.DataFrame(
        {"sample_id": train_X.index.astype(str), "split": "train", "PC1": train_scores[:, 0], "PC2": train_scores[:, 1]},
        index=train_X.index,
    )
    test_frame = pd.DataFrame(
        {"sample_id": test_X.index.astype(str), "split": "test", "PC1": test_scores[:, 0], "PC2": test_scores[:, 1]},
        index=test_X.index,
    )
    result = pd.concat([train_frame, test_frame], axis=0)
    result.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    return result

