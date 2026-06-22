"""Exploratory PCA screening before train/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAScreeningResult:
    scores: pd.DataFrame
    variance: pd.DataFrame
    preprocessing: pd.DataFrame


def compute_pca_screening(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    scale: bool = True,
) -> PCAScreeningResult:
    """Fit exploratory PCA on the current curated descriptor matrix.

    This is intentionally separate from the modeling pipeline. It is used for
    visual sample screening before the external split is created.
    """

    X_numeric = pd.DataFrame(X).copy()
    X_numeric.columns = X_numeric.columns.astype(str)
    X_numeric = X_numeric.apply(pd.to_numeric, errors="coerce")
    initial_descriptors = X_numeric.shape[1]

    all_missing_cols = X_numeric.columns[X_numeric.isna().all(axis=0)].tolist()
    X_numeric = X_numeric.drop(columns=all_missing_cols, errors="ignore")

    imputed_cols = X_numeric.columns[X_numeric.isna().any(axis=0)].tolist()
    medians = X_numeric.median(axis=0).fillna(0.0)
    X_numeric = X_numeric.fillna(medians)

    constant_cols = X_numeric.columns[X_numeric.nunique(dropna=False) <= 1].tolist()
    X_numeric = X_numeric.drop(columns=constant_cols, errors="ignore")
    if X_numeric.empty:
        raise ValueError("PCA cannot be computed because no non-constant numeric descriptors remain.")

    max_components = min(X_numeric.shape[0], X_numeric.shape[1])
    n_components = max(1, min(int(n_components), max_components))

    matrix = StandardScaler().fit_transform(X_numeric) if scale else X_numeric.to_numpy(dtype=float)
    pca = PCA(n_components=n_components)
    score_values = pca.fit_transform(matrix)

    scores = pd.DataFrame(
        score_values,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
        index=X_numeric.index,
    )
    scores.insert(0, "sample_id", X_numeric.index.astype(str))
    scores.insert(1, "endpoint", pd.Series(y, index=X_numeric.index).astype(float).values)

    variance = pd.DataFrame(
        {
            "PC": [f"PC{i}" for i in range(1, n_components + 1)],
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_percent": pca.explained_variance_ratio_ * 100.0,
            "cumulative_variance_percent": np.cumsum(pca.explained_variance_ratio_) * 100.0,
        }
    )

    preprocessing = pd.DataFrame(
        [
            {"Step": "Initial numeric descriptors", "Count": initial_descriptors, "Details": ""},
            {"Step": "All-missing descriptors removed", "Count": len(all_missing_cols), "Details": ", ".join(all_missing_cols)},
            {"Step": "Descriptors median-imputed", "Count": len(imputed_cols), "Details": ", ".join(imputed_cols)},
            {"Step": "Constant descriptors removed", "Count": len(constant_cols), "Details": ", ".join(constant_cols)},
            {"Step": "Descriptors used for PCA", "Count": X_numeric.shape[1], "Details": ""},
            {"Step": "Samples used for PCA", "Count": X_numeric.shape[0], "Details": ""},
        ]
    )
    return PCAScreeningResult(scores=scores, variance=variance, preprocessing=preprocessing)
