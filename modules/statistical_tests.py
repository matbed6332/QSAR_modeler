"""Endpoint-oriented statistical screening helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class EndpointGrouping:
    labels: pd.Series
    low_label: str
    high_label: str
    description: str


def endpoint_groups(y: pd.Series, method: str = "Lower vs upper quartile") -> EndpointGrouping:
    """Create two endpoint-defined groups suitable for exploratory tests."""

    values = pd.Series(y).astype(float).dropna()
    if values.shape[0] < 4:
        raise ValueError("Endpoint statistical screening requires at least 4 samples.")

    labels = pd.Series("unused", index=values.index, dtype="object")
    if method == "Median split":
        threshold = float(values.median())
        low_mask = values <= threshold
        high_mask = values > threshold
        description = f"Low <= median ({threshold:.5g}); high > median."
    elif method == "Lower vs upper quartile":
        low_threshold = float(values.quantile(0.25))
        high_threshold = float(values.quantile(0.75))
        low_mask = values <= low_threshold
        high_mask = values >= high_threshold
        description = f"Low <= Q1 ({low_threshold:.5g}); high >= Q3 ({high_threshold:.5g})."
    else:
        raise ValueError(f"Unsupported endpoint grouping method: {method}")

    labels.loc[low_mask] = "low_endpoint"
    labels.loc[high_mask] = "high_endpoint"
    if (labels == "low_endpoint").sum() < 2 or (labels == "high_endpoint").sum() < 2:
        raise ValueError("Endpoint grouping produced fewer than 2 samples in one of the compared groups.")

    return EndpointGrouping(labels=labels, low_label="low_endpoint", high_label="high_endpoint", description=description)


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if not np.isfinite(pooled_var) or np.isclose(pooled_var, 0.0):
        return np.nan
    return float((b.mean() - a.mean()) / np.sqrt(pooled_var))


def compare_endpoint_groups(
    scores: pd.DataFrame,
    y: pd.Series,
    score_columns: list[str],
    grouping_method: str = "Lower vs upper quartile",
    test_name: str = "Welch t-test",
) -> tuple[pd.DataFrame, EndpointGrouping]:
    """Compare numeric variable distributions between low/high endpoint groups."""

    grouping = endpoint_groups(y, grouping_method)
    aligned = pd.DataFrame(scores).reindex(grouping.labels.index)
    rows: list[dict[str, float | int | str]] = []

    for column in score_columns:
        low = pd.to_numeric(aligned.loc[grouping.labels == grouping.low_label, column], errors="coerce").dropna()
        high = pd.to_numeric(aligned.loc[grouping.labels == grouping.high_label, column], errors="coerce").dropna()
        if len(low) < 2 or len(high) < 2:
            statistic = np.nan
            p_value = np.nan
        elif test_name == "Student t-test":
            statistic, p_value = stats.ttest_ind(low, high, equal_var=True, nan_policy="omit")
        elif test_name == "Welch t-test":
            statistic, p_value = stats.ttest_ind(low, high, equal_var=False, nan_policy="omit")
        elif test_name == "Mann-Whitney U":
            statistic, p_value = stats.mannwhitneyu(low, high, alternative="two-sided")
        else:
            raise ValueError(f"Unsupported statistical test: {test_name}")

        rows.append(
            {
                "variable": column,
                "test": test_name,
                "low_n": int(len(low)),
                "high_n": int(len(high)),
                "low_mean": float(low.mean()) if len(low) else np.nan,
                "high_mean": float(high.mean()) if len(high) else np.nan,
                "mean_difference_high_minus_low": float(high.mean() - low.mean()) if len(low) and len(high) else np.nan,
                "effect_size_cohens_d": _cohens_d(low, high),
                "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["significant_0_05"] = result["p_value"] < 0.05
    return result, grouping


def endpoint_outlier_table(y: pd.Series, z_threshold: float = 3.0, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Flag endpoint outliers by classical z-score and IQR fences."""

    values = pd.Series(y).astype(float).dropna()
    if values.empty:
        return pd.DataFrame(columns=["sample_id", "endpoint", "z_score", "iqr_flag", "z_flag", "flagged"])

    std = values.std(ddof=1)
    z_scores = (values - values.mean()) / std if np.isfinite(std) and not np.isclose(std, 0.0) else pd.Series(0.0, index=values.index)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    iqr_flag = (values < lower) | (values > upper) if np.isfinite(iqr) and not np.isclose(iqr, 0.0) else pd.Series(False, index=values.index)
    z_flag = z_scores.abs() >= float(z_threshold)

    return pd.DataFrame(
        {
            "sample_id": values.index.astype(str),
            "endpoint": values.values,
            "z_score": z_scores.values,
            "iqr_lower_fence": float(lower),
            "iqr_upper_fence": float(upper),
            "iqr_flag": iqr_flag.values,
            "z_flag": z_flag.values,
            "flagged": (iqr_flag | z_flag).values,
        },
        index=values.index,
    )
