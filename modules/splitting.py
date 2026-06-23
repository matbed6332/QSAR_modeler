"""Train/test splitting strategies for QSAR/QSPR modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    membership: pd.DataFrame
    warnings: list[str]


def _split_summary(y_train: pd.Series, y_test: pd.Series) -> list[str]:
    warnings: list[str] = []
    if not y_test.empty:
        if y_test.min() < y_train.min() or y_test.max() > y_train.max():
            warnings.append(
                "The test endpoint range extends outside the training endpoint range; "
                "external predictions may be extrapolative."
            )
    return warnings


def random_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify_bins: int | None = None,
) -> SplitResult:
    """Random train/test split with optional binned-y stratification."""

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of aligned samples before splitting.")
    if len(y) < 4:
        raise ValueError("At least 4 samples are required to create an external train/test split.")
    test_size = float(test_size)
    if not 0.0 < test_size < 1.0:
        raise ValueError("Test set fraction must be between 0 and 1.")
    n_test = int(np.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    if n_train < 2 or n_test < 1:
        raise ValueError("Split fraction leaves too few train or test samples.")

    stratify = None
    warnings: list[str] = []
    if stratify_bins and stratify_bins > 1:
        try:
            bins = pd.qcut(y, q=stratify_bins, duplicates="drop")
            if bins.value_counts().min() >= 2 and bins.nunique() > 1:
                stratify = bins
            else:
                warnings.append("Stratified regression split was disabled because some bins had fewer than 2 samples.")
        except ValueError:
            warnings.append("Stratified regression split was disabled because endpoint bins could not be formed.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_index = set(X_train.index)
    membership = pd.DataFrame(
        {
            "sample_id": X.index.astype(str),
            "endpoint": y.values,
            "split": np.where(X.index.isin(train_index), "train", "test"),
        },
        index=X.index,
    )
    warnings.extend(_split_summary(y_train, y_test))
    return SplitResult(X_train, X_test, y_train, y_test, membership, warnings)


def sorted_endpoint_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_fraction: float = 0.75,
    strategy: str = "systematic",
    random_state: int = 42,
) -> SplitResult:
    """Split sorted by endpoint and force the min/max endpoint samples into training."""

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of aligned samples before splitting.")
    if len(y) < 4:
        raise ValueError("Sorted endpoint split requires at least 4 samples.")
    train_fraction = float(train_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("Training fraction must be between 0 and 1.")

    sorted_index = y.sort_values().index.to_list()
    n_samples = len(sorted_index)
    n_train = int(round(train_fraction * n_samples))
    n_train = min(max(n_train, 2), n_samples - 1)
    n_test = n_samples - n_train

    endpoint_positions = {0, n_samples - 1}
    remaining_positions = [i for i in range(n_samples) if i not in endpoint_positions]

    if strategy == "random_remaining":
        rng = np.random.default_rng(random_state)
        test_positions = set(rng.choice(remaining_positions, size=n_test, replace=False).tolist())
    else:
        if n_test >= len(remaining_positions):
            test_positions = set(remaining_positions)
        else:
            linspace = np.linspace(0, len(remaining_positions) - 1, num=n_test)
            test_positions = {remaining_positions[int(round(pos))] for pos in linspace}
            while len(test_positions) < n_test:
                for pos in remaining_positions:
                    if pos not in test_positions:
                        test_positions.add(pos)
                        break

    train_positions = set(range(n_samples)) - test_positions
    train_index = [sorted_index[pos] for pos in sorted(train_positions)]
    test_index = [sorted_index[pos] for pos in sorted(test_positions)]

    X_train = X.loc[train_index]
    X_test = X.loc[test_index]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]

    membership = pd.DataFrame(
        {
            "sample_id": [str(idx) for idx in sorted_index],
            "endpoint": [float(y.loc[idx]) for idx in sorted_index],
            "split": ["train" if idx in train_index else "test" for idx in sorted_index],
        },
        index=sorted_index,
    )
    warnings = _split_summary(y_train, y_test)
    return SplitResult(X_train, X_test, y_train, y_test, membership, warnings)


def split_range_table(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    def summary_row(label: str, values: pd.Series) -> dict[str, float | int | str]:
        return {
            "Set": label,
            "Samples": int(len(values)),
            "Endpoint min": float(values.min()) if len(values) else np.nan,
            "Endpoint max": float(values.max()) if len(values) else np.nan,
            "Endpoint mean": float(values.mean()) if len(values) else np.nan,
        }

    return pd.DataFrame([summary_row("Train", y_train), summary_row("Test", y_test)])

