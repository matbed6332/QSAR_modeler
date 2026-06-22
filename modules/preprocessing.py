"""Leakage-safe preprocessing for QSAR/QSPR descriptor matrices."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class EndpointTransformer:
    """Endpoint transformation with validation and metadata."""

    method: str = "none"

    def transform(self, y: pd.Series) -> pd.Series:
        values = pd.Series(y, index=y.index, name=y.name).astype(float)
        if self.method == "none":
            return values
        if self.method == "log10":
            if (values <= 0).any():
                raise ValueError("log10 endpoint transformation requires all y values to be > 0.")
            return pd.Series(np.log10(values), index=y.index, name=y.name)
        if self.method == "negative_log10":
            if (values <= 0).any():
                raise ValueError("-log10 endpoint transformation requires all y values to be > 0.")
            return pd.Series(-np.log10(values), index=y.index, name=y.name)
        raise ValueError(f"Unsupported endpoint transformation: {self.method}")


@dataclass
class PreprocessingConfig:
    missing_strategy: str = "median_impute"
    remove_constant: bool = True
    remove_low_variance: bool = True
    variance_threshold: float = 0.0
    remove_high_correlation: bool = True
    correlation_threshold: float = 0.90


@dataclass
class PreprocessingReport:
    initial_descriptors: int
    final_descriptors: int
    missing_strategy: str
    dropped_missing_columns: list[str] = field(default_factory=list)
    imputed_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    low_variance_columns: list[str] = field(default_factory=list)
    correlated_columns: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("Initial descriptors", self.initial_descriptors, ""),
            ("Missing-strategy columns removed", len(self.dropped_missing_columns), ", ".join(self.dropped_missing_columns)),
            ("Imputed columns", len(self.imputed_columns), ", ".join(self.imputed_columns)),
            ("Constant columns removed", len(self.constant_columns), ", ".join(self.constant_columns)),
            ("Low-variance columns removed", len(self.low_variance_columns), ", ".join(self.low_variance_columns)),
            ("Highly correlated columns removed", len(self.correlated_columns), ", ".join(self.correlated_columns)),
            ("Final descriptors", self.final_descriptors, ""),
        ]
        return pd.DataFrame(rows, columns=["Step", "Count", "Descriptors"])


class DescriptorPreprocessor(BaseEstimator, TransformerMixin):
    """Fit descriptor preprocessing only on training descriptors.

    The fitted object stores all columns removed or imputed on the training set
    and applies the exact same transformations to validation, test, and new data.
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DescriptorPreprocessor":
        X_work = self._coerce_frame(X)
        self.initial_columns_ = X_work.columns.tolist()
        self.dropped_missing_columns_: list[str] = []
        self.imputed_columns_: list[str] = []
        self.impute_values_: dict[str, float] = {}
        self.constant_columns_: list[str] = []
        self.low_variance_columns_: list[str] = []
        self.correlated_columns_: list[str] = []

        if self.config.missing_strategy == "drop_columns":
            self.dropped_missing_columns_ = X_work.columns[X_work.isna().any(axis=0)].tolist()
            X_work = X_work.drop(columns=self.dropped_missing_columns_, errors="ignore")
        elif self.config.missing_strategy in {"mean_impute", "median_impute"}:
            self.imputed_columns_ = X_work.columns[X_work.isna().any(axis=0)].tolist()
            for col in X_work.columns:
                if self.config.missing_strategy == "mean_impute":
                    value = X_work[col].mean()
                else:
                    value = X_work[col].median()
                if pd.isna(value):
                    value = 0.0
                self.impute_values_[col] = float(value)
            X_work = X_work.fillna(self.impute_values_)
        elif self.config.missing_strategy == "none":
            if X_work.isna().any().any():
                raise ValueError("Missing values remain. Choose imputation or column removal.")
        else:
            raise ValueError(f"Unsupported missing-value strategy: {self.config.missing_strategy}")

        if self.config.remove_constant:
            nunique = X_work.nunique(dropna=False)
            self.constant_columns_ = nunique[nunique <= 1].index.tolist()
            X_work = X_work.drop(columns=self.constant_columns_, errors="ignore")

        if self.config.remove_low_variance and not X_work.empty:
            variances = X_work.var(axis=0, ddof=0)
            self.low_variance_columns_ = variances[variances <= self.config.variance_threshold].index.tolist()
            X_work = X_work.drop(columns=self.low_variance_columns_, errors="ignore")

        if self.config.remove_high_correlation and X_work.shape[1] > 1:
            corr = X_work.corr().abs().fillna(0.0)
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            self.correlated_columns_ = [
                column for column in upper.columns if (upper[column] > self.config.correlation_threshold).any()
            ]
            X_work = X_work.drop(columns=self.correlated_columns_, errors="ignore")

        if X_work.empty:
            raise ValueError("All descriptors were removed during preprocessing.")

        self.final_columns_ = X_work.columns.tolist()
        self.report_ = PreprocessingReport(
            initial_descriptors=len(self.initial_columns_),
            final_descriptors=len(self.final_columns_),
            missing_strategy=self.config.missing_strategy,
            dropped_missing_columns=self.dropped_missing_columns_,
            imputed_columns=self.imputed_columns_,
            constant_columns=self.constant_columns_,
            low_variance_columns=self.low_variance_columns_,
            correlated_columns=self.correlated_columns_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        X_work = self._coerce_frame(X)
        X_work = X_work.reindex(columns=self.initial_columns_)

        if self.config.missing_strategy == "drop_columns":
            X_work = X_work.drop(columns=self.dropped_missing_columns_, errors="ignore")
        elif self.config.missing_strategy in {"mean_impute", "median_impute"}:
            X_work = X_work.fillna(self.impute_values_)

        X_work = X_work.drop(columns=self.constant_columns_, errors="ignore")
        X_work = X_work.drop(columns=self.low_variance_columns_, errors="ignore")
        X_work = X_work.drop(columns=self.correlated_columns_, errors="ignore")
        X_work = X_work.reindex(columns=self.final_columns_)
        return X_work

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_report(self) -> PreprocessingReport:
        self._check_is_fitted()
        return self.report_

    @staticmethod
    def _coerce_frame(X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        frame.columns = frame.columns.astype(str)
        return frame.apply(pd.to_numeric, errors="coerce")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "final_columns_"):
            raise RuntimeError("DescriptorPreprocessor has not been fitted yet.")


def drop_missing_rows(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, int]]:
    """Drop rows with descriptor missing values after the split.

    This option is provided because QSAR datasets are sometimes manually curated.
    It is deliberately applied after splitting so that train/test membership is
    not influenced by test-set descriptor distributions.
    """

    train_mask = ~X_train.isna().any(axis=1)
    test_mask = ~X_test.isna().any(axis=1)
    report = {
        "train_rows_removed": int((~train_mask).sum()),
        "test_rows_removed": int((~test_mask).sum()),
    }
    return (
        X_train.loc[train_mask],
        y_train.loc[train_mask],
        X_test.loc[test_mask],
        y_test.loc[test_mask],
        report,
    )

