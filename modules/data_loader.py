"""Excel loading and dataset alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd


@dataclass
class LoadedDataset:
    """Container returned after descriptor and endpoint sheets are aligned."""

    X: pd.DataFrame
    y: pd.Series
    endpoint_name: str
    sample_ids: pd.Index
    non_numeric_columns: list[str]
    partially_numeric_columns: list[str]
    warnings: list[str]
    endpoint_name_matches: list[str] = field(default_factory=list)
    endpoint_value_matches: list[str] = field(default_factory=list)
    artifact_name_matches: list[str] = field(default_factory=list)
    smiles: pd.Series | None = None
    smiles_column: str | None = None


def read_excel_sheets(uploaded_file: BinaryIO | BytesIO) -> dict[str, pd.DataFrame]:
    """Read every sheet from an uploaded Excel workbook."""

    return pd.read_excel(uploaded_file, sheet_name=None, engine="openpyxl")


def clean_sheet(df: pd.DataFrame, use_first_column_as_index: bool = False) -> pd.DataFrame:
    """Drop fully empty rows/columns and optionally use the first column as IDs."""

    cleaned = df.copy()
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    unnamed = [col for col in cleaned.columns if str(col).startswith("Unnamed:")]
    cleaned = cleaned.drop(columns=unnamed, errors="ignore")

    if use_first_column_as_index and len(cleaned.columns) > 0:
        first_col = cleaned.columns[0]
        cleaned = cleaned.set_index(first_col, drop=True)
        cleaned.index = cleaned.index.astype(str)

    return cleaned


def _numeric_descriptor_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    converted = X.apply(pd.to_numeric, errors="coerce")
    non_numeric = converted.columns[converted.notna().sum(axis=0) == 0].tolist()

    partially_numeric: list[str] = []
    for col in converted.columns:
        original_non_null = X[col].notna()
        introduced_missing = converted[col].isna() & original_non_null
        if introduced_missing.any() and col not in non_numeric:
            partially_numeric.append(str(col))

    converted = converted.drop(columns=non_numeric, errors="ignore")
    return converted, [str(col) for col in non_numeric], partially_numeric


def _normalize_column_name(name: object) -> str:
    return str(name).strip().casefold()


def _drop_endpoint_leakage_columns(
    X: pd.DataFrame,
    y: pd.Series,
    endpoint_column: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Remove columns that are clearly the endpoint, preventing target leakage."""

    endpoint_norm = _normalize_column_name(endpoint_column)
    endpoint_name_matches = [col for col in X.columns if _normalize_column_name(col) == endpoint_norm]
    X_checked = X.drop(columns=endpoint_name_matches, errors="ignore")

    endpoint_value_matches: list[str] = []
    y_values = y.astype(float).to_numpy()
    for col in X_checked.columns:
        x_values = pd.to_numeric(X_checked[col], errors="coerce").to_numpy()
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        if mask.any() and mask.sum() == np.isfinite(y_values).sum():
            if np.allclose(x_values[mask], y_values[mask], rtol=1e-10, atol=1e-12):
                endpoint_value_matches.append(col)

    X_checked = X_checked.drop(columns=endpoint_value_matches, errors="ignore")
    return X_checked, [str(col) for col in endpoint_name_matches], [str(col) for col in endpoint_value_matches]


def _drop_report_artifact_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    artifact_names = {
        "prediction",
        "predictions",
        "predicted",
        "observed",
        "residual",
        "residuals",
        "absolute_error",
        "split",
        "model",
        "mlr",
        "pcr",
        "pls",
        "svr",
        "svm",
        "rf",
        "r2",
        "q2",
        "rmse",
        "mae",
        "mse",
    }
    matches = [col for col in X.columns if _normalize_column_name(col) in artifact_names]
    return X.drop(columns=matches, errors="ignore"), [str(col) for col in matches]


def prepare_xy(
    x_sheet: pd.DataFrame,
    y_sheet: pd.DataFrame,
    endpoint_column: str,
    use_first_column_as_index: bool = False,
    smiles_sheet: pd.DataFrame | None = None,
    smiles_column: str | None = None,
) -> LoadedDataset:
    """Prepare X/y data and align rows by index.

    If the user chooses first-column IDs, alignment is done on those IDs.
    Otherwise the original Excel row order is used through the dataframe index.
    """

    warnings: list[str] = []
    X_raw = clean_sheet(x_sheet, use_first_column_as_index=use_first_column_as_index)
    y_raw = clean_sheet(y_sheet, use_first_column_as_index=use_first_column_as_index)

    if endpoint_column not in y_raw.columns:
        raise ValueError(f"Endpoint column '{endpoint_column}' was not found in the y sheet.")

    if X_raw.index.has_duplicates:
        raise ValueError("The descriptor sheet contains duplicated sample IDs/index values.")
    if y_raw.index.has_duplicates:
        raise ValueError("The endpoint sheet contains duplicated sample IDs/index values.")

    common_index = X_raw.index.intersection(y_raw.index)
    if common_index.empty:
        raise ValueError("No matching rows were found between descriptor and endpoint sheets.")
    if len(common_index) < len(X_raw.index) or len(common_index) < len(y_raw.index):
        warnings.append(
            f"Aligned {len(common_index)} shared rows; "
            f"{len(X_raw.index) - len(common_index)} X rows and "
            f"{len(y_raw.index) - len(common_index)} y rows were not matched."
        )

    X_aligned = X_raw.loc[common_index].copy()
    y_aligned = pd.to_numeric(y_raw.loc[common_index, endpoint_column], errors="coerce")
    y_aligned.name = str(endpoint_column)

    missing_y = y_aligned.isna()
    if missing_y.any():
        warnings.append(f"Removed {int(missing_y.sum())} rows with missing or non-numeric endpoint values.")
        X_aligned = X_aligned.loc[~missing_y]
        y_aligned = y_aligned.loc[~missing_y]

    smiles_aligned: pd.Series | None = None
    if smiles_sheet is not None and smiles_column:
        smiles_raw = clean_sheet(smiles_sheet, use_first_column_as_index=use_first_column_as_index)
        if smiles_column not in smiles_raw.columns:
            raise ValueError(f"SMILES column '{smiles_column}' was not found in the selected SMILES sheet.")
        smiles_aligned = smiles_raw.reindex(X_aligned.index)[smiles_column].astype("string")
        smiles_aligned = smiles_aligned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        smiles_aligned.name = str(smiles_column)
        missing_smiles = int(smiles_aligned.isna().sum())
        if missing_smiles:
            warnings.append(f"SMILES are missing for {missing_smiles} aligned sample(s). Structures will be unavailable for those samples.")

    X_numeric, non_numeric, partially_numeric = _numeric_descriptor_frame(X_aligned)
    X_numeric, endpoint_name_matches, endpoint_value_matches = _drop_endpoint_leakage_columns(
        X_numeric,
        y_aligned,
        endpoint_column,
    )
    X_numeric, artifact_name_matches = _drop_report_artifact_columns(X_numeric)
    if endpoint_name_matches:
        warnings.append(
            f"Removed {len(endpoint_name_matches)} descriptor column(s) matching the endpoint name: "
            f"{', '.join(endpoint_name_matches)}."
        )
    if endpoint_value_matches:
        warnings.append(
            f"Removed {len(endpoint_value_matches)} descriptor column(s) with values identical to the endpoint: "
            f"{', '.join(endpoint_value_matches)}."
        )
    if artifact_name_matches:
        warnings.append(
            f"Removed {len(artifact_name_matches)} report/model artifact column(s) from descriptors: "
            f"{', '.join(artifact_name_matches)}."
        )
    if non_numeric:
        warnings.append(f"Removed {len(non_numeric)} descriptor columns with no numeric values.")
    if partially_numeric:
        warnings.append(
            f"{len(partially_numeric)} descriptor columns had some non-numeric values converted to missing values."
        )
    if X_numeric.empty:
        raise ValueError("No numeric descriptor columns remain after cleaning.")

    return LoadedDataset(
        X=X_numeric,
        y=y_aligned.astype(float),
        endpoint_name=str(endpoint_column),
        sample_ids=X_numeric.index,
        non_numeric_columns=non_numeric,
        partially_numeric_columns=partially_numeric,
        warnings=warnings,
        endpoint_name_matches=endpoint_name_matches,
        endpoint_value_matches=endpoint_value_matches,
        artifact_name_matches=artifact_name_matches,
        smiles=smiles_aligned.reindex(X_numeric.index) if smiles_aligned is not None else None,
        smiles_column=str(smiles_column) if smiles_column else None,
    )


def dataset_summary(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Return compact statistics used by the dashboard."""

    descriptor_stats = X.describe().T
    if not descriptor_stats.empty:
        descriptor_stats = descriptor_stats[["mean", "std", "min", "max"]]

    endpoint_stats = y.describe().to_dict()
    endpoint_stats = {key: float(value) for key, value in endpoint_stats.items()}

    return {
        "samples": int(X.shape[0]),
        "descriptors": int(X.shape[1]),
        "missing_values": int(X.isna().sum().sum()),
        "missing_rows": int(X.isna().any(axis=1).sum()),
        "missing_columns": int(X.isna().any(axis=0).sum()),
        "endpoint_stats": endpoint_stats,
        "descriptor_stats": descriptor_stats,
    }


def endpoint_transform_preview(y: pd.Series, method: str) -> tuple[pd.Series, list[str]]:
    """Preview an endpoint transformation without mutating application state."""

    warnings: list[str] = []
    transformed = y.astype(float).copy()
    if method == "log10":
        if (transformed <= 0).any():
            warnings.append("log10 requires all endpoint values to be greater than zero.")
            return transformed, warnings
        transformed = np.log10(transformed)
    elif method == "negative_log10":
        if (transformed <= 0).any():
            warnings.append("-log10 requires all endpoint values to be greater than zero.")
            return transformed, warnings
        transformed = -np.log10(transformed)
    return pd.Series(transformed, index=y.index, name=y.name), warnings
