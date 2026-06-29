"""Model persistence and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import joblib
import pandas as pd

from modules.applicability_domain import distance_domain_results
from modules.models import flatten_prediction


@dataclass
class ModelBundle:
    model_label: str
    model_name: str
    estimator: Any
    preprocessor: Any
    feature_selector: Any
    endpoint_transformer: Any
    selected_descriptors: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    train_reference_X: pd.DataFrame | None = None
    statistics: dict[str, Any] = field(default_factory=dict)
    result_payload: dict[str, Any] = field(default_factory=dict)
    results_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    session_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRunBundle:
    """A persisted package containing every kept model from one training run."""

    run_label: str
    bundles: dict[str, ModelBundle]
    results_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)
    training_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_state: dict[str, Any] = field(default_factory=dict)


def bundle_to_bytes(bundle: Any) -> bytes:
    buffer = BytesIO()
    joblib.dump(bundle, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def bundle_from_bytes(data: bytes) -> ModelBundle | ModelRunBundle:
    buffer = BytesIO(data)
    return joblib.load(buffer)


def save_bundle(bundle: ModelBundle | ModelRunBundle, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> ModelBundle | ModelRunBundle:
    return joblib.load(path)


def predict_with_bundle(bundle: ModelBundle, X_new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply saved preprocessing, selection, model pipeline, and AD check."""

    X_numeric = pd.DataFrame(X_new).copy()
    X_numeric.columns = X_numeric.columns.astype(str)
    X_numeric = X_numeric.apply(pd.to_numeric, errors="coerce")
    X_preprocessed = bundle.preprocessor.transform(X_numeric)
    X_selected = bundle.feature_selector.transform(X_preprocessed)
    predictions = flatten_prediction(bundle.estimator.predict(X_selected))
    prediction_table = pd.DataFrame(
        {
            "sample_id": X_selected.index.astype(str),
            "prediction": predictions,
        },
        index=X_selected.index,
    )

    ad_table = pd.DataFrame()
    if bundle.train_reference_X is not None and not bundle.train_reference_X.empty:
        ad_table = distance_domain_results(bundle.train_reference_X, X_selected)
        ad_table = ad_table[ad_table["split"] == "test"].copy()
        ad_table["sample_id"] = X_selected.index.astype(str)
    return prediction_table, ad_table


def predict_with_run_bundle(
    run_bundle: ModelRunBundle,
    X_new: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Predict new compounds with every model in a saved run package."""

    prediction_frames: list[pd.DataFrame] = []
    ad_frames: list[pd.DataFrame] = []
    error_rows: list[dict[str, str]] = []

    for label, bundle in run_bundle.bundles.items():
        try:
            predictions, ad_table = predict_with_bundle(bundle, X_new)
        except Exception as exc:
            error_rows.append({"model_label": label, "error": str(exc)})
            continue

        predictions = predictions.copy()
        predictions.insert(0, "model_label", label)
        predictions.insert(1, "model_name", bundle.model_name)
        prediction_frames.append(predictions)

        if not ad_table.empty:
            ad_table = ad_table.copy()
            ad_table.insert(0, "model_label", label)
            ad_table.insert(1, "model_name", bundle.model_name)
            ad_frames.append(ad_table)

    prediction_table = pd.concat(prediction_frames, axis=0, ignore_index=True) if prediction_frames else pd.DataFrame()
    ad_table = pd.concat(ad_frames, axis=0, ignore_index=True) if ad_frames else pd.DataFrame()
    errors = pd.DataFrame(error_rows)
    return prediction_table, ad_table, errors

