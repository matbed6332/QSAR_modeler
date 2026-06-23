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


def bundle_to_bytes(bundle: ModelBundle) -> bytes:
    buffer = BytesIO()
    joblib.dump(bundle, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def bundle_from_bytes(data: bytes) -> ModelBundle:
    buffer = BytesIO(data)
    return joblib.load(buffer)


def save_bundle(bundle: ModelBundle, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> ModelBundle:
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

