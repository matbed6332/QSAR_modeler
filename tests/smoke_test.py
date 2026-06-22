from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.applicability_domain import distance_domain_results, williams_results
from modules.evaluation import evaluate_fitted_model
from modules.feature_selection import FeatureSelector
from modules.data_loader import endpoint_transform_preview, prepare_xy
from modules.model_io import ModelBundle, bundle_from_bytes, bundle_to_bytes, predict_with_bundle
from modules.models import build_pipeline
from modules.pca_screening import compute_pca_screening
from modules.preprocessing import DescriptorPreprocessor, EndpointTransformer, PreprocessingConfig
from modules.splitting import random_split


def main() -> None:
    rng = np.random.default_rng(42)
    n_samples = 80
    X = pd.DataFrame(
        rng.normal(size=(n_samples, 12)),
        columns=[f"d{i}" for i in range(12)],
        index=[f"cmpd_{i:03d}" for i in range(n_samples)],
    )
    X["constant"] = 1.0
    X["corr_d0"] = X["d0"] * 0.99 + rng.normal(scale=0.001, size=n_samples)
    X.iloc[0, 1] = np.nan
    y = pd.Series(
        1.8 * X["d0"].fillna(X["d0"].median()) - 0.9 * X["d3"] + rng.normal(scale=0.2, size=n_samples),
        index=X.index,
        name="activity",
    )
    preview_y, preview_warnings = endpoint_transform_preview(y, "none")
    assert preview_y.equals(y.astype(float))
    assert preview_warnings == []
    pca_screen = compute_pca_screening(X, y, n_components=4)
    assert {"sample_id", "endpoint", "PC1", "PC2"}.issubset(pca_screen.scores.columns)
    assert len(pca_screen.variance) == 4

    x_sheet = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "d1": [1.0, 2.0, 3.0, 4.0],
            "smiles": ["CCO", "c1ccccc1", "CC(=O)O", "CCN"],
            "endpoint": [10.0, 20.0, 30.0, 40.0],
            "PCR": [0.1, 0.2, 0.3, 0.4],
            "leaky_copy": [10.0, 20.0, 30.0, 40.0],
        }
    )
    y_sheet = pd.DataFrame({"id": ["a", "b", "c", "d"], "endpoint": [10.0, 20.0, 30.0, 40.0]})
    loaded = prepare_xy(
        x_sheet,
        y_sheet,
        "endpoint",
        use_first_column_as_index=True,
        smiles_sheet=x_sheet,
        smiles_column="smiles",
    )
    assert loaded.X.columns.tolist() == ["d1"]
    assert loaded.endpoint_name_matches == ["endpoint"]
    assert loaded.endpoint_value_matches == ["leaky_copy"]
    assert loaded.artifact_name_matches == ["PCR"]
    assert loaded.smiles.loc["a"] == "CCO"
    assert loaded.smiles_column == "smiles"

    endpoint = EndpointTransformer("none").transform(y)
    split = random_split(X, endpoint, test_size=0.25, random_state=7, stratify_bins=4)
    preprocessor = DescriptorPreprocessor(
        PreprocessingConfig(
            missing_strategy="median_impute",
            remove_constant=True,
            remove_low_variance=True,
            variance_threshold=0.0,
            remove_high_correlation=True,
            correlation_threshold=0.95,
        )
    )
    X_train = preprocessor.fit_transform(split.X_train, split.y_train)
    X_test = preprocessor.transform(split.X_test)

    selector = FeatureSelector("SelectKBest", {"k": min(5, X_train.shape[1])})
    X_train_selected = selector.fit_transform(X_train, split.y_train)
    X_test_selected = selector.transform(X_test)

    estimator = build_pipeline("MLR / Linear Regression", {"fit_intercept": True}, "StandardScaler")
    estimator.fit(X_train_selected, split.y_train)
    evaluation = evaluate_fitted_model(estimator, X_train_selected, split.y_train, X_test_selected, split.y_test, cv_folds=5)

    w_ad = williams_results(X_train_selected, X_test_selected, evaluation.train_predictions, evaluation.test_predictions)
    d_ad = distance_domain_results(X_train_selected, X_test_selected)
    assert not w_ad.empty
    assert not d_ad.empty
    assert evaluation.metrics["Q2 CV"] > 0.7

    bundle = ModelBundle(
        model_label="smoke",
        model_name="MLR / Linear Regression",
        estimator=estimator,
        preprocessor=preprocessor,
        feature_selector=selector,
        endpoint_transformer=EndpointTransformer("none"),
        selected_descriptors=selector.selected_descriptors_,
        train_reference_X=X_train_selected,
        statistics=evaluation.metrics,
    )
    loaded = bundle_from_bytes(bundle_to_bytes(bundle))
    predictions, ad = predict_with_bundle(loaded, X.head(5))
    assert len(predictions) == 5
    assert len(ad) == 5
    print("Smoke test passed")


if __name__ == "__main__":
    main()
