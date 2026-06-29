from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.applicability_domain import distance_domain_results, williams_results
from modules.evaluation import evaluate_fitted_model
from modules.feature_selection import FeatureSelector
from modules.statistical_tests import compare_endpoint_groups, endpoint_outlier_table
from modules.data_loader import endpoint_transform_preview, prepare_xy
from modules.model_io import ModelBundle, ModelRunBundle, bundle_from_bytes, bundle_to_bytes, predict_with_bundle, predict_with_run_bundle
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
    stat_table, grouping = compare_endpoint_groups(
        pca_screen.scores,
        y,
        ["PC1", "PC2"],
        grouping_method="Lower vs upper quartile",
        test_name="Welch t-test",
    )
    assert stat_table.shape[0] == 2
    assert grouping.low_label == "low_endpoint"
    endpoint_flags = endpoint_outlier_table(y)
    assert {"sample_id", "endpoint", "flagged"}.issubset(endpoint_flags.columns)

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

    for method, params in [
        ("None", {}),
        ("Manual", {"manual_descriptors": X_train.columns[:3].tolist()}),
        ("Variance threshold", {"threshold": 0.0}),
        ("SelectKBest", {"k": min(5, X_train.shape[1])}),
        ("RFE", {"n_features": min(3, X_train.shape[1])}),
    ]:
        smoke_selector = FeatureSelector(method, params)
        selected_train = smoke_selector.fit_transform(X_train, split.y_train)
        selected_test = smoke_selector.transform(X_test)
        assert selected_train.shape[1] >= 1
        assert selected_test.shape[1] == selected_train.shape[1]

    ga_selector = FeatureSelector(
        "Genetic Algorithm",
        {
            "population_size": 6,
            "generations": 2,
            "min_descriptors": 1,
            "max_descriptors": min(4, X_train.shape[1]),
            "cv_folds": 0,
            "random_seed": 42,
        },
    )
    X_ga = ga_selector.fit_transform(
        X_train,
        split.y_train,
        estimator_factory=lambda n=None: build_pipeline("MLR / Linear Regression", {"fit_intercept": True}, "StandardScaler"),
    )
    assert X_ga.shape[1] >= 1
    assert set(ga_selector.ga_history_["validation_mode"]) == {"Training score"}

    selector = FeatureSelector("SelectKBest", {"k": min(5, X_train.shape[1])})
    X_train_selected = selector.fit_transform(X_train, split.y_train)
    X_test_selected = selector.transform(X_test)

    estimator = build_pipeline("MLR / Linear Regression", {"fit_intercept": True}, "StandardScaler")
    estimator.fit(X_train_selected, split.y_train)
    evaluation = evaluate_fitted_model(
        estimator,
        X_train_selected,
        split.y_train,
        X_test_selected,
        split.y_test,
        cv_folds=5,
        cv_repeats=2,
    )
    assert evaluation.cv_scores.shape[0] == 10

    ada = build_pipeline(
        "AdaBoost / Adaptive Boosting",
        {"n_estimators": 8, "learning_rate": 0.05, "max_depth": 2, "random_state": 42},
        "None",
    )
    ada.fit(X_train_selected, split.y_train)
    assert ada.predict(X_test_selected).shape[0] == len(split.y_test)
    assert hasattr(ada.named_steps["regressor"], "feature_importances_")

    gbr = build_pipeline(
        "GBR / Gradient Boosting",
        {"n_estimators": 8, "learning_rate": 0.05, "max_depth": 2, "random_state": 42},
        "None",
    )
    gbr.fit(X_train_selected, split.y_train)
    assert gbr.predict(X_test_selected).shape[0] == len(split.y_test)
    assert hasattr(gbr.named_steps["regressor"], "feature_importances_")

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
        result_payload={"model_name": "MLR / Linear Regression"},
        results_table=pd.DataFrame({"Model label": ["smoke"]}),
        session_state={"excluded_sample_ids": ["cmpd_001"]},
    )
    loaded = bundle_from_bytes(bundle_to_bytes(bundle))
    assert loaded.result_payload["model_name"] == "MLR / Linear Regression"
    assert loaded.session_state["excluded_sample_ids"] == ["cmpd_001"]
    predictions, ad = predict_with_bundle(loaded, X.head(5))
    assert len(predictions) == 5
    assert len(ad) == 5

    run_bundle = ModelRunBundle(
        run_label="smoke run",
        bundles={"smoke": bundle, "smoke copy": bundle},
        results_table=pd.DataFrame({"Model label": ["smoke", "smoke copy"]}),
        training_results={"smoke": {"model_name": "MLR / Linear Regression"}},
        session_state={"excluded_sample_ids": ["cmpd_001"]},
    )
    loaded_run = bundle_from_bytes(bundle_to_bytes(run_bundle))
    assert isinstance(loaded_run, ModelRunBundle)
    assert loaded_run.training_results["smoke"]["model_name"] == "MLR / Linear Regression"
    assert loaded_run.session_state["excluded_sample_ids"] == ["cmpd_001"]
    run_predictions, run_ad, run_errors = predict_with_run_bundle(loaded_run, X.head(5))
    assert len(run_predictions) == 10
    assert len(run_ad) == 10
    assert run_errors.empty
    print("Smoke test passed")


if __name__ == "__main__":
    main()
