"""Model factories and parameter helpers."""

from __future__ import annotations

from typing import Any

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR


SCALERS = {
    "None": None,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


MODEL_NAMES = [
    "MLR / Linear Regression",
    "PCR / Principal Component Regression",
    "PLS / Partial Least Squares",
    "SVR / Support Vector Regression",
    "RF / Random Forest",
]


def make_scaler(name: str):
    scaler_cls = SCALERS.get(name)
    return scaler_cls() if scaler_cls else None


def build_regressor(model_name: str, params: dict[str, Any]):
    if model_name == "MLR / Linear Regression":
        return LinearRegression(fit_intercept=bool(params.get("fit_intercept", True)))
    if model_name == "PLS / Partial Least Squares":
        return PLSRegression(
            n_components=int(params.get("n_components", 2)),
            scale=bool(params.get("scale", False)),
        )
    if model_name == "SVR / Support Vector Regression":
        return SVR(
            kernel=params.get("kernel", "rbf"),
            C=float(params.get("C", 10.0)),
            epsilon=float(params.get("epsilon", 0.1)),
            gamma=params.get("gamma", "scale"),
            degree=int(params.get("degree", 3)),
        )
    if model_name == "RF / Random Forest":
        max_depth = params.get("max_depth")
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=None if max_depth in {None, 0, "None"} else int(max_depth),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(model_name: str, params: dict[str, Any], scaler_name: str = "StandardScaler") -> Pipeline:
    """Build an sklearn pipeline.

    Scaling is part of the model pipeline so that cross-validation fits it only
    on each training fold and applies it to the corresponding validation fold.
    """

    steps = []
    scaler = make_scaler(scaler_name)
    if scaler is not None:
        steps.append(("scaler", scaler))

    if model_name == "PCR / Principal Component Regression":
        steps.append(("pca", PCA(n_components=int(params.get("n_components", 2)))))
        steps.append(("regressor", LinearRegression(fit_intercept=bool(params.get("fit_intercept", True)))))
    else:
        steps.append(("regressor", build_regressor(model_name, params)))
    return Pipeline(steps)


def flatten_prediction(prediction):
    return prediction.ravel()


def estimator_parameters(model_name: str, params: dict[str, Any], scaler_name: str) -> dict[str, Any]:
    result = {"model": model_name, "scaler": scaler_name}
    result.update(params)
    return result

