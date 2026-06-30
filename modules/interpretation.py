"""Model interpretation helpers for descriptor importance and equations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_1d(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array.reshape(-1)


def _pipeline_scaler(estimator):
    return getattr(estimator, "named_steps", {}).get("scaler")


def _pipeline_regressor(estimator):
    return getattr(estimator, "named_steps", {}).get("regressor", estimator)


def _rescale_linear_coefficients(
    coefficients,
    intercept: float,
    scaler,
) -> tuple[np.ndarray, float, str]:
    """Convert linear coefficients from pipeline-scaled X back to descriptor units."""

    coef = _as_1d(coefficients)
    intercept_value = float(np.asarray(intercept, dtype=float).reshape(-1)[0])

    if scaler is None:
        return coef, intercept_value, "original descriptor scale"

    scaler_name = scaler.__class__.__name__
    if hasattr(scaler, "scale_") and hasattr(scaler, "min_"):
        scale = _as_1d(scaler.scale_)
        offset = _as_1d(scaler.min_)
        return coef * scale, intercept_value + float(np.dot(coef, offset)), f"original descriptor scale after {scaler_name}"

    if hasattr(scaler, "scale_"):
        scale = _as_1d(scaler.scale_)
        center = np.zeros_like(scale)
        if hasattr(scaler, "mean_") and scaler.mean_ is not None:
            center = _as_1d(scaler.mean_)
        elif hasattr(scaler, "center_") and scaler.center_ is not None:
            center = _as_1d(scaler.center_)

        safe_scale = np.where(np.isclose(scale, 0.0), 1.0, scale)
        original_coef = coef / safe_scale
        original_intercept = intercept_value - float(np.dot(original_coef, center))
        return original_coef, original_intercept, f"original descriptor scale after {scaler_name}"

    return coef, intercept_value, f"pipeline-scaled descriptor space ({scaler_name} could not be inverted)"


def _linear_regressor_coefficients(estimator, model_name: str) -> tuple[np.ndarray, float, str] | None:
    regressor = _pipeline_regressor(estimator)

    if hasattr(regressor, "feature_importances_"):
        return None

    if model_name == "PCR / Principal Component Regression":
        steps = getattr(estimator, "named_steps", {})
        pca = steps.get("pca")
        if pca is None or not hasattr(regressor, "coef_"):
            return None
        pc_coef = _as_1d(regressor.coef_)
        descriptor_coef = np.asarray(pca.components_, dtype=float).T @ pc_coef
        pca_center = _as_1d(getattr(pca, "mean_", np.zeros_like(descriptor_coef)))
        intercept = float(np.asarray(regressor.intercept_, dtype=float).reshape(-1)[0])
        intercept -= float(np.dot(pca_center, descriptor_coef))
        return descriptor_coef, intercept, "back-projected PCR coefficient"

    if hasattr(regressor, "coef_"):
        coefficient = _as_1d(regressor.coef_)
        intercept = float(np.asarray(getattr(regressor, "intercept_", 0.0), dtype=float).reshape(-1)[0])
        if model_name == "SVR / Support Vector Regression":
            kind = "linear SVR coefficient"
        elif model_name == "PLS / Partial Least Squares":
            kind = "PLS coefficient"
        else:
            kind = "linear coefficient"
        return coefficient, intercept, kind

    return None


def descriptor_importance_frame(estimator, descriptors: list[str], model_name: str) -> pd.DataFrame:
    """Return native descriptor importance where the fitted model exposes it."""

    descriptors = [str(descriptor) for descriptor in descriptors]
    regressor = _pipeline_regressor(estimator)

    if hasattr(regressor, "feature_importances_"):
        importances = _as_1d(regressor.feature_importances_)
        frame = pd.DataFrame(
            {
                "descriptor": descriptors,
                "importance": importances,
                "abs_importance": np.abs(importances),
                "sign": "",
                "interpretation": "tree ensemble feature_importances_",
            }
        )
        return frame.sort_values("abs_importance", ascending=False, ignore_index=True)

    linear = _linear_regressor_coefficients(estimator, model_name)
    if linear is None:
        return pd.DataFrame(columns=["descriptor", "coefficient", "importance", "abs_importance", "sign", "interpretation"])

    coefficients, intercept, interpretation = linear
    coefficients, intercept, scale_note = _rescale_linear_coefficients(coefficients, intercept, _pipeline_scaler(estimator))
    if len(coefficients) != len(descriptors):
        return pd.DataFrame(columns=["descriptor", "coefficient", "importance", "abs_importance", "sign", "interpretation"])

    frame = pd.DataFrame(
        {
            "descriptor": descriptors,
            "coefficient": coefficients,
            "importance": coefficients,
            "abs_importance": np.abs(coefficients),
            "sign": np.where(coefficients >= 0, "+", "-"),
            "interpretation": f"{interpretation}; {scale_note}",
        }
    )
    frame.attrs["intercept"] = intercept
    frame.attrs["scale_note"] = scale_note
    return frame.sort_values("abs_importance", ascending=False, ignore_index=True)


def mlr_equation(estimator, descriptors: list[str], precision: int = 6) -> tuple[str, pd.DataFrame]:
    """Build an MLR equation on the original descriptor scale when possible."""

    regressor = _pipeline_regressor(estimator)
    if regressor.__class__.__name__ != "LinearRegression":
        return "", pd.DataFrame()

    coef = _as_1d(regressor.coef_)
    intercept = float(np.asarray(regressor.intercept_, dtype=float).reshape(-1)[0])
    coef, intercept, scale_note = _rescale_linear_coefficients(coef, intercept, _pipeline_scaler(estimator))
    descriptors = [str(descriptor) for descriptor in descriptors]
    if len(coef) != len(descriptors):
        return "", pd.DataFrame()

    terms = [f"{intercept:.{precision}g}"]
    for descriptor, value in zip(descriptors, coef):
        sign = "+" if value >= 0 else "-"
        terms.append(f"{sign} {abs(value):.{precision}g}*{descriptor}")
    equation = "y_hat = " + " ".join(terms)
    frame = pd.DataFrame(
        {
            "term": ["intercept", *descriptors],
            "coefficient": [intercept, *coef.tolist()],
            "scale": [scale_note, *([scale_note] * len(descriptors))],
        }
    )
    return equation, frame
