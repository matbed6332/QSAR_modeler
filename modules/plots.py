"""Matplotlib plotting functions and image export helpers."""

from __future__ import annotations

from io import BytesIO
import os
import tempfile

import matplotlib

_mpl_config_dir = os.path.join(tempfile.gettempdir(), "qsar_qspr_mplconfig")
os.makedirs(_mpl_config_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_config_dir)

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


apply_publication_style()


TRAIN_COLOR = "#1f77b4"
TEST_COLOR = "#d62728"
ACCENT = "#2ca02c"
MUTED = "#6c757d"


def endpoint_histogram(y: pd.Series, bins: int = 20, title: str = "Endpoint distribution"):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(y.dropna(), bins=bins, color="#4c78a8", edgecolor="white", alpha=0.88)
    ax.set_title(title)
    ax.set_xlabel(y.name or "Endpoint")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def observed_vs_predicted(train_predictions: pd.DataFrame, test_predictions: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    if not train_predictions.empty:
        ax.scatter(train_predictions["observed"], train_predictions["predicted"], label="Train", color=TRAIN_COLOR, alpha=0.8)
    if not test_predictions.empty:
        ax.scatter(test_predictions["observed"], test_predictions["predicted"], label="Test", color=TEST_COLOR, alpha=0.85)
    all_values = pd.concat(
        [
            train_predictions[["observed", "predicted"]],
            test_predictions[["observed", "predicted"]],
        ],
        axis=0,
    ).to_numpy(dtype=float)
    finite = all_values[np.isfinite(all_values)]
    if finite.size:
        low, high = finite.min(), finite.max()
        padding = (high - low) * 0.05 if not np.isclose(high, low) else 1.0
        ax.plot([low - padding, high + padding], [low - padding, high + padding], "--", color="#222222", linewidth=1.2, label="Ideal")
        ax.set_xlim(low - padding, high + padding)
        ax.set_ylim(low - padding, high + padding)
    ax.set_title(title)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.legend()
    fig.tight_layout()
    return fig


def residual_plot(train_predictions: pd.DataFrame, test_predictions: pd.DataFrame, title: str = "Residual plot"):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if not train_predictions.empty:
        ax.scatter(train_predictions["predicted"], train_predictions["residual"], label="Train", color=TRAIN_COLOR, alpha=0.8)
    if not test_predictions.empty:
        ax.scatter(test_predictions["predicted"], test_predictions["residual"], label="Test", color=TEST_COLOR, alpha=0.85)
    ax.axhline(0, color="#222222", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.legend()
    fig.tight_layout()
    return fig


def residual_histogram(train_predictions: pd.DataFrame, test_predictions: pd.DataFrame, bins: int = 20):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    residuals = pd.concat([train_predictions["residual"], test_predictions["residual"]], axis=0)
    ax.hist(residuals.dropna(), bins=bins, color="#59a14f", edgecolor="white", alpha=0.86)
    ax.axvline(0, color="#222222", linewidth=1.1)
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def cv_score_plot(cv_scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(cv_scores["fold"], cv_scores["R2"], marker="o", color="#4c78a8", label="R2")
    ax.axhline(cv_scores["R2"].mean(), color="#222222", linestyle="--", linewidth=1.0, label="Mean R2")
    ax.set_title("Cross-validation R2 by fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("R2")
    ax.legend()
    fig.tight_layout()
    return fig


def ga_progress_plot(history: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if not history.empty:
        ax.plot(history["generation"], history["best_score"], color="#4c78a8", marker="o", label="Best")
        ax.plot(history["generation"], history["mean_score"], color="#f28e2b", marker="s", label="Mean")
    ax.set_title("Genetic algorithm score progression")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Optimization score")
    ax.legend()
    fig.tight_layout()
    return fig


def williams_plot(ad_results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    if ad_results.empty:
        return fig
    for split, color in [("train", TRAIN_COLOR), ("test", TEST_COLOR)]:
        data = ad_results[ad_results["split"] == split]
        ax.scatter(data["leverage"], data["standardized_residual"], color=color, label=split.title(), alpha=0.82)
    h_warning = ad_results["h_warning"].iloc[0]
    ax.axvline(h_warning, color="#222222", linestyle="--", linewidth=1.1, label="h*")
    ax.axhline(3, color="#aa3a3a", linestyle="--", linewidth=1.0)
    ax.axhline(-3, color="#aa3a3a", linestyle="--", linewidth=1.0)
    ax.set_title("Williams plot")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residual")
    ax.legend()
    fig.tight_layout()
    return fig


def distance_domain_plot(distance_results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if distance_results.empty:
        return fig
    for split, color in [("train", TRAIN_COLOR), ("test", TEST_COLOR)]:
        data = distance_results[distance_results["split"] == split].reset_index(drop=True)
        ax.scatter(data.index, data["distance_to_train_centroid"], color=color, label=split.title(), alpha=0.82)
    threshold = distance_results["distance_threshold"].iloc[0]
    ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.1, label="AD threshold")
    ax.set_title("Distance-based applicability domain")
    ax.set_xlabel("Sample order")
    ax.set_ylabel("Distance to training centroid")
    ax.legend()
    fig.tight_layout()
    return fig


def pca_score_plot(pca_scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    if pca_scores.empty:
        return fig
    for split, color in [("train", TRAIN_COLOR), ("test", TEST_COLOR)]:
        data = pca_scores[pca_scores["split"] == split]
        ax.scatter(data["PC1"], data["PC2"], color=color, label=split.title(), alpha=0.82)
    evr = pca_scores.attrs.get("explained_variance_ratio", [np.nan, np.nan])
    ax.set_title("PCA applicability-domain view")
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)" if np.isfinite(evr[0]) else "PC1")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)" if len(evr) > 1 and np.isfinite(evr[1]) else "PC2")
    ax.legend()
    fig.tight_layout()
    return fig


def pca_explained_variance_plot(estimator):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if "pca" not in estimator.named_steps:
        ax.text(0.5, 0.5, "PCA is not part of this model.", ha="center", va="center")
        return fig
    ratios = estimator.named_steps["pca"].explained_variance_ratio_
    ax.bar(np.arange(1, len(ratios) + 1), ratios * 100, color="#4c78a8")
    ax.plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios) * 100, color="#f28e2b", marker="o", label="Cumulative")
    ax.set_title("PCR explained variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.legend()
    fig.tight_layout()
    return fig


def descriptor_importance_plot(importance: pd.DataFrame, top_n: int = 20, title: str = "Descriptor importance"):
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    if importance.empty or "descriptor" not in importance.columns:
        ax.text(0.5, 0.5, "Native descriptor importance is not available for this model.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    data = importance.copy()
    if "abs_importance" not in data.columns:
        data["abs_importance"] = data.get("importance", 0.0).abs()
    data = data.sort_values("abs_importance", ascending=False).head(top_n)

    if "coefficient" in data.columns and data["coefficient"].notna().any():
        values = data["coefficient"].astype(float)
        colors = np.where(values >= 0, "#4c78a8", "#d62728")
        ax.barh(data["descriptor"].iloc[::-1], values.iloc[::-1], color=colors[::-1], alpha=0.9)
        ax.axvline(0, color="#222222", linewidth=1.0)
        ax.set_xlabel("Coefficient")
    else:
        values = data["abs_importance"].astype(float)
        ax.barh(data["descriptor"].iloc[::-1], values.iloc[::-1], color="#59a14f", alpha=0.9)
        ax.set_xlabel("Importance")

    ax.set_title(title)
    fig.tight_layout()
    return fig


def model_comparison_plot(results_table: pd.DataFrame, metric: str):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if results_table.empty or metric not in results_table.columns:
        return fig
    data = results_table[["Model label", metric]].dropna()
    ax.barh(data["Model label"], data[metric], color="#4c78a8")
    ax.set_title(f"Model comparison: {metric}")
    ax.set_xlabel(metric)
    fig.tight_layout()
    return fig


def correlation_heatmap(X: pd.DataFrame, max_features: int = 40):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    if X.empty:
        return fig
    subset = X.iloc[:, :max_features]
    corr = subset.corr().fillna(0.0)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Descriptor correlation heatmap")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_to_bytes(fig, fmt: str = "png") -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format=fmt, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()
