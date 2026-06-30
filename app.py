from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.applicability_domain import distance_domain_results, pca_domain_scores, williams_results
from modules.chemistry import smiles_to_png_bytes
from modules.data_loader import clean_sheet, dataset_summary, endpoint_transform_preview, prepare_xy, read_excel_sheets
from modules.evaluation import evaluate_fitted_model, rank_models, results_table
from modules.export import dataframe_to_csv_bytes, dataframes_to_excel_bytes, figures_to_zip_bytes, list_to_frame, safe_file_stem
from modules.feature_selection import FeatureSelector
from modules.interpretation import descriptor_importance_frame, mlr_equation
from modules.model_io import ModelBundle, ModelRunBundle, bundle_from_bytes, bundle_to_bytes, predict_with_bundle, predict_with_run_bundle
from modules.models import MODEL_NAMES, build_pipeline, estimator_parameters
from modules.pca_screening import compute_pca_screening
from modules.plots import (
    correlation_heatmap,
    cv_score_plot,
    descriptor_importance_plot,
    distance_domain_plot,
    endpoint_histogram,
    fig_to_bytes,
    ga_progress_plot,
    model_comparison_plot,
    observed_vs_predicted,
    pca_explained_variance_plot,
    pca_score_plot,
    residual_histogram,
    residual_plot,
    williams_plot,
)
from modules.preprocessing import (
    DescriptorPreprocessor,
    EndpointTransformer,
    PreprocessingConfig,
    drop_missing_rows,
)
from modules.splitting import random_split, sorted_endpoint_split, split_range_table
from modules.statistical_tests import compare_endpoint_groups, endpoint_outlier_table


st.set_page_config(
    page_title="QSAR/QSPR Modeling Studio",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


FEATURE_IMPORTANCE_MODELS = {
    "RF / Random Forest",
    "AdaBoost / Adaptive Boosting",
    "GBR / Gradient Boosting",
}


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.75rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .main-title {
        font-size: 2.1rem;
        line-height: 1.22;
        font-weight: 760;
        color: #18212f;
        margin-top: 0.35rem;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #51606f;
        font-size: 1.02rem;
        max-width: 1020px;
        margin-bottom: 1.0rem;
    }
    .metric-panel {
        border: 1px solid #dce3ea;
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        background: #fbfcfe;
    }
    .metric-label {
        color: #627083;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0;
    }
    .metric-value {
        color: #18212f;
        font-size: 1.4rem;
        font-weight: 720;
        margin-top: 0.1rem;
    }
    .status-good {
        color: #18643b;
        font-weight: 650;
    }
    .status-warn {
        color: #965400;
        font-weight: 650;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Reading Excel workbook...")
def cached_read_excel_sheets(data: bytes) -> dict[str, pd.DataFrame]:
    return read_excel_sheets(BytesIO(data))


@st.cache_data(show_spinner=False)
def cached_pca_screening(X: pd.DataFrame, y: pd.Series, n_components: int, scale: bool):
    return compute_pca_screening(X, y, n_components=n_components, scale=scale)


def init_state() -> None:
    defaults = {
        "sheets": None,
        "uploaded_name": None,
        "dataset": None,
        "split_preview": None,
        "training_results": {},
        "results_df": pd.DataFrame(),
        "last_run_warnings": [],
        "loaded_bundle": None,
        "loaded_bundle_key": None,
        "bundle_restore_messages": [],
        "prediction_output": None,
        "excluded_sample_ids": [],
        "outlier_log": pd.DataFrame(columns=["sample_id", "reason", "removed_at", "source"]),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def metric_panel(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_messages(messages: list[str], level: str = "warning") -> None:
    if not messages:
        return
    for message in messages:
        if level == "error":
            st.error(message)
        elif level == "info":
            st.info(message)
        else:
            st.warning(message)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "estimating"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def training_progress_text(
    action: str,
    completed_units: float,
    total_units: int,
    start_time: float,
    detail: str | None = None,
) -> str:
    total_units = max(1, int(total_units))
    completed_units = max(0.0, min(float(completed_units), float(total_units)))
    elapsed = time.perf_counter() - start_time
    fraction = completed_units / total_units
    eta = (elapsed / fraction) - elapsed if fraction > 0 else None
    percent = int(round(fraction * 100))
    parts = [f"{action} ({percent}%)", f"elapsed {format_duration(elapsed)}", f"ETA {format_duration(eta)}"]
    if detail:
        parts.append(detail)
    return " | ".join(parts)


def sync_widget(source_key: str, target_key: str) -> None:
    st.session_state[target_key] = st.session_state[source_key]


def synced_int_control(
    label: str,
    min_value: int,
    max_value: int,
    value: int,
    step: int,
    key: str,
) -> int:
    """Render a slider plus exact numeric input that stay synchronized."""

    slider_key = f"{key}_slider"
    input_key = f"{key}_input"
    default = int(max(min_value, min(value, max_value)))
    if max_value <= min_value:
        st.session_state[input_key] = int(min_value)
        st.number_input(label, min_value=min_value, max_value=max_value, value=min_value, step=step, disabled=True, key=input_key)
        return int(min_value)
    st.session_state.setdefault(slider_key, default)
    st.session_state.setdefault(input_key, default)
    st.session_state[slider_key] = int(max(min_value, min(st.session_state[slider_key], max_value)))
    st.session_state[input_key] = int(max(min_value, min(st.session_state[input_key], max_value)))
    left, right = st.columns([1.7, 0.8])
    with left:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=slider_key,
            on_change=sync_widget,
            args=(slider_key, input_key),
        )
    with right:
        st.number_input(
            "Exact",
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=input_key,
            on_change=sync_widget,
            args=(input_key, slider_key),
            label_visibility="visible",
        )
    return int(st.session_state[input_key])


def active_dataset():
    dataset = st.session_state.dataset
    if dataset is None:
        return None
    excluded = {str(sample_id) for sample_id in st.session_state.excluded_sample_ids}
    keep_index = [idx for idx in dataset.X.index if str(idx) not in excluded]
    smiles = dataset.smiles.reindex(keep_index) if getattr(dataset, "smiles", None) is not None else None
    return replace(
        dataset,
        X=dataset.X.loc[keep_index].copy(),
        y=dataset.y.loc[keep_index].copy(),
        sample_ids=dataset.X.loc[keep_index].index,
        smiles=smiles,
    )


def excluded_samples_frame() -> pd.DataFrame:
    log = st.session_state.outlier_log.copy()
    if log.empty:
        return pd.DataFrame(columns=["sample_id", "reason", "removed_at", "source"])
    active_excluded = {str(sample_id) for sample_id in st.session_state.excluded_sample_ids}
    return log[log["sample_id"].astype(str).isin(active_excluded)].drop_duplicates("sample_id", keep="last")


def reset_modeling_outputs() -> None:
    st.session_state.split_preview = None
    st.session_state.training_results = {}
    st.session_state.results_df = pd.DataFrame()
    st.session_state.last_run_warnings = []


def sample_id_from_plotly_selection(event) -> str | None:
    try:
        points = event.selection.points
    except Exception:
        try:
            points = event.get("selection", {}).get("points", [])
        except Exception:
            return None
    if not points:
        return None
    point = points[0]
    try:
        customdata = point.get("customdata")
    except AttributeError:
        customdata = getattr(point, "customdata", None)
    if customdata is not None and len(customdata) > 0:
        return str(customdata[0])
    for key in ("hovertext", "text"):
        try:
            value = point.get(key)
        except AttributeError:
            value = getattr(point, key, None)
        if value:
            return str(value)
    return None


def render_structure_panel(
    smiles_series: pd.Series | None,
    key: str,
    selected_sample_id: str | None = None,
    sample_ids: list[str] | None = None,
) -> None:
    if smiles_series is None:
        st.info("No SMILES column was selected during data import.")
        return

    smiles_clean = smiles_series.dropna().astype(str)
    smiles_clean = smiles_clean[~smiles_clean.str.strip().str.lower().isin(["", "nan", "none", "<na>"])]
    if smiles_clean.empty:
        st.info("No valid SMILES are available for the current samples.")
        return

    if sample_ids:
        wanted = {str(sample_id) for sample_id in sample_ids}
        smiles_clean = smiles_clean[smiles_clean.index.astype(str).isin(wanted)]
    if smiles_clean.empty:
        st.info("No valid SMILES are available for the samples in this plot.")
        return

    ids = smiles_clean.index.astype(str).tolist()
    widget_key = f"{key}_structure_id"
    plot_key = f"{key}_last_plot_structure_id"
    if selected_sample_id in ids and selected_sample_id != st.session_state.get(plot_key):
        st.session_state[widget_key] = selected_sample_id
        st.session_state[plot_key] = selected_sample_id
        st.caption(f"Selected from plot: {selected_sample_id}")
    elif selected_sample_id in ids:
        st.caption(f"Last plot selection: {selected_sample_id}")
    else:
        st.session_state[plot_key] = None
    default_index = ids.index(st.session_state[widget_key]) if st.session_state.get(widget_key) in ids else 0
    selected_id = st.selectbox("Show structure for sample ID", ids, index=default_index, key=widget_key)
    smiles = str(smiles_clean.loc[smiles_clean.index.astype(str) == selected_id].iloc[0])
    st.code(smiles, language="text")
    image_bytes, error = smiles_to_png_bytes(smiles)
    if image_bytes:
        st.image(image_bytes, caption=f"Structure for {selected_id}", use_container_width=False)
    else:
        st.warning(error or "Structure rendering failed.")


def mapping_missing_strategy(label: str) -> tuple[str, bool]:
    mapping = {
        "Median imputation": ("median_impute", False),
        "Mean imputation": ("mean_impute", False),
        "Remove descriptor columns with missing values": ("drop_columns", False),
        "Remove rows with missing descriptors after split": ("none", True),
        "Require complete descriptor matrix": ("none", False),
    }
    return mapping[label]


def endpoint_label(endpoint_name: str, method: str) -> str:
    if method == "log10":
        return f"log10({endpoint_name})"
    if method == "negative_log10":
        return f"-log10({endpoint_name})"
    return endpoint_name


def prepared_pca_inputs(dataset) -> tuple[pd.DataFrame, pd.Series, str, DescriptorPreprocessor, list[str]]:
    """Apply current preprocessing settings for exploratory PCA screening."""

    endpoint_method = st.session_state.get("endpoint_method", "none")
    transformed_y = EndpointTransformer(endpoint_method).transform(dataset.y)
    transformed_y.name = endpoint_label(dataset.endpoint_name, endpoint_method)

    config = st.session_state.get("preprocessing_config", PreprocessingConfig())
    notes: list[str] = []
    pca_config = config
    if st.session_state.get("drop_rows_after_split", False) and config.missing_strategy == "none":
        pca_config = replace(config, missing_strategy="median_impute")
        notes.append("PCA screening used median imputation because row removal is only applied after train/test split.")

    preprocessor = DescriptorPreprocessor(pca_config)
    X_preprocessed = preprocessor.fit_transform(dataset.X, transformed_y)
    return X_preprocessed, transformed_y, transformed_y.name, preprocessor, notes


def safe_component_count(model_name: str, params: dict[str, Any], n_features: int, n_train: int, cv_folds: int) -> dict[str, Any]:
    safe = dict(params)
    if model_name in {"PCR / Principal Component Regression", "PLS / Partial Least Squares"}:
        fold_train_size = max(2, int(np.floor(n_train * (cv_folds - 1) / max(cv_folds, 2))))
        max_components = max(1, min(n_features, fold_train_size - 1))
        requested = int(safe.get("n_components", min(2, max_components)))
        safe["n_components"] = max(1, min(requested, max_components))
    return safe


def make_estimator_factory(model_name: str, raw_params: dict[str, Any], scaler_name: str, n_train: int, cv_folds: int):
    def factory(n_features: int | None = None):
        features = n_features if n_features is not None else 1
        params = safe_component_count(model_name, raw_params, features, n_train, cv_folds)
        return build_pipeline(model_name, params, scaler_name)

    return factory


def make_split(X: pd.DataFrame, y: pd.Series, split_config: dict[str, Any]):
    if split_config["method"] == "Random split":
        return random_split(
            X,
            y,
            test_size=float(split_config["test_size"]),
            random_state=int(split_config["random_state"]),
            stratify_bins=split_config.get("stratify_bins"),
        )
    return sorted_endpoint_split(
        X,
        y,
        train_fraction=float(split_config["train_fraction"]),
        strategy=split_config["sorted_strategy"],
        random_state=int(split_config["random_state"]),
    )


def apply_interactive_plot_style(fig):
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#000000"},
        title={"font": {"color": "#000000"}},
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "#000000",
            "font": {"color": "#000000"},
        },
        legend={"font": {"color": "#000000"}},
        coloraxis={
            "colorbar": {
                "tickfont": {"color": "#000000"},
                "title": {"font": {"color": "#000000"}},
                "tickcolor": "#000000",
            }
        },
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1.4,
        linecolor="#000000",
        mirror=True,
        gridcolor="#d0d7de",
        zerolinecolor="#000000",
        tickfont={"color": "#000000"},
        title_font={"color": "#000000"},
        tickcolor="#000000",
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1.4,
        linecolor="#000000",
        mirror=True,
        gridcolor="#d0d7de",
        zerolinecolor="#000000",
        tickfont={"color": "#000000"},
        title_font={"color": "#000000"},
        tickcolor="#000000",
    )
    return fig


def pca_score_figure(scores: pd.DataFrame, pc_x: str, pc_y: str, endpoint_label: str = "Endpoint"):
    custom_data = ["sample_id"]
    has_smiles = "smiles" in scores.columns
    if has_smiles:
        custom_data.append("smiles")
    fig = px.scatter(
        scores,
        x=pc_x,
        y=pc_y,
        color="endpoint",
        hover_name="sample_id",
        hover_data={"endpoint": ":.5g", pc_x: ":.4f", pc_y: ":.4f"},
        custom_data=custom_data,
        color_continuous_scale="Viridis",
        template="plotly_white",
        title="PCA score plot",
        height=560,
    )
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        + f"{pc_x}: %{{x:.4f}}<br>"
        + f"{pc_y}: %{{y:.4f}}<br>"
        + f"{endpoint_label}: %{{marker.color:.5g}}"
    )
    if has_smiles:
        hovertemplate += "<br>SMILES: %{customdata[1]}"
    hovertemplate += "<extra></extra>"
    fig.update_traces(
        marker={"size": 10, "line": {"width": 0.8, "color": "#1d2733"}},
        hovertemplate=hovertemplate,
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 56, "b": 20},
        coloraxis_colorbar={
            "title": {"text": endpoint_label, "font": {"color": "#000000"}},
            "tickfont": {"color": "#000000"},
            "tickcolor": "#000000",
        },
    )
    return apply_interactive_plot_style(fig)


def interactive_observed_vs_predicted(train_predictions: pd.DataFrame, test_predictions: pd.DataFrame, title: str):
    fig = go.Figure()
    for split_name, data, color in [
        ("Train", train_predictions, "#1f77b4"),
        ("Test", test_predictions, "#d62728"),
    ]:
        if data.empty:
            continue
        custom_columns = [
            data["sample_id"].astype(str),
            data["residual"].astype(float),
            data["absolute_error"].astype(float),
        ]
        has_smiles = "smiles" in data.columns
        if has_smiles:
            custom_columns.append(data["smiles"].fillna("").astype(str))
        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Observed: %{x:.5g}<br>"
            "Predicted: %{y:.5g}<br>"
            "Residual: %{customdata[1]:.5g}<br>"
            "Absolute error: %{customdata[2]:.5g}"
        )
        if has_smiles:
            hovertemplate += "<br>SMILES: %{customdata[3]}"
        hovertemplate += "<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=data["observed"],
                y=data["predicted"],
                mode="markers",
                name=split_name,
                marker={"size": 10, "color": color, "line": {"width": 0.8, "color": "#1d2733"}},
                customdata=np.stack(custom_columns, axis=-1),
                hovertemplate=hovertemplate,
            )
        )

    all_values = pd.concat(
        [train_predictions[["observed", "predicted"]], test_predictions[["observed", "predicted"]]],
        axis=0,
    ).to_numpy(dtype=float)
    finite = all_values[np.isfinite(all_values)]
    if finite.size:
        low, high = float(finite.min()), float(finite.max())
        padding = (high - low) * 0.05 if not np.isclose(high, low) else 1.0
        fig.add_trace(
            go.Scatter(
                x=[low - padding, high + padding],
                y=[low - padding, high + padding],
                mode="lines",
                name="Ideal",
                line={"color": "#222222", "dash": "dash", "width": 1.5},
                hoverinfo="skip",
            )
        )
        fig.update_xaxes(range=[low - padding, high + padding])
        fig.update_yaxes(range=[low - padding, high + padding])

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=560,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        xaxis_title="Observed",
        yaxis_title="Predicted",
    )
    return apply_interactive_plot_style(fig)


def interactive_williams_plot(ad_results: pd.DataFrame):
    fig = go.Figure()
    if ad_results.empty:
        return fig

    for split_name, color in [("train", "#1f77b4"), ("test", "#d62728")]:
        data = ad_results[ad_results["split"] == split_name].copy()
        if data.empty:
            continue
        custom_columns = [
            data["sample_id"].astype(str),
            data["outside_leverage"].astype(str),
            data["outside_residual"].astype(str),
            data["outside_ad"].astype(str),
        ]
        has_smiles = "smiles" in data.columns
        if has_smiles:
            custom_columns.append(data["smiles"].fillna("").astype(str))
        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Leverage: %{x:.5g}<br>"
            "Standardized residual: %{y:.5g}<br>"
            "Outside leverage: %{customdata[1]}<br>"
            "Outside residual: %{customdata[2]}<br>"
            "Outside AD: %{customdata[3]}"
        )
        if has_smiles:
            hovertemplate += "<br>SMILES: %{customdata[4]}"
        hovertemplate += "<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=data["leverage"],
                y=data["standardized_residual"],
                mode="markers",
                name=split_name.title(),
                marker={
                    "size": 10,
                    "color": color,
                    "symbol": np.where(data["outside_ad"], "x", "circle"),
                    "line": {"width": 0.8, "color": "#1d2733"},
                },
                customdata=np.stack(custom_columns, axis=-1),
                hovertemplate=hovertemplate,
            )
        )

    h_warning = float(ad_results["h_warning"].iloc[0])
    fig.add_vline(x=h_warning, line_dash="dash", line_color="#222222", annotation_text="h*")
    fig.add_hline(y=3, line_dash="dash", line_color="#aa3a3a", annotation_text="+3")
    fig.add_hline(y=-3, line_dash="dash", line_color="#aa3a3a", annotation_text="-3")
    fig.update_layout(
        title="Williams plot",
        template="plotly_white",
        height=560,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        xaxis_title="Leverage",
        yaxis_title="Standardized residual",
    )
    return apply_interactive_plot_style(fig)


def payload_descriptor_importance(payload: dict[str, Any]) -> pd.DataFrame:
    importance = payload.get("descriptor_importance")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        return importance.copy()
    try:
        return descriptor_importance_frame(
            payload["estimator"],
            payload.get("selected_descriptors", []),
            payload.get("model_name", ""),
        )
    except Exception:
        return pd.DataFrame()


def payload_mlr_equation(payload: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    equation = payload.get("model_equation", "")
    terms = payload.get("equation_terms", pd.DataFrame())
    if equation:
        return str(equation), terms if isinstance(terms, pd.DataFrame) else pd.DataFrame()
    if payload.get("model_name") != "MLR / Linear Regression":
        return "", pd.DataFrame()
    try:
        return mlr_equation(payload["estimator"], payload.get("selected_descriptors", []))
    except Exception:
        return "", pd.DataFrame()


def create_figures_for_result(label: str, payload: dict[str, Any]) -> dict[str, object]:
    evaluation = payload["evaluation"]
    figures = {
        "Observed vs predicted": observed_vs_predicted(
            evaluation.train_predictions,
            evaluation.test_predictions,
            f"{label}: observed vs predicted",
        ),
        "Residual plot": residual_plot(evaluation.train_predictions, evaluation.test_predictions),
        "Residual histogram": residual_histogram(evaluation.train_predictions, evaluation.test_predictions),
        "Cross-validation scores": cv_score_plot(evaluation.cv_scores),
        "Applicability Domain - Williams plot": williams_plot(payload["williams_ad"]),
        "Applicability Domain - Distance plot": distance_domain_plot(payload["distance_ad"]),
    }
    if not payload["pca_ad"].empty:
        figures["Applicability Domain - PCA plot"] = pca_score_plot(payload["pca_ad"])
    if payload["feature_selector"].method == "Genetic Algorithm" and not payload["feature_selector"].ga_history_.empty:
        figures["GA progress"] = ga_progress_plot(payload["feature_selector"].ga_history_)
    importance = payload_descriptor_importance(payload)
    if not importance.empty:
        figures["Descriptor importance"] = descriptor_importance_plot(
            importance,
            title=f"{label}: descriptor importance",
        )
    if payload["model_name"] == "PCR / Principal Component Regression":
        figures["PCR explained variance"] = pca_explained_variance_plot(payload["estimator"])
    return figures


def removed_descriptors_frame(preprocessor: DescriptorPreprocessor) -> pd.DataFrame:
    report = preprocessor.get_report()
    groups = {
        "missing_columns": report.dropped_missing_columns,
        "constant": report.constant_columns,
        "low_variance": report.low_variance_columns,
        "high_correlation": report.correlated_columns,
    }
    rows = [
        {"reason": reason, "descriptor": descriptor}
        for reason, descriptors in groups.items()
        for descriptor in descriptors
    ]
    return pd.DataFrame(rows, columns=["reason", "descriptor"])


def build_report_sheets(label: str, payload: dict[str, Any], all_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    evaluation = payload["evaluation"]
    params_frame = pd.DataFrame(
        [{"parameter": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in payload["parameters"].items()]
    )
    warnings_frame = pd.DataFrame({"warning": payload.get("warnings", [])})
    importance_frame = payload_descriptor_importance(payload).reset_index(drop=True)
    equation, equation_terms = payload_mlr_equation(payload)
    equation_frame = pd.DataFrame({"model_equation": [equation]}) if equation else pd.DataFrame(columns=["model_equation"])
    if isinstance(equation_terms, pd.DataFrame) and not equation_terms.empty:
        equation_frame = pd.concat([equation_frame, equation_terms.reset_index(drop=True)], axis=1)
    return {
        "Summary": all_results,
        "Preprocessing": payload["preprocessor"].get_report().to_frame(),
        "Removed descriptors": removed_descriptors_frame(payload["preprocessor"]),
        "Selected descriptors": list_to_frame(payload["selected_descriptors"]),
        "Descriptor importance": importance_frame,
        "Model equation": equation_frame,
        "Train predictions": evaluation.train_predictions.reset_index(drop=True),
        "Test predictions": evaluation.test_predictions.reset_index(drop=True),
        "Model statistics": pd.DataFrame([evaluation.metrics]),
        "Cross-validation": evaluation.cv_scores,
        "CV predictions": evaluation.cv_predictions.reset_index(drop=True),
        "Williams AD": payload["williams_ad"].reset_index(drop=True),
        "Distance AD": payload["distance_ad"].reset_index(drop=True),
        "Excluded samples": payload.get("excluded_samples", pd.DataFrame()).reset_index(drop=True),
        "Parameters": params_frame,
        "Warnings": warnings_frame,
    }


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def persisted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep result data needed to restore the Results tab, excluding bulky figures."""

    return {key: _snapshot_value(value) for key, value in payload.items() if key not in {"figures", "bundle"}}


def build_run_state_snapshot() -> dict[str, Any]:
    """Capture the UI/data state needed to reopen a completed modeling run."""

    state_keys = [
        "dataset",
        "excluded_sample_ids",
        "outlier_log",
        "last_run_warnings",
        "endpoint_method",
        "preprocessing_config",
        "split_config",
        "drop_rows_after_split",
        "model_config",
        "feature_selection",
        "split_preview",
        "uploaded_name",
    ]
    widget_keys = [
        "pca_components_input",
        "pca_components_slider",
        "scale_pca",
        "pca_pc_x",
        "pca_pc_y",
        "endpoint_grouping_method",
        "endpoint_statistical_test",
        "endpoint_z_threshold",
        "endpoint_iqr_multiplier",
    ]
    snapshot = {key: _snapshot_value(st.session_state[key]) for key in state_keys if key in st.session_state}
    snapshot["pca_controls"] = {key: _snapshot_value(st.session_state[key]) for key in widget_keys if key in st.session_state}
    return snapshot


def _set_synced_int_state(key: str, value: Any) -> None:
    try:
        integer_value = int(value)
    except Exception:
        return
    st.session_state[f"{key}_input"] = integer_value
    st.session_state[f"{key}_slider"] = integer_value


def _missing_label_from_config(config: PreprocessingConfig, drop_rows_after_split: bool) -> str:
    if drop_rows_after_split:
        return "Remove rows with missing descriptors after split"
    reverse = {
        "median_impute": "Median imputation",
        "mean_impute": "Mean imputation",
        "drop_columns": "Remove descriptor columns with missing values",
        "none": "Require complete descriptor matrix",
    }
    return reverse.get(config.missing_strategy, "Median imputation")


def apply_restored_config_to_widgets() -> None:
    """Seed widget keys from restored configs before later tabs render."""

    preprocessing_config = st.session_state.get("preprocessing_config")
    if isinstance(preprocessing_config, PreprocessingConfig):
        st.session_state["missing_label"] = _missing_label_from_config(
            preprocessing_config,
            bool(st.session_state.get("drop_rows_after_split", False)),
        )
        st.session_state["remove_constant"] = bool(preprocessing_config.remove_constant)
        st.session_state["remove_low_variance"] = bool(preprocessing_config.remove_low_variance)
        st.session_state["variance_threshold"] = float(preprocessing_config.variance_threshold)
        st.session_state["remove_high_corr"] = bool(preprocessing_config.remove_high_correlation)
        st.session_state["corr_threshold"] = float(preprocessing_config.correlation_threshold)

    split_config = st.session_state.get("split_config")
    if isinstance(split_config, dict) and split_config:
        st.session_state["split_method"] = split_config.get("method", "Random split")
        st.session_state["random_seed"] = int(split_config.get("random_state", 42))
        if split_config.get("method") == "Random split":
            st.session_state["test_size"] = float(split_config.get("test_size", 0.25))
            st.session_state["use_stratified"] = split_config.get("stratify_bins") is not None
            if split_config.get("stratify_bins") is not None:
                st.session_state["stratify_bins"] = int(split_config.get("stratify_bins", 5))
        else:
            st.session_state["train_fraction"] = float(split_config.get("train_fraction", 0.75))
            st.session_state["sorted_strategy"] = split_config.get("sorted_strategy", "systematic")

    model_config = st.session_state.get("model_config")
    if isinstance(model_config, dict) and model_config:
        for key in ["scaler_name", "cv_folds", "cv_repeats", "ranking_metric", "selected_models"]:
            if key in model_config:
                st.session_state[key] = _snapshot_value(model_config[key])
        for model_name, params in model_config.get("model_params", {}).items():
            for param_name, value in params.items():
                widget_key = f"{model_name}_{param_name}"
                if param_name in {"n_components", "n_estimators"}:
                    _set_synced_int_state(widget_key, value)
                else:
                    st.session_state[widget_key] = _snapshot_value(value)

    feature_selection = st.session_state.get("feature_selection")
    if isinstance(feature_selection, dict) and feature_selection:
        method = feature_selection.get("method", "None")
        params = feature_selection.get("params", {}) or {}
        st.session_state["fs_method"] = method
        st.session_state["manual_descriptors"] = params.get("manual_descriptors", [])
        if "threshold" in params:
            st.session_state["fs_variance_threshold"] = float(params["threshold"])
        if "k" in params:
            _set_synced_int_state("select_k_best_k", params["k"])
        if "n_features" in params:
            _set_synced_int_state("rfe_n_features", params["n_features"])
        if method == "Genetic Algorithm":
            st.session_state["use_ga_cv"] = int(params.get("cv_folds", 5)) >= 2
            ga_synced = {
                "population_size": "ga_population_size",
                "generations": "ga_generations",
                "cv_folds": "ga_cv_folds",
                "early_stopping_rounds": "ga_early_stopping_rounds",
                "min_descriptors": "ga_min_descriptors",
                "max_descriptors": "ga_max_descriptors",
                "candidate_count": "ga_candidate_count",
                "keep_top_n": "ga_keep_top_n",
            }
            for param_name, key in ga_synced.items():
                value = feature_selection.get(param_name, params.get(param_name))
                if value is not None and not (param_name == "cv_folds" and int(value) < 2):
                    _set_synced_int_state(key, value)
            for param_name, key in {
                "crossover_probability": "ga_crossover_probability",
                "mutation_probability": "ga_mutation_probability",
                "tournament_size": "ga_tournament_size",
                "random_seed": "ga_random_seed",
                "scoring_metric": "ga_scoring_metric",
            }.items():
                if param_name in params:
                    st.session_state[key] = _snapshot_value(params[param_name])


def build_selected_model_bundle(label: str, payload: dict[str, Any], results_df: pd.DataFrame) -> ModelBundle:
    """Attach enough run state to a single-model export to restore the UI later."""

    return replace(
        payload["bundle"],
        result_payload=persisted_payload(payload),
        results_table=results_df.copy(),
        session_state=build_run_state_snapshot(),
    )


def build_model_run_bundle(results: dict[str, dict[str, Any]], results_df: pd.DataFrame, ranking_metric: str) -> ModelRunBundle:
    """Persist every kept model and the completed-run snapshot."""

    created_at = datetime.now().isoformat(timespec="seconds")
    bundles = {label: build_selected_model_bundle(label, payload, results_df) for label, payload in results.items()}
    training_results = {label: persisted_payload(payload) for label, payload in results.items()}
    return ModelRunBundle(
        run_label=f"QSAR model run {created_at}",
        bundles=bundles,
        results_table=results_df.copy(),
        metadata={
            "created_at": created_at,
            "ranking_metric": ranking_metric,
            "kept_models": len(bundles),
            "model_labels": list(bundles.keys()),
            "bundle_format": "run_snapshot_v2",
        },
        training_results=training_results,
        session_state=build_run_state_snapshot(),
    )


def restore_state_snapshot(snapshot: dict[str, Any]) -> bool:
    if not snapshot:
        return False
    for key, value in snapshot.items():
        if key == "pca_controls":
            continue
        st.session_state[key] = _snapshot_value(value)
    for key, value in snapshot.get("pca_controls", {}).items():
        st.session_state[key] = _snapshot_value(value)
    if "dataset" in snapshot and "sheets" not in snapshot:
        st.session_state.sheets = None
    apply_restored_config_to_widgets()
    return True


def hydrate_saved_payloads(saved_results: dict[str, dict[str, Any]], bundles: dict[str, ModelBundle]) -> dict[str, dict[str, Any]]:
    restored: dict[str, dict[str, Any]] = {}
    for label, payload in saved_results.items():
        hydrated = {key: _snapshot_value(value) for key, value in payload.items()}
        if label in bundles:
            hydrated["bundle"] = bundles[label]
        try:
            hydrated["figures"] = create_figures_for_result(label, hydrated)
        except Exception as exc:
            hydrated["figures"] = {}
            hydrated.setdefault("warnings", []).append(f"Could not regenerate saved plots after loading bundle: {exc}")
        restored[label] = hydrated
    return restored


def ensure_loaded_model_config(loaded: ModelBundle | ModelRunBundle) -> None:
    ranking_metric = getattr(loaded, "metadata", {}).get("ranking_metric") or "R2 test"
    current = st.session_state.get("model_config", {})
    if not isinstance(current, dict):
        current = {}
    current.setdefault("ranking_metric", ranking_metric)
    current.setdefault("selected_models", [])
    current.setdefault("model_params", {})
    current.setdefault("cv_folds", "restored")
    current.setdefault("cv_repeats", "restored")
    current.setdefault("scaler_name", "restored")
    st.session_state.model_config = current


def restore_loaded_bundle_to_session(loaded: ModelBundle | ModelRunBundle) -> list[tuple[str, str]]:
    """Restore Results/PCA state when a saved bundle contains a run snapshot."""

    messages: list[tuple[str, str]] = []
    if isinstance(loaded, ModelRunBundle):
        restored_state = restore_state_snapshot(getattr(loaded, "session_state", {}))
        saved_results = getattr(loaded, "training_results", {}) or {}
        if saved_results:
            st.session_state.training_results = hydrate_saved_payloads(saved_results, loaded.bundles)
            st.session_state.results_df = loaded.results_table.copy()
            st.session_state.last_run_warnings = st.session_state.get("last_run_warnings", [])
            ensure_loaded_model_config(loaded)
            messages.append(("success", f"Restored full run snapshot with {len(st.session_state.training_results)} model(s)."))
        else:
            messages.append(("warning", "This run bundle contains model objects, but no saved Results/PCA snapshot. Re-export the run with the current app version to restore the full workflow."))
        if restored_state:
            messages.append(("info", "Restored dataset, PCA exclusions, and modeling configuration from the bundle."))
        elif saved_results:
            messages.append(("warning", "This bundle restored model results, but it does not contain the original dataset needed for PCA screening."))
        return messages

    restored_state = restore_state_snapshot(getattr(loaded, "session_state", {}))
    saved_payload = getattr(loaded, "result_payload", {}) or {}
    if saved_payload:
        restored = hydrate_saved_payloads({loaded.model_label: saved_payload}, {loaded.model_label: loaded})
        st.session_state.training_results = restored
        table = getattr(loaded, "results_table", pd.DataFrame())
        if not table.empty and "Model label" in table.columns:
            table = table[table["Model label"] == loaded.model_label].copy()
        st.session_state.results_df = table.copy() if not table.empty else pd.DataFrame([{"Model label": loaded.model_label, **loaded.statistics}])
        ensure_loaded_model_config(loaded)
        messages.append(("success", f"Restored saved model snapshot for {loaded.model_label}."))
    else:
        messages.append(("warning", "This older single-model bundle can be used for prediction, but it does not contain the saved Results/PCA workflow snapshot."))
    if restored_state:
        messages.append(("info", "Restored dataset and modeling configuration stored inside the selected model bundle."))
    return messages


def loaded_model_map(loaded: ModelBundle | ModelRunBundle) -> dict[str, ModelBundle]:
    if isinstance(loaded, ModelRunBundle):
        return loaded.bundles
    return {loaded.model_label: loaded}


def run_training_workflow(
    dataset,
    endpoint_method: str,
    split_config: dict[str, Any],
    preprocessing_config: PreprocessingConfig,
    drop_rows_after_split: bool,
    selected_models: list[str],
    model_params: dict[str, dict[str, Any]],
    scaler_name: str,
    feature_selection: dict[str, Any],
    cv_folds: int,
    cv_repeats: int,
    ranking_metric: str,
    excluded_samples: pd.DataFrame | None = None,
):
    if not selected_models:
        raise ValueError("Select at least one model to train.")

    endpoint_transformer = EndpointTransformer(endpoint_method)
    y = endpoint_transformer.transform(dataset.y)
    selector_candidates = max(1, int(feature_selection.get("candidate_count", 1)))
    keep_top_n = max(1, int(feature_selection.get("keep_top_n", selector_candidates * len(selected_models))))
    if feature_selection["method"] != "Genetic Algorithm":
        selector_candidates = 1
        keep_top_n = max(1, len(selected_models))
    split_seed = int(split_config["random_state"])
    total_jobs = max(1, selector_candidates * len(selected_models))
    start_time = time.perf_counter()
    progress = st.progress(0, text=training_progress_text("Starting model training", 0, total_jobs, start_time))
    runtime_status = st.empty()
    runtime_status.caption(training_progress_text("Starting model training", 0, total_jobs, start_time))
    results: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    if feature_selection["method"] == "Genetic Algorithm" and int(feature_selection.get("params", {}).get("cv_folds", 5)) < 2:
        warnings.append(
            "GA descriptor selection used training-score fitness only. This is faster, but final CV and external test statistics should drive model choice."
        )
    job_counter = 0

    split = make_split(dataset.X, y, split_config)
    X_train_raw = split.X_train.copy()
    X_test_raw = split.X_test.copy()
    y_train = split.y_train.copy()
    y_test = split.y_test.copy()

    row_drop_report = {}
    if drop_rows_after_split:
        X_train_raw, y_train, X_test_raw, y_test, row_drop_report = drop_missing_rows(
            X_train_raw,
            y_train,
            X_test_raw,
            y_test,
        )

    if len(X_train_raw) < 4:
        raise ValueError("At least 4 training samples are required after preprocessing choices.")
    if len(X_test_raw) < 1:
        raise ValueError("No test samples remain after preprocessing choices.")

    preprocessor = DescriptorPreprocessor(preprocessing_config)
    X_train_preprocessed = preprocessor.fit_transform(X_train_raw, y_train)
    X_test_preprocessed = preprocessor.transform(X_test_raw)

    if X_train_preprocessed.shape[1] < 1:
        raise ValueError("No descriptors remain after preprocessing.")

    split_warnings = split.warnings.copy()
    if row_drop_report:
        split_warnings.append(
            f"Missing-row strategy removed {row_drop_report['train_rows_removed']} train rows and "
            f"{row_drop_report['test_rows_removed']} test rows."
        )
    warnings.extend(split_warnings)
    smiles_lookup = {}
    if getattr(dataset, "smiles", None) is not None:
        smiles_lookup = dataset.smiles.dropna().astype(str).to_dict()

    base_selector_seed = int(feature_selection.get("params", {}).get("random_seed", split_seed))
    for candidate_index in range(1, selector_candidates + 1):
        selector_seed = base_selector_seed + candidate_index - 1
        for model_name in selected_models:
            job_counter += 1
            job_start_units = job_counter - 1
            start_text = training_progress_text(
                f"Training {job_counter}/{total_jobs} models",
                job_start_units,
                total_jobs,
                start_time,
            )
            progress.progress(job_start_units / total_jobs, text=start_text)
            runtime_status.caption(start_text)

            raw_params = dict(model_params.get(model_name, {}))
            if model_name in FEATURE_IMPORTANCE_MODELS:
                raw_params["random_state"] = split_seed
            estimator_factory = make_estimator_factory(model_name, raw_params, scaler_name, len(y_train), cv_folds)

            selector_params = dict(feature_selection["params"])
            if feature_selection["method"] == "Genetic Algorithm":
                selector_params["random_seed"] = selector_seed
                ga_generations = int(selector_params.get("generations", 0))

                def ga_progress_callback(history_row, current_job=job_counter, generations=ga_generations):
                    generation = int(history_row.get("generation", 0))
                    total_generation_steps = max(1, generations + 1)
                    generation_units = min((generation + 1) / total_generation_steps, 0.98)
                    completed_units = (current_job - 1) + generation_units
                    best_score = history_row.get("best_score", np.nan)
                    best_score_text = f"{float(best_score):.4g}" if np.isfinite(best_score) else "n/a"
                    detail = (
                        f"GA generation {generation}/{generations}; "
                        f"evaluated subsets {int(history_row.get('evaluated_subsets', 0))}; "
                        f"best score {best_score_text}"
                    )
                    text = training_progress_text(
                        f"Training {current_job}/{total_jobs} models",
                        completed_units,
                        total_jobs,
                        start_time,
                        detail,
                    )
                    progress.progress(min(completed_units / total_jobs, 0.999), text=text)
                    runtime_status.caption(text)

                selector_params["progress_callback"] = ga_progress_callback
            selector_config = dict(feature_selection)
            selector_config["params"] = {key: value for key, value in selector_params.items() if key != "progress_callback"}

            selector = FeatureSelector(method=selector_config["method"], params=selector_params)
            X_train_selected = selector.fit_transform(X_train_preprocessed, y_train, estimator_factory=estimator_factory)
            X_test_selected = selector.transform(X_test_preprocessed)

            adjusted_params = safe_component_count(model_name, raw_params, X_train_selected.shape[1], len(y_train), cv_folds)
            estimator = build_pipeline(model_name, adjusted_params, scaler_name)
            estimator.fit(X_train_selected, y_train)
            evaluation = evaluate_fitted_model(
                estimator,
                X_train_selected,
                y_train,
                X_test_selected,
                y_test,
                cv_folds=cv_folds,
                cv_repeats=cv_repeats,
                random_state=split_seed,
            )
            if smiles_lookup:
                for frame in [evaluation.train_predictions, evaluation.test_predictions, evaluation.cv_predictions]:
                    frame["smiles"] = frame["sample_id"].map(smiles_lookup).fillna("")

            w_ad = williams_results(X_train_selected, X_test_selected, evaluation.train_predictions, evaluation.test_predictions)
            d_ad = distance_domain_results(X_train_selected, X_test_selected)
            pca_ad = pca_domain_scores(X_train_selected, X_test_selected)
            descriptor_importance = descriptor_importance_frame(estimator, selector.selected_descriptors_, model_name)
            model_equation = ""
            equation_terms = pd.DataFrame()
            if model_name == "MLR / Linear Regression":
                model_equation, equation_terms = mlr_equation(estimator, selector.selected_descriptors_)
            if smiles_lookup:
                for frame in [w_ad, d_ad, pca_ad]:
                    if not frame.empty and "sample_id" in frame.columns:
                        frame["smiles"] = frame["sample_id"].map(smiles_lookup).fillna("")
            model_short = model_name.split(" / ")[0]
            if selector_candidates > 1:
                model_label = f"Candidate {candidate_index:03d} | {model_short} - {feature_selection['method']} | seed {selector_seed}"
            else:
                model_label = f"{model_short} - {feature_selection['method']}"

            parameters = estimator_parameters(model_name, adjusted_params, scaler_name)
            parameters["feature_selection"] = selector_config
            parameters["endpoint_transformation"] = endpoint_method
            parameters["preprocessing"] = preprocessing_config.__dict__
            parameters["split"] = split_config
            parameters["candidate_index"] = candidate_index
            parameters["selector_seed"] = selector_seed
            parameters["selector_candidates"] = selector_candidates
            parameters["cv_folds"] = cv_folds
            parameters["cv_repeats"] = cv_repeats
            parameters["excluded_samples"] = excluded_samples.to_dict("records") if excluded_samples is not None else []
            if model_equation:
                parameters["model_equation"] = model_equation

            bundle = ModelBundle(
                model_label=model_label,
                model_name=model_name,
                estimator=estimator,
                preprocessor=preprocessor,
                feature_selector=selector,
                endpoint_transformer=endpoint_transformer,
                selected_descriptors=selector.selected_descriptors_,
                metadata={
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "endpoint_name": dataset.endpoint_name,
                    "ranking_metric": ranking_metric,
                    "parameters": parameters,
                    "model_equation": model_equation,
                    "excluded_samples": excluded_samples.to_dict("records") if excluded_samples is not None else [],
                },
                train_reference_X=X_train_selected,
                statistics=evaluation.metrics,
            )

            payload = {
                "model_name": model_name,
                "candidate_index": candidate_index,
                "random_seed": selector_seed,
                "estimator": estimator,
                "preprocessor": preprocessor,
                "feature_selector": selector,
                "selected_descriptors": selector.selected_descriptors_,
                "evaluation": evaluation,
                "williams_ad": w_ad,
                "distance_ad": d_ad,
                "pca_ad": pca_ad,
                "parameters": parameters,
                "descriptor_importance": descriptor_importance,
                "model_equation": model_equation,
                "equation_terms": equation_terms,
                "warnings": evaluation.warnings.copy(),
                "bundle": bundle,
                "split_membership": split.membership,
                "X_train_selected": X_train_selected,
                "X_test_selected": X_test_selected,
                "excluded_samples": excluded_samples if excluded_samples is not None else pd.DataFrame(),
            }
            if split_warnings:
                payload["warnings"].extend(split_warnings)
            if d_ad[d_ad["split"] == "test"]["outside_ad"].any():
                payload["warnings"].append("One or more test compounds are outside the distance-based applicability domain.")
            if w_ad[w_ad["split"] == "test"]["outside_ad"].any():
                payload["warnings"].append("One or more test compounds are outside the Williams-plot applicability domain.")
            results[model_label] = payload

            done_text = training_progress_text(
                f"Training {job_counter}/{total_jobs} models",
                job_counter,
                total_jobs,
                start_time,
            )
            progress.progress(job_counter / total_jobs, text=done_text)
            runtime_status.caption(done_text)

    table = results_table(
        {
            label: {
                "metrics": payload["evaluation"].metrics,
                "model_name": payload["model_name"],
                "candidate_index": payload.get("candidate_index"),
                "random_seed": payload.get("random_seed"),
                "selected_descriptors": payload["selected_descriptors"],
                "parameters": payload["parameters"],
            }
            for label, payload in results.items()
        }
    )
    ranked = rank_models(table, ranking_metric).head(keep_top_n).reset_index(drop=True)
    kept_labels = ranked["Model label"].tolist() if not ranked.empty else []
    results = {label: results[label] for label in kept_labels}
    elapsed_text = format_duration(time.perf_counter() - start_time)
    progress.progress(1.0, text=f"Model training complete in {elapsed_text}. Kept top {len(results)} of {total_jobs} models.")
    runtime_status.success(f"Training finished in {elapsed_text}. Kept top {len(results)} of {total_jobs} models.")
    return results, ranked, warnings


def render_saved_bundle_loader() -> None:
    model_upload = st.file_uploader("Saved model/run bundle (.joblib)", type=["joblib", "pkl"], key="bundle_upload")
    if model_upload is not None:
        bundle_bytes = model_upload.getvalue()
        bundle_key = f"{model_upload.name}:{len(bundle_bytes)}"
        if st.session_state.loaded_bundle_key != bundle_key:
            try:
                loaded = bundle_from_bytes(bundle_bytes)
                if not isinstance(loaded, (ModelBundle, ModelRunBundle)):
                    raise TypeError("The uploaded file is not a QSAR model bundle.")
                st.session_state.loaded_bundle = loaded
                st.session_state.loaded_bundle_key = bundle_key
                st.session_state.prediction_output = None
                st.session_state.bundle_restore_messages = restore_loaded_bundle_to_session(loaded)
                if isinstance(loaded, ModelRunBundle):
                    st.success(f"Loaded model run: {loaded.run_label} ({len(loaded.bundles)} models)")
                else:
                    st.success(f"Loaded model: {loaded.model_label}")
            except Exception as exc:
                st.session_state.loaded_bundle = None
                st.session_state.loaded_bundle_key = None
                st.session_state.bundle_restore_messages = []
                st.error(f"Could not load model bundle: {exc}")

    for level, message in st.session_state.bundle_restore_messages:
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)

    if st.session_state.loaded_bundle is None:
        st.caption("Upload a current full-run bundle to restore PCA screening, preprocessing, split, feature selection, results, plots, and export state.")
        return

    loaded = st.session_state.loaded_bundle
    model_map = loaded_model_map(loaded)
    is_run_bundle = isinstance(loaded, ModelRunBundle)

    if is_run_bundle:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_panel("Bundle type", "Full run")
        with c2:
            metric_panel("Models", len(model_map))
        with c3:
            metric_panel("Ranking rows", len(loaded.results_table))

        if not loaded.results_table.empty:
            with st.expander("Loaded run ranking table", expanded=False):
                st.dataframe(loaded.results_table, use_container_width=True)

        selected_loaded_label = st.selectbox("Inspect model from loaded run", list(model_map.keys()), key="loaded_run_model")
        bundle = model_map[selected_loaded_label]
        st.info("Prediction on a new descriptor workbook will be executed for every model in this loaded run package.")
    else:
        bundle = next(iter(model_map.values()))

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_panel("Model", bundle.model_name)
    with c2:
        metric_panel("Descriptors", len(bundle.selected_descriptors))
    with c3:
        metric_panel("Endpoint", bundle.metadata.get("endpoint_name", "n/a"))

    with st.expander("Loaded bundle metadata", expanded=False):
        if is_run_bundle:
            st.caption("Run metadata")
            st.json(loaded.metadata)
        st.caption("Selected/inspected model metadata")
        st.json(bundle.metadata)
        st.dataframe(list_to_frame(bundle.selected_descriptors), use_container_width=True)

    with st.expander("Predict new compounds with loaded bundle", expanded=False):
        new_data = st.file_uploader("New descriptor workbook", type=["xlsx", "xlsm", "xls"], key="new_data_upload")
        if new_data is not None:
            try:
                new_sheets = cached_read_excel_sheets(new_data.getvalue())
                new_sheet_name = st.selectbox("Descriptor sheet for prediction", list(new_sheets.keys()))
                use_new_id = st.checkbox("Use first column as sample ID for prediction", value=False, key="new_use_id")
                X_new = clean_sheet(new_sheets[new_sheet_name], use_first_column_as_index=use_new_id)
                st.dataframe(X_new.head(20), use_container_width=True)
                if st.button("Predict new compounds", type="primary"):
                    if is_run_bundle:
                        predictions, ad_table, errors = predict_with_run_bundle(loaded, X_new)
                    else:
                        predictions, ad_table = predict_with_bundle(bundle, X_new)
                        errors = pd.DataFrame()
                    st.session_state.prediction_output = {
                        "predictions": predictions,
                        "ad": ad_table,
                        "errors": errors,
                        "is_run_bundle": is_run_bundle,
                    }
            except Exception as exc:
                st.error(f"Could not prepare new descriptors: {exc}")

        if st.session_state.prediction_output is not None:
            predictions = st.session_state.prediction_output["predictions"]
            ad_table = st.session_state.prediction_output["ad"]
            errors = st.session_state.prediction_output.get("errors", pd.DataFrame())

            if predictions.empty:
                st.error("No predictions were generated. Check descriptor names and preprocessing compatibility with the saved model bundle.")
            else:
                st.caption("New compound predictions")
                st.dataframe(predictions, use_container_width=True)

            if errors is not None and not errors.empty:
                st.warning("Some models from the loaded run could not predict these descriptors.")
                st.dataframe(errors, use_container_width=True)

            if not ad_table.empty:
                st.caption("Applicability-domain assessment")
                st.dataframe(ad_table, use_container_width=True)
                if ad_table["outside_ad"].any():
                    st.warning("One or more new compounds are outside the saved model distance-based applicability domain.")

            export_predictions = predictions.copy()
            if not export_predictions.empty and not ad_table.empty:
                ad_export = ad_table.drop(columns=["split", "model_name"], errors="ignore")
                merge_keys = ["sample_id"]
                if "model_label" in export_predictions.columns and "model_label" in ad_export.columns:
                    merge_keys = ["model_label", "sample_id"]
                export_predictions = export_predictions.merge(ad_export, on=merge_keys, how="left")
            if not export_predictions.empty:
                st.download_button(
                    "Download new predictions CSV",
                    data=dataframe_to_csv_bytes(export_predictions.reset_index(drop=True)),
                    file_name="new_compound_predictions.csv",
                    mime="text/csv",
                )


def show_dataset_gate():
    if st.session_state.dataset is None:
        st.info("Upload and align a workbook in Data upload to activate this section.")
        return False
    return True


init_state()

st.markdown('<div class="main-title">QSAR/QSPR Modeling Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A leakage-aware workspace for descriptor preprocessing, regression modeling, validation, applicability-domain analysis, and export.</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "1. Data upload",
        "2. Preprocessing",
        "3. PCA screening",
        "4. Train/test split",
        "5. Model configuration",
        "6. Feature selection",
        "7. Training and validation",
        "8. Results and plots",
        "9. Export",
    ]
)


with tabs[0]:
    st.subheader("Data upload")
    st.markdown("#### Restore saved run or model")
    render_saved_bundle_loader()
    st.divider()
    st.markdown("#### Start from Excel workbook")
    uploaded = st.file_uploader("Excel workbook", type=["xlsx", "xlsm", "xls"])
    if uploaded is not None and uploaded.name != st.session_state.uploaded_name:
        try:
            st.session_state.sheets = cached_read_excel_sheets(uploaded.getvalue())
            st.session_state.uploaded_name = uploaded.name
            st.session_state.dataset = None
            st.session_state.loaded_bundle = None
            st.session_state.loaded_bundle_key = None
            st.session_state.bundle_restore_messages = []
            st.session_state.prediction_output = None
            st.session_state.excluded_sample_ids = []
            st.session_state.outlier_log = pd.DataFrame(columns=["sample_id", "reason", "removed_at", "source"])
            reset_modeling_outputs()
            st.success(f"Loaded {len(st.session_state.sheets)} sheet(s) from {uploaded.name}.")
        except Exception as exc:
            st.session_state.sheets = None
            st.error(f"Could not read workbook: {exc}")

    if uploaded is None and not st.session_state.sheets and st.session_state.dataset is None and st.session_state.loaded_bundle is None:
        st.info("Upload an Excel workbook to begin, or restore a saved .joblib run above.")

    if st.session_state.sheets:
        sheets = st.session_state.sheets
        sheet_names = list(sheets.keys())
        col_a, col_b = st.columns([1, 1])
        with col_a:
            use_id_column = st.checkbox("Use first column as sample ID", value=False)
            x_sheet_name = st.selectbox("Descriptor sheet (X)", sheet_names, key="x_sheet")
        with col_b:
            y_sheet_name = st.selectbox("Endpoint sheet (y)", sheet_names, key="y_sheet")
            y_clean = clean_sheet(sheets[y_sheet_name], use_first_column_as_index=use_id_column)
            endpoint_columns = [str(col) for col in y_clean.columns]
            if not endpoint_columns:
                st.error("The selected endpoint sheet has no usable columns after empty columns were removed.")
                st.stop()
            endpoint_column = st.selectbox("Endpoint column", endpoint_columns)

        smiles_sheet_name = None
        smiles_column = None
        use_smiles = st.checkbox("Optional: select SMILES column for structure preview", value=False)
        if use_smiles:
            smiles_cols = st.columns([1, 1])
            with smiles_cols[0]:
                smiles_sheet_name = st.selectbox("SMILES sheet", sheet_names, key="smiles_sheet")
            with smiles_cols[1]:
                smiles_clean = clean_sheet(sheets[smiles_sheet_name], use_first_column_as_index=use_id_column)
                smiles_columns = [str(col) for col in smiles_clean.columns]
                if smiles_columns:
                    default_smiles_idx = next(
                        (idx for idx, col in enumerate(smiles_columns) if "smiles" in col.casefold()),
                        0,
                    )
                    smiles_column = st.selectbox("SMILES column", smiles_columns, index=default_smiles_idx, key="smiles_column")
                else:
                    st.warning("The selected SMILES sheet has no usable columns.")

        preview_cols = st.columns(2)
        with preview_cols[0]:
            st.caption("Descriptor preview")
            st.dataframe(clean_sheet(sheets[x_sheet_name], use_first_column_as_index=use_id_column).head(20), use_container_width=True)
        with preview_cols[1]:
            st.caption("Endpoint preview")
            st.dataframe(y_clean.head(20), use_container_width=True)

        if st.button("Align descriptor and endpoint data", type="primary"):
            try:
                dataset = prepare_xy(
                    sheets[x_sheet_name],
                    sheets[y_sheet_name],
                    endpoint_column,
                    use_first_column_as_index=use_id_column,
                    smiles_sheet=sheets[smiles_sheet_name] if smiles_sheet_name else None,
                    smiles_column=smiles_column,
                )
                st.session_state.dataset = dataset
                st.session_state.loaded_bundle = None
                st.session_state.loaded_bundle_key = None
                st.session_state.bundle_restore_messages = []
                st.session_state.prediction_output = None
                st.session_state.excluded_sample_ids = []
                st.session_state.outlier_log = pd.DataFrame(columns=["sample_id", "reason", "removed_at", "source"])
                reset_modeling_outputs()
                st.success(f"Aligned {dataset.X.shape[0]} samples and {dataset.X.shape[1]} numeric descriptors.")
                display_messages(dataset.warnings)
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.dataset is not None:
        raw_dataset = st.session_state.dataset
        dataset = active_dataset()
        summary = dataset_summary(dataset.X, dataset.y)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            metric_panel("Total samples", raw_dataset.X.shape[0])
        with m2:
            metric_panel("Active samples", dataset.X.shape[0])
        with m3:
            metric_panel("Numeric descriptors", summary["descriptors"])
        with m4:
            metric_panel("Missing descriptor values", summary["missing_values"])
        with m5:
            metric_panel("Endpoint", dataset.endpoint_name)
        with m6:
            smiles_count = int(dataset.smiles.dropna().shape[0]) if dataset.smiles is not None else 0
            metric_panel("SMILES", smiles_count)
        display_messages(dataset.warnings)
        if st.session_state.excluded_sample_ids:
            st.info(f"{len(st.session_state.excluded_sample_ids)} sample(s) are excluded from modeling after screening.")
        with st.expander("Initial descriptor statistics", expanded=False):
            st.dataframe(summary["descriptor_stats"].head(100), use_container_width=True)

with tabs[1]:
    st.subheader("Preprocessing")
    if show_dataset_gate():
        dataset = active_dataset()
        summary = dataset_summary(dataset.X, dataset.y)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_panel("Missing rows", summary["missing_rows"])
        with c2:
            metric_panel("Missing columns", summary["missing_columns"])
        with c3:
            metric_panel("Endpoint min", f"{dataset.y.min():.4g}")
        with c4:
            metric_panel("Endpoint max", f"{dataset.y.max():.4g}")

        left, right = st.columns([0.9, 1.1])
        with left:
            endpoint_method = st.selectbox(
                "Endpoint transformation",
                ["none", "log10", "negative_log10"],
                format_func=lambda value: {"none": "No transformation", "log10": "log10(y)", "negative_log10": "-log10(y)"}[value],
                key="endpoint_method",
            )
            histogram_bins = st.slider("Endpoint histogram bins", 5, 80, 24, key="histogram_bins")
            missing_label = st.selectbox(
                "Missing descriptor handling",
                [
                    "Median imputation",
                    "Mean imputation",
                    "Remove descriptor columns with missing values",
                    "Remove rows with missing descriptors after split",
                    "Require complete descriptor matrix",
                ],
                key="missing_label",
            )
            missing_strategy, drop_rows_flag = mapping_missing_strategy(missing_label)
            remove_constant = st.checkbox("Remove constant descriptors", value=True, key="remove_constant")
            remove_low_variance = st.checkbox("Remove low-variance descriptors", value=True, key="remove_low_variance")
            variance_threshold = st.number_input(
                "Low-variance threshold",
                min_value=0.0,
                value=0.0,
                step=0.0001,
                format="%.6f",
                key="variance_threshold",
            )
            remove_high_corr = st.checkbox("Remove highly correlated descriptors", value=True, key="remove_high_corr")
            corr_threshold = st.slider("Absolute correlation threshold", 0.50, 0.999, 0.90, 0.01, key="corr_threshold")

        preview_config = PreprocessingConfig(
            missing_strategy=missing_strategy,
            remove_constant=remove_constant,
            remove_low_variance=remove_low_variance,
            variance_threshold=variance_threshold,
            remove_high_correlation=remove_high_corr,
            correlation_threshold=corr_threshold,
        )
        st.session_state.preprocessing_config = preview_config
        st.session_state.drop_rows_after_split = drop_rows_flag

        with right:
            transformed_y, transform_warnings = endpoint_transform_preview(dataset.y, endpoint_method)
            transformed_y.name = endpoint_label(dataset.endpoint_name, endpoint_method)
            display_messages(transform_warnings)
            hist_cols = st.columns(2)
            with hist_cols[0]:
                st.pyplot(endpoint_histogram(dataset.y, bins=histogram_bins, title="Endpoint before transformation"))
            with hist_cols[1]:
                st.pyplot(endpoint_histogram(transformed_y, bins=histogram_bins, title="Endpoint after transformation"))
            stats_frame = pd.DataFrame(
                {
                    "Before": dataset.y.describe(),
                    "After": transformed_y.describe(),
                }
            )
            st.dataframe(stats_frame, use_container_width=True)

        with st.expander("Endpoint statistical screening", expanded=False):
            st.caption(
                "Exploratory preprocessing check: descriptors are compared between endpoint-defined groups, "
                "and endpoint outliers are flagged with z-score/IQR rules. These tests do not select descriptors automatically."
            )
            stat_cols = st.columns(5)
            with stat_cols[0]:
                grouping_method = st.selectbox(
                    "Endpoint groups",
                    ["Lower vs upper quartile", "Median split"],
                    key="endpoint_grouping_method",
                )
            with stat_cols[1]:
                statistical_test = st.selectbox(
                    "Statistical test",
                    ["Welch t-test", "Student t-test", "Mann-Whitney U"],
                    key="endpoint_statistical_test",
                )
            with stat_cols[2]:
                z_threshold = st.number_input(
                    "Endpoint z threshold",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.25,
                    key="endpoint_z_threshold",
                )
            with stat_cols[3]:
                iqr_multiplier = st.number_input(
                    "IQR multiplier",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.5,
                    step=0.25,
                    key="endpoint_iqr_multiplier",
                )
            with stat_cols[4]:
                descriptor_rows = st.number_input(
                    "Descriptor rows shown",
                    min_value=5,
                    max_value=1000,
                    value=50,
                    step=5,
                    key="endpoint_stat_descriptor_rows",
                )

            screening_config = preview_config
            if drop_rows_flag and preview_config.missing_strategy == "none":
                screening_config = replace(preview_config, missing_strategy="median_impute")
                st.info("This exploratory screening uses median imputation because row removal is applied only after train/test split.")
            st.caption(
                "For modeling, preprocessing is still fitted only on the training set after the split. "
                "This table is a data-review preview for the currently active dataset."
            )
            try:
                screening_preprocessor = DescriptorPreprocessor(screening_config)
                X_screening = screening_preprocessor.fit_transform(dataset.X, transformed_y)
                descriptor_columns = X_screening.columns.astype(str).tolist()
                stat_table, endpoint_grouping = compare_endpoint_groups(
                    X_screening,
                    transformed_y,
                    descriptor_columns,
                    grouping_method=grouping_method,
                    test_name=statistical_test,
                )
                if not stat_table.empty:
                    stat_table["abs_effect_size"] = stat_table["effect_size_cohens_d"].abs()
                    stat_table = stat_table.sort_values(
                        ["p_value", "abs_effect_size"],
                        ascending=[True, False],
                        na_position="last",
                    )
                st.caption(endpoint_grouping.description)
                st.dataframe(
                    stat_table.head(int(descriptor_rows)).style.format(
                        {
                            "low_mean": "{:.5g}",
                            "high_mean": "{:.5g}",
                            "mean_difference_high_minus_low": "{:.5g}",
                            "effect_size_cohens_d": "{:.4f}",
                            "abs_effect_size": "{:.4f}",
                            "statistic": "{:.4g}",
                            "p_value": "{:.4g}",
                        }
                    ),
                    use_container_width=True,
                )
                if len(stat_table) > int(descriptor_rows):
                    st.caption(f"Showing top {int(descriptor_rows)} of {len(stat_table)} descriptors sorted by p-value and effect size.")
            except Exception as exc:
                st.warning(f"Endpoint group descriptor comparison could not be calculated: {exc}")

            endpoint_flags = endpoint_outlier_table(
                transformed_y,
                z_threshold=z_threshold,
                iqr_multiplier=iqr_multiplier,
            )
            flagged_endpoint_ids = endpoint_flags.loc[endpoint_flags["flagged"], "sample_id"].astype(str).tolist()
            st.caption("Endpoint outlier flags")
            st.dataframe(
                endpoint_flags.style.format(
                    {
                        "endpoint": "{:.5g}",
                        "z_score": "{:.4f}",
                        "iqr_lower_fence": "{:.5g}",
                        "iqr_upper_fence": "{:.5g}",
                    }
                ),
                use_container_width=True,
            )
            if flagged_endpoint_ids:
                st.warning(f"Endpoint statistical screening flagged {len(flagged_endpoint_ids)} sample(s). Review before excluding.")
                if st.button("Exclude endpoint-flagged samples", key="exclude_endpoint_flags"):
                    existing = {str(sample_id) for sample_id in st.session_state.excluded_sample_ids}
                    new_ids = [sample_id for sample_id in flagged_endpoint_ids if sample_id not in existing]
                    if new_ids:
                        st.session_state.excluded_sample_ids = sorted(existing.union(new_ids))
                        new_log = pd.DataFrame(
                            {
                                "sample_id": new_ids,
                                "reason": "Endpoint statistical screening",
                                "removed_at": datetime.now().isoformat(timespec="seconds"),
                                "source": "Preprocessing endpoint statistics",
                            }
                        )
                        st.session_state.outlier_log = pd.concat(
                            [st.session_state.outlier_log, new_log],
                            ignore_index=True,
                        )
                        reset_modeling_outputs()
                        st.rerun()
            else:
                st.success("No endpoint outliers were flagged by the selected z-score/IQR rules.")

with tabs[2]:
    st.subheader("PCA screening")
    if show_dataset_gate():
        dataset = active_dataset()
        raw_dataset = st.session_state.dataset
        excluded = excluded_samples_frame()
        if st.session_state.excluded_sample_ids:
            st.info(
                f"{len(st.session_state.excluded_sample_ids)} sample(s) are currently excluded from modeling. "
                "PCA below is recalculated on the active dataset."
            )

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            metric_panel("Active samples", dataset.X.shape[0])
        with p2:
            metric_panel("Excluded samples", len(st.session_state.excluded_sample_ids))
        with p3:
            metric_panel("Descriptors", dataset.X.shape[1])
        with p4:
            metric_panel("Endpoint", raw_dataset.endpoint_name)

        try:
            pca_X, pca_y, pca_endpoint_label, pca_preprocessor, pca_notes = prepared_pca_inputs(dataset)
        except Exception as exc:
            st.error(f"PCA preprocessing failed: {exc}")
            pca_X = pd.DataFrame()
            pca_y = pd.Series(dtype=float)
            pca_endpoint_label = raw_dataset.endpoint_name
            pca_preprocessor = None
            pca_notes = []

        if not pca_X.empty:
            p_metric_1, p_metric_2 = st.columns(2)
            with p_metric_1:
                metric_panel("PCA descriptors", pca_X.shape[1])
            with p_metric_2:
                metric_panel("Endpoint scale", pca_endpoint_label)

        if dataset.X.shape[0] < 3:
            st.warning("PCA screening requires at least 3 active samples.")
        elif pca_X.shape[1] < 2:
            st.warning("PCA screening requires at least 2 descriptors after current preprocessing settings.")
        else:
            max_pcs = min(pca_X.shape[0], pca_X.shape[1], 100)
            n_pcs = synced_int_control("PCA components", 2, max(2, max_pcs), min(5, max_pcs), 1, "pca_components")
            scale_pca = st.checkbox("Standardize descriptors before PCA", value=True, key="scale_pca")
            try:
                pca_result = cached_pca_screening(pca_X, pca_y, n_components=n_pcs, scale=scale_pca)
                if dataset.smiles is not None:
                    pca_result.scores["smiles"] = dataset.smiles.reindex(pca_result.scores.index).astype("string").fillna("")
                variance = pca_result.variance.copy()
                score_cols = [col for col in pca_result.scores.columns if col.startswith("PC")]
                pc_left, pc_right = st.columns(2)
                with pc_left:
                    pc_x = st.selectbox("X axis PC", score_cols, index=0, key="pca_pc_x")
                with pc_right:
                    default_y_index = 1 if len(score_cols) > 1 else 0
                    pc_y = st.selectbox("Y axis PC", score_cols, index=default_y_index, key="pca_pc_y")

                pca_event = st.plotly_chart(
                    pca_score_figure(pca_result.scores, pc_x, pc_y, endpoint_label=pca_endpoint_label),
                    use_container_width=True,
                    key="pca_screening_plot",
                    on_select="rerun",
                    selection_mode="points",
                )
                selected_pca_id = sample_id_from_plotly_selection(pca_event)
                render_structure_panel(
                    dataset.smiles,
                    "pca_screening",
                    selected_sample_id=selected_pca_id,
                    sample_ids=pca_result.scores["sample_id"].astype(str).tolist(),
                )

                v1, v2 = st.columns([1.0, 1.2])
                with v1:
                    st.caption("PCA eigenvalues and explained variance")
                    st.dataframe(
                        variance.style.format(
                            {
                                "eigenvalue": "{:.5g}",
                                "explained_variance_ratio": "{:.5f}",
                                "explained_variance_percent": "{:.2f}",
                                "cumulative_variance_percent": "{:.2f}",
                            }
                        ),
                        use_container_width=True,
                    )
                with v2:
                    st.caption("PCA scores")
                    st.dataframe(pca_result.scores, use_container_width=True, height=305)

                with st.expander("PCA preprocessing summary", expanded=False):
                    if pca_notes:
                        display_messages(pca_notes, level="info")
                    if pca_preprocessor is not None:
                        st.caption("Current preprocessing applied before PCA")
                        st.dataframe(pca_preprocessor.get_report().to_frame(), use_container_width=True)
                    st.caption("PCA internal safety preprocessing")
                    st.dataframe(pca_result.preprocessing, use_container_width=True)

                st.markdown("#### Remove outliers from modeling")
                selectable_ids = pca_result.scores["sample_id"].astype(str).tolist()
                selected_outliers = st.multiselect(
                    "Sample IDs to exclude",
                    selectable_ids,
                    help="Hover a point in the PCA plot to read its ID, then select it here to exclude it from modeling.",
                )
                reason = st.text_input("Reason stored in report", value="PCA outlier review")
                action_cols = st.columns([1, 1, 2])
                with action_cols[0]:
                    if st.button("Exclude selected", type="primary", disabled=not selected_outliers):
                        existing = {str(sample_id) for sample_id in st.session_state.excluded_sample_ids}
                        new_ids = [str(sample_id) for sample_id in selected_outliers if str(sample_id) not in existing]
                        if new_ids:
                            st.session_state.excluded_sample_ids = sorted(existing.union(new_ids))
                            new_log = pd.DataFrame(
                                {
                                    "sample_id": new_ids,
                                    "reason": reason or "PCA outlier review",
                                    "removed_at": datetime.now().isoformat(timespec="seconds"),
                                    "source": "PCA screening",
                                }
                            )
                            st.session_state.outlier_log = pd.concat(
                                [st.session_state.outlier_log, new_log],
                                ignore_index=True,
                            )
                            reset_modeling_outputs()
                            st.rerun()
                with action_cols[1]:
                    if st.button("Restore all excluded", disabled=not st.session_state.excluded_sample_ids):
                        st.session_state.excluded_sample_ids = []
                        reset_modeling_outputs()
                        st.rerun()

                if not excluded.empty:
                    st.caption("Samples excluded from modeling and report")
                    st.dataframe(excluded, use_container_width=True)
                    restore_ids = st.multiselect("Restore selected IDs", excluded["sample_id"].astype(str).tolist())
                    if st.button("Restore selected", disabled=not restore_ids):
                        restore_set = {str(sample_id) for sample_id in restore_ids}
                        st.session_state.excluded_sample_ids = [
                            sample_id for sample_id in st.session_state.excluded_sample_ids if str(sample_id) not in restore_set
                        ]
                        reset_modeling_outputs()
                        st.rerun()
            except Exception as exc:
                st.error(f"PCA screening failed: {exc}")

with tabs[3]:
    st.subheader("Train/test split")
    if show_dataset_gate():
        dataset = active_dataset()
        split_method = st.radio("Split method", ["Random split", "Sorted endpoint split"], horizontal=True, key="split_method")
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="random_seed")
        split_config: dict[str, Any] = {"method": split_method, "random_state": int(seed)}

        if split_method == "Random split":
            test_size = st.slider("Test set fraction", 0.10, 0.50, 0.25, 0.05, key="test_size")
            use_stratified = st.checkbox("Use binned endpoint stratification", value=True, key="use_stratified")
            stratify_bins = st.slider("Endpoint bins", 2, 10, 5, key="stratify_bins") if use_stratified else None
            split_config.update({"test_size": test_size, "stratify_bins": stratify_bins})
        else:
            train_fraction = st.slider("Training fraction", 0.50, 0.90, 0.75, 0.05, key="train_fraction")
            sorted_strategy = st.selectbox(
                "Sorted split assignment",
                ["systematic", "random_remaining"],
                format_func=lambda value: "Systematic test spacing" if value == "systematic" else "Random remaining samples",
                key="sorted_strategy",
            )
            split_config.update({"train_fraction": train_fraction, "sorted_strategy": sorted_strategy})

        st.session_state.split_config = split_config

        if st.button("Preview split", type="primary"):
            try:
                endpoint_method = st.session_state.get("endpoint_method", "none")
                y_split = EndpointTransformer(endpoint_method).transform(dataset.y)
                st.session_state.split_preview = make_split(dataset.X, y_split, split_config)
            except Exception as exc:
                st.error(str(exc))

        if st.session_state.split_preview is not None:
            split = st.session_state.split_preview
            range_cols = st.columns(2)
            with range_cols[0]:
                st.dataframe(split_range_table(split.y_train, split.y_test), use_container_width=True)
            with range_cols[1]:
                st.dataframe(split.membership, use_container_width=True, height=260)
            display_messages(split.warnings)


with tabs[4]:
    st.subheader("Model configuration")
    if show_dataset_gate():
        scaler_name = st.selectbox("Scaling", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"], index=1, key="scaler_name")
        cv_cols = st.columns(3)
        with cv_cols[0]:
            cv_folds = st.slider("Cross-validation folds", 2, 10, 5, key="cv_folds")
        with cv_cols[1]:
            cv_repeats = st.slider("Repeated CV runs", 1, 5, 1, key="cv_repeats")
        with cv_cols[2]:
            ranking_metric = st.selectbox(
                "Ranking metric",
                ["Q2 CV", "R2 test", "RMSE test", "RMSE CV", "MAE test", "MAE CV", "R2 train"],
                key="ranking_metric",
            )

        selected_models = st.multiselect(
            "Models to train",
            MODEL_NAMES,
            default=[
                "MLR / Linear Regression",
                "PLS / Partial Least Squares",
                "SVR / Support Vector Regression",
                "RF / Random Forest",
            ],
            key="selected_models",
        )

        model_params: dict[str, dict[str, Any]] = {}
        for model_name in selected_models:
            with st.expander(model_name, expanded=False):
                params: dict[str, Any] = {}
                if model_name == "MLR / Linear Regression":
                    params["fit_intercept"] = st.checkbox("Fit intercept", value=True, key=f"{model_name}_fit_intercept")
                elif model_name == "PCR / Principal Component Regression":
                    params["n_components"] = synced_int_control(
                        "Principal components",
                        1,
                        max(1, min(100, active_dataset().X.shape[1])),
                        2,
                        1,
                        f"{model_name}_n_components",
                    )
                    params["fit_intercept"] = st.checkbox("Fit intercept", value=True, key=f"{model_name}_fit_intercept")
                elif model_name == "PLS / Partial Least Squares":
                    params["n_components"] = synced_int_control(
                        "PLS components",
                        1,
                        max(1, min(100, active_dataset().X.shape[1])),
                        2,
                        1,
                        f"{model_name}_n_components",
                    )
                    params["scale"] = st.checkbox("Use PLS internal scaling", value=False, key=f"{model_name}_scale")
                elif model_name == "SVR / Support Vector Regression":
                    params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key=f"{model_name}_kernel")
                    params["C"] = st.number_input("C", min_value=0.001, value=10.0, step=1.0, key=f"{model_name}_C")
                    params["epsilon"] = st.number_input("Epsilon", min_value=0.0, value=0.1, step=0.05, key=f"{model_name}_epsilon")
                    params["gamma"] = st.selectbox("Gamma", ["scale", "auto"], key=f"{model_name}_gamma")
                    params["degree"] = st.slider("Polynomial degree", 2, 6, 3, key=f"{model_name}_degree")
                elif model_name == "RF / Random Forest":
                    params["n_estimators"] = synced_int_control("Trees", 50, 5000, 300, 50, f"{model_name}_n_estimators")
                    params["max_depth"] = st.number_input("Max depth (0 for unlimited)", min_value=0, value=0, step=1, key=f"{model_name}_max_depth")
                    params["min_samples_split"] = st.slider("Min samples split", 2, 20, 2, key=f"{model_name}_min_samples_split")
                    params["min_samples_leaf"] = st.slider("Min samples leaf", 1, 20, 1, key=f"{model_name}_min_samples_leaf")
                    params["max_features"] = st.selectbox("Max features", ["sqrt", "log2", None, 1.0], key=f"{model_name}_max_features")
                    params["random_state"] = int(st.session_state.get("random_seed", 42))
                elif model_name == "AdaBoost / Adaptive Boosting":
                    st.caption("AdaBoost regression uses shallow decision trees as weak learners. Interpret descriptor influence with feature importance, not linear coefficients.")
                    params["n_estimators"] = synced_int_control("Estimators", 10, 5000, 200, 10, f"{model_name}_n_estimators")
                    params["learning_rate"] = st.number_input("Learning rate", min_value=0.001, value=0.05, step=0.01, format="%.3f", key=f"{model_name}_learning_rate")
                    params["loss"] = st.selectbox("Loss", ["linear", "square", "exponential"], key=f"{model_name}_loss")
                    params["max_depth"] = st.slider("Weak tree max depth", 1, 20, 2, key=f"{model_name}_max_depth")
                    params["min_samples_split"] = st.slider("Min samples split", 2, 20, 2, key=f"{model_name}_min_samples_split")
                    params["min_samples_leaf"] = st.slider("Min samples leaf", 1, 20, 1, key=f"{model_name}_min_samples_leaf")
                    params["random_state"] = int(st.session_state.get("random_seed", 42))
                elif model_name == "GBR / Gradient Boosting":
                    st.caption("Gradient Boosting is often strong for nonlinear QSAR/QSPR patterns, but descriptor effects are ensemble-level rather than coefficient-level.")
                    params["n_estimators"] = synced_int_control("Estimators", 10, 5000, 300, 10, f"{model_name}_n_estimators")
                    params["learning_rate"] = st.number_input("Learning rate", min_value=0.001, value=0.05, step=0.01, format="%.3f", key=f"{model_name}_learning_rate")
                    params["max_depth"] = st.slider("Tree max depth", 1, 20, 3, key=f"{model_name}_max_depth")
                    params["min_samples_split"] = st.slider("Min samples split", 2, 20, 2, key=f"{model_name}_min_samples_split")
                    params["min_samples_leaf"] = st.slider("Min samples leaf", 1, 20, 1, key=f"{model_name}_min_samples_leaf")
                    params["subsample"] = st.slider("Subsample", 0.1, 1.0, 1.0, 0.05, key=f"{model_name}_subsample")
                    params["loss"] = st.selectbox("Loss", ["squared_error", "absolute_error", "huber"], key=f"{model_name}_loss")
                    params["max_features"] = st.selectbox("Max features", [None, "sqrt", "log2", 1.0], key=f"{model_name}_max_features")
                    params["random_state"] = int(st.session_state.get("random_seed", 42))
                model_params[model_name] = params

        st.session_state.model_config = {
            "scaler_name": scaler_name,
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
            "ranking_metric": ranking_metric,
            "selected_models": selected_models,
            "model_params": model_params,
        }


with tabs[5]:
    st.subheader("Feature selection")
    if show_dataset_gate():
        dataset = active_dataset()
        fs_method = st.selectbox(
            "Selection method",
            ["None", "Manual", "Variance threshold", "SelectKBest", "RFE", "Genetic Algorithm"],
            key="fs_method",
        )
        fs_params: dict[str, Any] = {}
        candidate_count = 1
        keep_top_n = max(1, len(st.session_state.get("selected_models", [])))
        if fs_method == "Manual":
            fs_params["manual_descriptors"] = st.multiselect(
                "Manual descriptors",
                dataset.X.columns.tolist(),
                default=dataset.X.columns[: min(10, dataset.X.shape[1])].tolist(),
                key="manual_descriptors",
            )
        elif fs_method == "Variance threshold":
            fs_params["threshold"] = st.number_input("Selection variance threshold", min_value=0.0, value=0.01, step=0.01, key="fs_variance_threshold")
        elif fs_method == "SelectKBest":
            fs_params["k"] = synced_int_control(
                "Number of descriptors",
                1,
                max(1, dataset.X.shape[1]),
                min(10, dataset.X.shape[1]),
                1,
                "select_k_best_k",
            )
        elif fs_method == "RFE":
            fs_params["n_features"] = synced_int_control(
                "Descriptors to keep",
                1,
                max(1, dataset.X.shape[1]),
                min(10, dataset.X.shape[1]),
                1,
                "rfe_n_features",
            )
        elif fs_method == "Genetic Algorithm":
            st.info("GA can score each descriptor subset with internal CV, or skip GA CV for a much faster training-score search.")
            use_ga_cv = st.checkbox(
                "Use cross-validation inside GA fitness",
                value=True,
                key="use_ga_cv",
                help="Disable this when you want GA to search quickly and leave CV only for the final trained model metrics.",
            )
            g1, g2, g3 = st.columns(3)
            with g1:
                fs_params["population_size"] = synced_int_control("Population size", 6, 500, 30, 2, "ga_population_size")
                fs_params["generations"] = synced_int_control("Generations", 1, 500, 20, 1, "ga_generations")
                if use_ga_cv:
                    fs_params["cv_folds"] = synced_int_control("GA CV folds", 2, 10, 5, 1, "ga_cv_folds")
                else:
                    fs_params["cv_folds"] = 0
                    st.warning("GA CV is disabled. GA will optimize training-set score only, so final CV/test metrics are the real check for overfitting.")
            with g2:
                fs_params["crossover_probability"] = st.slider("Crossover probability", 0.0, 1.0, 0.8, 0.05, key="ga_crossover_probability")
                fs_params["mutation_probability"] = st.slider("Mutation probability", 0.0, 0.5, 0.05, 0.01, key="ga_mutation_probability")
                fs_params["tournament_size"] = st.slider("Tournament size", 2, 10, 3, key="ga_tournament_size")
                fs_params["early_stopping_rounds"] = synced_int_control("Early stopping generations", 0, 100, 10, 1, "ga_early_stopping_rounds")
            with g3:
                fs_params["min_descriptors"] = synced_int_control(
                    "Minimum descriptors",
                    1,
                    max(1, dataset.X.shape[1]),
                    min(2, max(1, dataset.X.shape[1])),
                    1,
                    "ga_min_descriptors",
                )
                fs_params["max_descriptors"] = synced_int_control(
                    "Maximum descriptors",
                    fs_params["min_descriptors"],
                    max(1, dataset.X.shape[1]),
                    min(max(fs_params["min_descriptors"], 20), dataset.X.shape[1]),
                    1,
                    "ga_max_descriptors",
                )
                fs_params["random_seed"] = st.number_input("GA random seed", min_value=0, value=42, step=1, key="ga_random_seed")
            fs_params["scoring_metric"] = st.selectbox(
                "GA scoring metric",
                ["Q2 / CV R2", "R2", "RMSE", "MAE", "Combined score"],
                key="ga_scoring_metric",
            )
            st.markdown("#### Candidate model search")
            c1, c2 = st.columns(2)
            with c1:
                candidate_count = synced_int_control(
                    "GA descriptor subsets to build",
                    1,
                    500,
                    20,
                    1,
                    "ga_candidate_count",
                )
            with c2:
                total_candidate_models = candidate_count * max(1, len(st.session_state.get("selected_models", [])))
                keep_top_n = synced_int_control(
                    "Keep best trained models",
                    1,
                    5000,
                    min(50, max(1, total_candidate_models)),
                    1,
                    "ga_keep_top_n",
                )
            st.caption("Each candidate uses the same train/test split, but GA receives a different seed and may select a different descriptor subset.")
        st.session_state.feature_selection = {
            "method": fs_method,
            "params": fs_params,
            "candidate_count": candidate_count,
            "keep_top_n": keep_top_n,
        }


with tabs[6]:
    st.subheader("Training and validation")
    if show_dataset_gate():
        ready = all(
            key in st.session_state
            for key in ["preprocessing_config", "split_config", "model_config", "feature_selection"]
        )
        if not ready:
            st.info("Complete preprocessing, split, model, and feature-selection controls before training.")
        else:
            config = st.session_state.model_config
            summary_cols = st.columns(6)
            selector_candidate_count = int(st.session_state.feature_selection.get("candidate_count", 1))
            total_candidates = len(config["selected_models"]) * selector_candidate_count
            with summary_cols[0]:
                metric_panel("Models", len(config["selected_models"]))
            with summary_cols[1]:
                metric_panel("Descriptor candidates", selector_candidate_count)
            with summary_cols[2]:
                metric_panel("Trained candidates", total_candidates)
            with summary_cols[3]:
                metric_panel("Keep top", st.session_state.feature_selection.get("keep_top_n", total_candidates))
            with summary_cols[4]:
                metric_panel("CV folds", config["cv_folds"])
            with summary_cols[5]:
                metric_panel("Scaler", config["scaler_name"])

            fs_cols = st.columns(2)
            with fs_cols[0]:
                metric_panel("Feature selection", st.session_state.feature_selection["method"])
            with fs_cols[1]:
                if st.session_state.feature_selection["method"] == "Genetic Algorithm":
                    ga_folds = int(st.session_state.feature_selection.get("params", {}).get("cv_folds", 5))
                    metric_panel("GA fitness", f"{ga_folds}-fold CV" if ga_folds >= 2 else "Training score")
                else:
                    metric_panel("GA fitness", "n/a")

            if st.session_state.feature_selection["method"] == "Genetic Algorithm":
                ga_params = st.session_state.feature_selection.get("params", {})
                ga_population = int(ga_params.get("population_size", 30))
                ga_generations = int(ga_params.get("generations", 20))
                ga_folds = int(ga_params.get("cv_folds", 5))
                folds_per_subset = max(1, ga_folds if ga_folds >= 2 else 1)
                approx_ga_subsets = total_candidates * ga_population * (ga_generations + 1)
                approx_estimator_fits = approx_ga_subsets * folds_per_subset
                st.caption(
                    "Approximate GA workload before cache/early stopping: "
                    f"about {approx_ga_subsets:,} descriptor-subset evaluations and "
                    f"up to {approx_estimator_fits:,} estimator fits. Live elapsed time and ETA appear after training starts."
                )
            else:
                st.caption("Live elapsed time and ETA appear after training starts.")

            if st.button("Run training workflow", type="primary"):
                try:
                    results, table, warnings = run_training_workflow(
                        active_dataset(),
                        st.session_state.get("endpoint_method", "none"),
                        st.session_state.split_config,
                        st.session_state.preprocessing_config,
                        st.session_state.drop_rows_after_split,
                        config["selected_models"],
                        config["model_params"],
                        config["scaler_name"],
                        st.session_state.feature_selection,
                        config["cv_folds"],
                        config["cv_repeats"],
                        config["ranking_metric"],
                        excluded_samples_frame(),
                    )
                    for label, payload in results.items():
                        payload["figures"] = create_figures_for_result(label, payload)
                    st.session_state.training_results = results
                    st.session_state.results_df = table
                    st.session_state.last_run_warnings = warnings
                    st.success("Training and validation completed.")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        if not st.session_state.results_df.empty:
            st.dataframe(st.session_state.results_df, use_container_width=True)
            display_messages(st.session_state.last_run_warnings)


with tabs[7]:
    st.subheader("Results and plots")
    if not st.session_state.training_results:
        st.info("Run training to populate model results and figures.")
    else:
        results = st.session_state.training_results
        table = st.session_state.results_df
        ranking_metric = st.session_state.model_config["ranking_metric"]
        best_label = table.iloc[0]["Model label"] if not table.empty else list(results.keys())[0]
        best_payload = results[best_label]

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            metric_panel("Best model", best_label)
        with b2:
            metric_panel(ranking_metric, f"{table.iloc[0][ranking_metric]:.4g}" if ranking_metric in table.columns else "n/a")
        with b3:
            metric_panel("Descriptors", len(best_payload["selected_descriptors"]))
        with b4:
            outside_count = int(best_payload["williams_ad"]["outside_ad"].sum())
            metric_panel("Williams AD flags", outside_count)

        display_messages(best_payload.get("warnings", []))
        st.dataframe(table, use_container_width=True)
        st.pyplot(model_comparison_plot(table, ranking_metric))

        selected_label = st.selectbox("Model result", list(results.keys()), index=list(results.keys()).index(best_label))
        payload = results[selected_label]
        eval_result = payload["evaluation"]

        metric_cols = st.columns(6)
        key_metrics = ["R2 train", "R2 test", "Q2 CV", "RMSE train", "RMSE test", "RMSE CV"]
        for col, metric in zip(metric_cols, key_metrics):
            with col:
                value = eval_result.metrics.get(metric, np.nan)
                metric_panel(metric, f"{value:.4g}" if pd.notna(value) else "n/a")

        st.markdown("#### Descriptor interpretation")
        importance_frame = payload_descriptor_importance(payload)
        model_equation, equation_terms = payload_mlr_equation(payload)
        if model_equation:
            st.caption("MLR equation using coefficients converted back to the original descriptor scale after the selected scaler.")
            st.code(model_equation, language="text")
            if isinstance(equation_terms, pd.DataFrame) and not equation_terms.empty:
                with st.expander("MLR equation coefficients", expanded=False):
                    st.dataframe(equation_terms, use_container_width=True)
        if not importance_frame.empty:
            importance_fig = payload["figures"].get("Descriptor importance")
            if importance_fig is None:
                importance_fig = descriptor_importance_plot(importance_frame, title=f"{selected_label}: descriptor importance")
            st.pyplot(importance_fig)
            formatters = {
                column: "{:.6g}"
                for column in ["coefficient", "importance", "abs_importance"]
                if column in importance_frame.columns
            }
            st.dataframe(importance_frame.style.format(formatters), use_container_width=True)
        else:
            st.info("Native descriptor importance is not available for this model. For SVR, use a linear kernel or compare with linear/tree models for descriptor-level interpretation.")

        st.markdown("#### Observed vs predicted")
        st.plotly_chart(
            interactive_observed_vs_predicted(
                eval_result.train_predictions,
                eval_result.test_predictions,
                f"{selected_label}: observed vs predicted",
            ),
            use_container_width=True,
        )

        st.markdown("#### Predictions")
        pred_tabs = st.tabs(["Train", "Test", "CV"])
        with pred_tabs[0]:
            st.dataframe(eval_result.train_predictions, use_container_width=True)
        with pred_tabs[1]:
            st.dataframe(eval_result.test_predictions, use_container_width=True)
        with pred_tabs[2]:
            st.dataframe(eval_result.cv_predictions, use_container_width=True)

        st.markdown("#### Applicability Domain")
        distance_fig = payload["figures"].get("Applicability Domain - Distance plot") or payload["figures"].get("Distance AD")
        pca_ad_fig = payload["figures"].get("Applicability Domain - PCA plot") or payload["figures"].get("PCA AD")
        ad_tabs = st.tabs(["Williams plot", "Distance plot", "PCA space"])
        with ad_tabs[0]:
            williams_event = st.plotly_chart(
                interactive_williams_plot(payload["williams_ad"]),
                use_container_width=True,
                key=f"williams_plot_{selected_label}",
                on_select="rerun",
                selection_mode="points",
            )
            selected_williams_id = sample_id_from_plotly_selection(williams_event)
            williams_smiles = None
            if "smiles" in payload["williams_ad"].columns:
                williams_smiles = (
                    payload["williams_ad"]
                    .drop_duplicates("sample_id")
                    .set_index("sample_id")["smiles"]
                )
            render_structure_panel(
                williams_smiles,
                f"williams_{selected_label}",
                selected_sample_id=selected_williams_id,
                sample_ids=payload["williams_ad"]["sample_id"].astype(str).tolist(),
            )
            st.dataframe(payload["williams_ad"], use_container_width=True)
        with ad_tabs[1]:
            if distance_fig is not None:
                st.pyplot(distance_fig)
            else:
                st.info("Rerun training to regenerate the distance applicability-domain plot.")
            st.dataframe(payload["distance_ad"], use_container_width=True)
        with ad_tabs[2]:
            if pca_ad_fig is not None:
                st.pyplot(pca_ad_fig)
            else:
                st.info("PCA applicability-domain plot needs at least two selected descriptors.")

        st.markdown("#### Other plots")
        plot_names = [
            name
            for name in payload["figures"].keys()
            if not name.startswith("Applicability Domain") and name not in {"Observed vs predicted", "Descriptor importance"}
        ]
        if plot_names:
            selected_plot = st.selectbox("Plot", plot_names)
            st.pyplot(payload["figures"][selected_plot])

        with st.expander("Applicability domain tables", expanded=False):
            st.caption("Williams plot results")
            st.dataframe(payload["williams_ad"], use_container_width=True)
            st.caption("Distance-domain results")
            st.dataframe(payload["distance_ad"], use_container_width=True)

        with st.expander("Selected descriptors", expanded=False):
            st.dataframe(list_to_frame(payload["selected_descriptors"]), use_container_width=True)

        with st.expander("Descriptor correlation heatmap after preprocessing", expanded=False):
            st.pyplot(correlation_heatmap(payload["X_train_selected"]))


with tabs[8]:
    st.subheader("Export")
    if not st.session_state.training_results:
        st.info("Run training before exporting models, reports, and plots.")
    else:
        results = st.session_state.training_results
        selected_label = st.selectbox("Export model", list(results.keys()), key="export_model_label")
        payload = results[selected_label]
        export_stem = safe_file_stem(selected_label)
        report_sheets = build_report_sheets(selected_label, payload, st.session_state.results_df)
        run_bundle = build_model_run_bundle(
            results,
            st.session_state.results_df,
            st.session_state.model_config["ranking_metric"],
        )

        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "Download Excel report",
                data=dataframes_to_excel_bytes(report_sheets),
                file_name=f"{export_stem}_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            predictions = pd.concat([payload["evaluation"].train_predictions, payload["evaluation"].test_predictions], axis=0)
            st.download_button(
                "Download predictions CSV",
                data=dataframe_to_csv_bytes(predictions.reset_index(drop=True)),
                file_name=f"{export_stem}_predictions.csv",
                mime="text/csv",
            )
        with e2:
            st.download_button(
                "Download selected descriptors CSV",
                data=dataframe_to_csv_bytes(list_to_frame(payload["selected_descriptors"])),
                file_name=f"{export_stem}_selected_descriptors.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download removed descriptors CSV",
                data=dataframe_to_csv_bytes(removed_descriptors_frame(payload["preprocessor"])),
                file_name=f"{export_stem}_removed_descriptors.csv",
                mime="text/csv",
            )
            importance_export = payload_descriptor_importance(payload)
            if not importance_export.empty:
                st.download_button(
                    "Download descriptor importance CSV",
                    data=dataframe_to_csv_bytes(importance_export),
                    file_name=f"{export_stem}_descriptor_importance.csv",
                    mime="text/csv",
                )
            equation_export, _ = payload_mlr_equation(payload)
            if equation_export:
                st.download_button(
                    "Download MLR equation TXT",
                    data=equation_export.encode("utf-8"),
                    file_name=f"{export_stem}_mlr_equation.txt",
                    mime="text/plain",
                )
            selected_model_bundle = build_selected_model_bundle(selected_label, payload, st.session_state.results_df)
            st.download_button(
                "Download selected model bundle",
                data=bundle_to_bytes(selected_model_bundle),
                file_name=f"{export_stem}_model.joblib",
                mime="application/octet-stream",
                help="Contains this model plus the run snapshot needed to restore Results and PCA context.",
            )
            st.download_button(
                "Download full run bundle",
                data=bundle_to_bytes(run_bundle),
                file_name=f"{safe_file_stem(run_bundle.run_label)}.joblib",
                mime="application/octet-stream",
                help="Contains all kept models from this run, their preprocessing, selected descriptors, metadata, and ranking table.",
            )
        with e3:
            image_format = st.selectbox("Plot image format", ["png", "jpg"])
            st.download_button(
                "Download plots ZIP",
                data=figures_to_zip_bytes(payload["figures"], fmt=image_format),
                file_name=f"{export_stem}_plots.zip",
                mime="application/zip",
            )
            single_plot = st.selectbox("Single plot", list(payload["figures"].keys()))
            st.download_button(
                "Download selected plot",
                data=fig_to_bytes(payload["figures"][single_plot], fmt=image_format),
                file_name=f"{safe_file_stem(single_plot.lower())}.{image_format}",
                mime=f"image/{'jpeg' if image_format == 'jpg' else 'png'}",
            )
