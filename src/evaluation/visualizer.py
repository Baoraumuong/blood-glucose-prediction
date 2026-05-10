"""
evaluation/visualizer.py
------------------------
All result-comparison plots, mirroring the figures in the original notebook
but as reusable functions that save to disk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["figure.dpi"] = 120


def _save(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# EDA plots (called from EDA notebook helper functions)
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing_heatmap(
    missing_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    pivot = missing_df.pivot(index="patient", columns="feature", values="missing_pct")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Missing %"})
    ax.set_title("Missing Data (%) per Patient & Feature", fontsize=14, pad=12)
    if save_path:
        _save(fig, save_path)
    return fig


def plot_raw_glucose(
    train_files: dict[str, str],
    save_path: str | None = None,
) -> plt.Figure:
    from src.data.parser import OhioT1DMParser

    n = len(train_files)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (key, path) in zip(axes, sorted(train_files.items())):
        df = OhioT1DMParser(path, "glucose_level").make_dataframe()
        ax.plot(df.index, df["value"], lw=0.8, color="royalblue")
        ax.axhline(70,  ls="--", color="red",    lw=0.8, alpha=0.7, label="Hypo (70)")
        ax.axhline(180, ls="--", color="orange", lw=0.8, alpha=0.7, label="Hyper (180)")
        ax.set_ylabel("mg/dL", fontsize=9)
        ax.set_title(f"Patient {key} – Training CGM", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.suptitle("Raw Glucose Levels – All Training Patients", y=1.01, fontsize=13)
    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Result comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_rmse_r2_bars(
    res_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Grouped bar charts: CV RMSE, Test RMSE, CV R², Test R²."""
    model_names = res_df["model"].unique()
    x     = np.arange(len(model_names))
    width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for ax, metric, ylabel, title, cmap in [
        (axes[0, 0], "cv_rmse",   "RMSE (mg/dL)", "CV RMSE",   ("steelblue", "tomato")),
        (axes[0, 1], "test_rmse", "RMSE (mg/dL)", "Test RMSE", ("steelblue", "tomato")),
        (axes[1, 0], "cv_r2",     "R²",            "CV R²",     ("seagreen",  "salmon")),
        (axes[1, 1], "test_r2",   "R²",            "Test R²",   ("seagreen",  "salmon")),
    ]:
        sub_f = res_df[res_df.features == "with features"].set_index("model")
        sub_b = res_df[res_df.features == "baseline"].set_index("model")
        vals_f = [sub_f.loc[m, metric] if m in sub_f.index else 0 for m in model_names]
        vals_b = [sub_b.loc[m, metric] if m in sub_b.index else 0 for m in model_names]
        ax.bar(x - width / 2, vals_f, width, label="With Features", color=cmap[0], alpha=0.85)
        ax.bar(x + width / 2, vals_b, width, label="Baseline (GL only)", color=cmap[1], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle("Model Performance Comparison", fontsize=14, y=1.01)
    if save_path:
        _save(fig, save_path)
    return fig


def plot_feature_impact(
    res_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Δ RMSE and Δ R² bars (with features − baseline)."""
    records = []
    for m in res_df["model"].unique():
        sf = res_df[(res_df.model == m) & (res_df.features == "with features")]
        sb = res_df[(res_df.model == m) & (res_df.features == "baseline")]
        if sf.empty or sb.empty:
            continue
        records.append(dict(
            model=m,
            delta_rmse=sf["test_rmse"].iloc[0] - sb["test_rmse"].iloc[0],
            delta_r2=  sf["test_r2"].iloc[0]   - sb["test_r2"].iloc[0],
        ))
    delta_df = pd.DataFrame(records).sort_values("delta_rmse")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    c_rmse = ["seagreen" if v < 0 else "tomato" for v in delta_df["delta_rmse"]]
    c_r2   = ["seagreen" if v > 0 else "tomato" for v in delta_df["delta_r2"]]
    axes[0].barh(delta_df["model"], delta_df["delta_rmse"], color=c_rmse, alpha=0.85)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("ΔRMSE  (with features − baseline)\nnegative = features help")
    axes[0].set_title("Impact on Test RMSE")
    axes[1].barh(delta_df["model"], delta_df["delta_r2"], color=c_r2, alpha=0.85)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("ΔR²  (with features − baseline)\npositive = features help")
    axes[1].set_title("Impact on Test R²")
    fig.suptitle("Feature Impact Analysis", fontsize=13, y=1.02)
    if save_path:
        _save(fig, save_path)
    return fig


def plot_best_model_predictions(
    res_df: pd.DataFrame,
    result_store: Any,
    y_test: np.ndarray,
    n_show: int = 500,
    save_path: str | None = None,
) -> plt.Figure:
    """Time-series and scatter plot for the best model on the 30-min horizon."""
    best = res_df[res_df.features == "with features"].sort_values("test_rmse").iloc[0]
    name = best["model"]
    preds = result_store.get_predictions(name, "with features")
    if preds is None:
        logger.warning("No predictions found for %s", name)
        return plt.figure()

    n = min(n_show, len(y_test))
    actuals = y_test[:n, -1]
    pred_1d = preds[:n, -1] if preds.ndim > 1 else preds[:n]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(actuals, label="Actual",    lw=1.2, color="royalblue")
    axes[0].plot(pred_1d, label="Predicted", lw=1.2, color="tomato", alpha=0.8)
    axes[0].set_title(f"{name} – 30-min Ahead (first {n} windows)")
    axes[0].set_ylabel("Glucose (mg/dL)")
    axes[0].legend()

    axes[1].scatter(actuals, pred_1d, alpha=0.3, s=8, color="steelblue")
    lims = [min(actuals.min(), pred_1d.min()), max(actuals.max(), pred_1d.max())]
    axes[1].plot(lims, lims, "k--", lw=1)
    axes[1].set_xlabel("Actual (mg/dL)")
    axes[1].set_ylabel("Predicted (mg/dL)")
    axes[1].set_title(f"{name} – Scatter")

    fig.suptitle(
        f"Best Model: {name}  |  RMSE={best['test_rmse']:.2f}  R²={best['test_r2']:.3f}",
        fontsize=12, y=1.02,
    )
    if save_path:
        _save(fig, save_path)
    return fig


def plot_horizon_rmse(
    res_df: pd.DataFrame,
    result_store: Any,
    y_test: np.ndarray,
    forecast_steps: int = 6,
    freq_minutes: int = 5,
    save_path: str | None = None,
) -> plt.Figure:
    """Per-step RMSE for LSTM and GRU across forecast horizon."""
    from sklearn.metrics import mean_squared_error

    labels = [f"+{(i + 1) * freq_minutes} min" for i in range(forecast_steps)]
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_name in ["LSTM", "GRU"]:
        for tag in ["with features", "baseline"]:
            preds = result_store.get_predictions(model_name, tag)
            if preds is None:
                continue
            rmse_steps = [
                np.sqrt(mean_squared_error(y_test[:, s], preds[: len(y_test), s]))
                for s in range(forecast_steps)
            ]
            ls = "-" if tag == "with features" else "--"
            ax.plot(labels, rmse_steps, marker="o", ls=ls, label=f"{model_name} ({tag})")

    ax.set_title("RMSE vs Forecast Horizon – LSTM & GRU", fontsize=12)
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_xlabel("Forecast Horizon")
    ax.legend(fontsize=9)
    if save_path:
        _save(fig, save_path)
    return fig


def plot_summary_heatmap(
    res_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmaps of Test RMSE and Test R² for all model × feature-set combinations."""
    pivot_rmse = res_df.pivot(index="model", columns="features", values="test_rmse")
    pivot_r2   = res_df.pivot(index="model", columns="features", values="test_r2")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(pivot_rmse, annot=True, fmt=".2f", cmap="YlOrRd_r",
                linewidths=0.5, ax=axes[0], cbar_kws={"label": "RMSE (mg/dL)"})
    axes[0].set_title("Test RMSE (lower = better)", fontsize=11)

    sns.heatmap(pivot_r2, annot=True, fmt=".3f", cmap="YlGn",
                linewidths=0.5, ax=axes[1], cbar_kws={"label": "R²"})
    axes[1].set_title("Test R² (higher = better)", fontsize=11)

    fig.suptitle("Summary – All Models & Feature Sets", fontsize=13, y=1.02)
    if save_path:
        _save(fig, save_path)
    return fig
