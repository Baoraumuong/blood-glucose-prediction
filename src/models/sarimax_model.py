"""
Patient-wise SARIMAX forecasting with KFold CV and saved patient artifacts.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.metrics import ResultStore, compute_all_metrics
from src.models.persistence import safe_name, save_pickle_artifact
from src.models.time_series_utils import (
    fill_series_gaps,
    is_finite_window,
    normalize_series_and_exog,
    ensure_supported_series,
)

logger = logging.getLogger(__name__)


def _feature_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    excluded = set(cfg.get("feature_engineering", {}).get("exclude_features", []))
    return [
        c
        for c in df.columns
        if c not in {"raw_glucose_level", "glucose_level"} and c not in excluded
    ]


def _scale_exog(
    train_exog: pd.DataFrame,
    test_exog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Standardize exogenous variables with train statistics only."""
    train = train_exog.astype(float).ffill().fillna(0)
    test = test_exog.astype(float).ffill().fillna(0)
    train.attrs = {}
    test.attrs = {}
    mean = train.mean()
    std = train.std().replace(0, 1).fillna(1)
    return (train - mean) / std, (test - mean) / std, mean, std


def _fit_sarimax(
    train_series: pd.Series,
    train_exog: pd.DataFrame,
    order: tuple[int, int, int],
) -> tuple[Any | None, pd.Series | None, pd.DataFrame | None, pd.Series | None, pd.Series | None]:
    train_model = fill_series_gaps(train_series)
    train_model, train_exog = normalize_series_and_exog(train_model, train_exog)
    if train_model.empty:
        return None, None, None, None, None

    train_smooth = train_model.ewm(span=5, adjust=False).mean()
    train_x, _, mean, std = _scale_exog(train_exog, train_exog)
    try:
        fit = SARIMAX(
            train_smooth,
            exog=train_x,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            missing="none",
        ).fit(disp=False)
    except Exception as exc:
        logger.warning("SARIMAX fit failed: %s", exc)
        return None, None, None, None, None
    return fit, train_smooth, train_x, mean, std


def _transform_exog(exog: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    clean = exog.astype(float).ffill().fillna(0)
    return (clean - mean) / std


def _forecast_windows(
    fit: Any,
    train_smooth: pd.Series,
    train_x: pd.DataFrame,
    test_series: pd.Series,
    test_exog: pd.DataFrame,
    exog_mean: pd.Series,
    exog_std: pd.Series,
    n_forecast: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    test_model = fill_series_gaps(test_series)
    test_model, test_exog = normalize_series_and_exog(test_model, test_exog)
    if test_model.empty:
        return None, None

    test_x = _transform_exog(test_exog, exog_mean, exog_std)
    all_model_series = pd.concat([train_smooth, test_model])
    all_actual_series = pd.concat([train_smooth, pd.to_numeric(test_series, errors="coerce")])
    all_exog = pd.concat([train_x, test_x])
    preds_list, actuals_list = [], []
    n_train = len(train_smooth)
    n_test = min(len(test_model), len(test_series))

    for i in range(0, n_test - n_forecast + 1, n_forecast):
        start = n_train + i
        end = start + n_forecast
        actual = all_actual_series.iloc[start:end].values
        future_exog = all_exog.iloc[start:end]
        if len(actual) < n_forecast or len(future_exog) < n_forecast:
            break
        if not is_finite_window(actual):
            logger.debug("SARIMAX window %d skipped because actuals contain NaN/inf", i)
            continue
        try:
            history = all_model_series.iloc[:start]
            history_exog = all_exog.iloc[:start]
            if isinstance(history.index, pd.DatetimeIndex) and history.index.freq is None:
                history = history.copy()
                history_exog = history_exog.copy()
                future_exog = future_exog.copy()
                history.index = pd.RangeIndex(len(history))
                history_exog.index = pd.RangeIndex(len(history_exog))
                future_exog.index = pd.RangeIndex(len(future_exog))
            fc = fit.apply(history, exog=history_exog, refit=False).forecast(
                n_forecast,
                exog=future_exog,
            )
            if not is_finite_window(fc.values):
                logger.debug("SARIMAX window %d skipped because predictions contain NaN/inf", i)
                continue
            preds_list.append(fc.values)
            actuals_list.append(actual)
        except Exception as exc:
            logger.debug("SARIMAX window %d failed: %s", i, exc)

    if not preds_list:
        return None, None
    return np.array(preds_list), np.array(actuals_list)


def _sarimax_patient_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    train_exog: pd.DataFrame,
    test_exog: pd.DataFrame,
    order: tuple[int, int, int],
    n_forecast: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    fit, train_smooth, train_x, mean, std = _fit_sarimax(train_series, train_exog, order)
    if fit is None or train_smooth is None or train_x is None or mean is None or std is None:
        return None, None, None

    preds, actuals = _forecast_windows(
        fit,
        train_smooth,
        train_x,
        test_series,
        test_exog,
        mean,
        std,
        n_forecast,
    )
    artifact = {
        "fit": fit,
        "train_smooth": train_smooth,
        "train_x": train_x,
        "exog_mean": mean,
        "exog_std": std,
        "order": order,
    }
    return preds, actuals, artifact


def _sarimax_cv_metrics(
    train_series: pd.Series,
    train_exog: pd.DataFrame,
    order: tuple[int, int, int],
    n_forecast: int,
    n_folds: int,
    shuffle: bool,
) -> dict[str, float] | None:
    clean = fill_series_gaps(train_series)
    if len(clean) < n_forecast * 2:
        return None

    n_splits = min(n_folds, len(clean))
    if n_splits < 2:
        return None

    fold_metrics: list[dict[str, float]] = []
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(clean), start=1):
        tr_idx = np.sort(tr_idx)
        val_idx = np.sort(val_idx)
        fold_train = clean.iloc[tr_idx]
        fold_val = clean.iloc[val_idx]
        fold_train = ensure_supported_series(fold_train)
        fold_val = ensure_supported_series(fold_val)
        if len(fold_val) < n_forecast or len(fold_train) < n_forecast:
            continue
        fold_train_exog = train_exog.reindex(clean.index).iloc[tr_idx]
        fold_val_exog = train_exog.reindex(clean.index).iloc[val_idx]
        preds, actuals, _ = _sarimax_patient_forecast(
            fold_train,
            fold_val,
            fold_train_exog,
            fold_val_exog,
            order,
            n_forecast,
        )
        if preds is None or actuals is None:
            continue
        metrics = compute_all_metrics(actuals, preds)
        fold_metrics.append(metrics)
        logger.debug("  SARIMAX CV fold %d -> RMSE=%.3f R2=%.4f", fold_idx, metrics["rmse"], metrics["r2"])

    if not fold_metrics:
        return None
    return {
        f"cv_{metric}": float(np.mean([m[metric] for m in fold_metrics]))
        for metric in fold_metrics[0]
    }


def run_sarimax(
    train_masters: dict[str, pd.DataFrame],
    test_masters: dict[str, pd.DataFrame],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    """Run patient-wise SARIMAX with KFold CV on training data and hold-out test evaluation."""
    model_cfg = cfg["models"].get("sarimax", cfg["models"].get("arima", {}))
    eval_cfg = cfg.get("evaluation", {})
    order = tuple(model_cfg.get("order", [2, 1, 2]))
    n_folds = eval_cfg.get("n_folds", 5)
    shuffle = eval_cfg.get("kfold_shuffle", False)
    w = cfg["window"]
    n_forecast = w["forecast_minutes"] // w["freq_minutes"]

    for input_col, smoothing_label in [
        ("glucose_level", "Savitzky-Golay"),
        ("raw_glucose_level", "no Savitzky-Golay"),
    ]:
        all_preds: list[np.ndarray] = []
        test_metrics_by_patient: list[dict[str, float]] = []
        cv_metrics_by_patient: list[dict[str, float]] = []
        feature_tag = f"with features + {smoothing_label}"

        for key in sorted(train_masters.keys()):
            if key not in test_masters:
                logger.warning("SARIMAX: test data missing for %s, skipping", key)
                continue

            train_df = train_masters[key]
            test_df = test_masters[key]
            exog_cols = _feature_columns(train_df, cfg)
            exog_cols = [c for c in exog_cols if c in test_df.columns]
            if not exog_cols:
                logger.warning("SARIMAX: no exogenous columns for %s, skipping", key)
                continue
            if input_col not in train_df.columns:
                logger.warning("SARIMAX: %s missing for %s, skipping", input_col, key)
                continue

            tr_ser = train_df[input_col]
            te_ser = test_df.get("raw_glucose_level", test_df["glucose_level"])
            cv_metrics = _sarimax_cv_metrics(
                tr_ser,
                train_df[exog_cols],
                order,
                n_forecast,
                n_folds,
                shuffle,
            )
            if cv_metrics is not None:
                cv_metrics_by_patient.append(cv_metrics)

            preds, actuals, artifact = _sarimax_patient_forecast(
                tr_ser,
                te_ser,
                train_df[exog_cols],
                test_df[exog_cols],
                order,
                n_forecast,
            )
            if preds is None or actuals is None or artifact is None:
                continue

            test_metrics = compute_all_metrics(actuals, preds)
            test_metrics_by_patient.append(test_metrics)
            all_preds.append(preds)
            save_path = save_pickle_artifact(
                cfg,
                "SARIMAX",
                feature_tag,
                f"{safe_name(key)}.pkl",
                {
                    **artifact,
                    "patient_key": key,
                    "model_name": "SARIMAX",
                    "feature_tag": feature_tag,
                    "input_col": input_col,
                    "exog_cols": exog_cols,
                    "n_forecast": n_forecast,
                },
            )
            logger.info("Saved SARIMAX (%s, %s) artifact: %s", key, smoothing_label, save_path)
            logger.info(
                "  SARIMAX %s (%s, %d exog) -> RMSE=%.3f R2=%.4f",
                key,
                smoothing_label,
                len(exog_cols),
                test_metrics["rmse"],
                test_metrics["r2"],
            )

        if not test_metrics_by_patient:
            logger.error("SARIMAX produced no results for %s.", smoothing_label)
            continue

        avg_test = {
            k: float(np.mean([m[k] for m in test_metrics_by_patient]))
            for k in test_metrics_by_patient[0]
        }
        avg_cv = (
            {
                k: float(np.mean([m[k] for m in cv_metrics_by_patient]))
                for k in cv_metrics_by_patient[0]
            }
            if cv_metrics_by_patient
            else {f"cv_{k}": float("nan") for k in avg_test}
        )

        store.add("SARIMAX", feature_tag, avg_cv, avg_test, np.vstack(all_preds))
        logger.info(
            "SARIMAX avg (%s) -> CV RMSE=%.3f | Test RMSE=%.3f R2=%.4f",
            smoothing_label,
            avg_cv.get("cv_rmse", float("nan")),
            avg_test["rmse"],
            avg_test["r2"],
        )
