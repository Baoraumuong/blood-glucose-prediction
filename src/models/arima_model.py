"""
Patient-wise ARIMA forecasting with KFold CV and saved patient artifacts.
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
    ensure_monotonic_series,
    ensure_supported_series,
)

logger = logging.getLogger(__name__)


def _fit_arima(
    train_series: pd.Series,
    order: tuple[int, int, int],
) -> tuple[Any | None, pd.Series | None]:
    train_model = fill_series_gaps(train_series)
    if train_model.empty:
        return None, None

    try:
        fit = SARIMAX(
            train_model,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            missing="none",
        ).fit(disp=False)
    except Exception as exc:
        logger.warning("ARIMA fit failed: %s", exc)
        return None, None
    return fit, train_model


def _forecast_windows(
    fit: Any,
    train_history: pd.Series,
    test_series: pd.Series,
    n_forecast: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    test_model = fill_series_gaps(test_series)
    if test_model.empty:
        return None, None

    all_model_series = pd.concat([train_history, test_model])
    all_actual_series = pd.concat([train_history, pd.to_numeric(test_series, errors="coerce")])
    preds_list, actuals_list = [], []
    n_train = len(train_history)
    n_test = min(len(test_model), len(test_series))

    for i in range(0, n_test - n_forecast + 1, n_forecast):
        start = n_train + i
        actual = all_actual_series.iloc[start : start + n_forecast].values
        if len(actual) < n_forecast:
            break
        if not is_finite_window(actual):
            logger.debug("ARIMA window %d skipped because actuals contain NaN/inf", i)
            continue
        try:
            history = all_model_series.iloc[:start]
            if isinstance(history.index, pd.DatetimeIndex) and history.index.freq is None:
                history = history.copy()
                history.index = pd.RangeIndex(len(history))
            fc = fit.apply(history, refit=False).forecast(n_forecast)
            if not is_finite_window(fc.values):
                logger.debug("ARIMA window %d skipped because predictions contain NaN/inf", i)
                continue
            preds_list.append(fc.values)
            actuals_list.append(actual)
        except Exception as exc:
            logger.debug("ARIMA window %d failed: %s", i, exc)

    if not preds_list:
        return None, None
    return np.array(preds_list), np.array(actuals_list)


def _arima_patient_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    n_forecast: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    fit, train_history = _fit_arima(train_series, order)
    if fit is None or train_history is None:
        return None, None, None

    preds, actuals = _forecast_windows(fit, train_history, test_series, n_forecast)
    artifact = {"fit": fit, "train_history": train_history, "order": order}
    return preds, actuals, artifact


def _arima_cv_metrics(
    train_series: pd.Series,
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
        fold_train = clean.iloc[np.sort(tr_idx)]
        fold_val = clean.iloc[np.sort(val_idx)]
        fold_train = ensure_supported_series(fold_train)
        fold_val = ensure_supported_series(fold_val)
        if len(fold_val) < n_forecast or len(fold_train) < n_forecast:
            continue
        fit, train_history = _fit_arima(fold_train, order)
        if fit is None or train_history is None:
            continue
        preds, actuals = _forecast_windows(fit, train_history, fold_val, n_forecast)
        if preds is None or actuals is None:
            continue
        metrics = compute_all_metrics(actuals, preds)
        fold_metrics.append(metrics)
        logger.debug("  ARIMA CV fold %d -> RMSE=%.3f R2=%.4f", fold_idx, metrics["rmse"], metrics["r2"])

    if not fold_metrics:
        return None
    return {
        f"cv_{metric}": float(np.mean([m[metric] for m in fold_metrics]))
        for metric in fold_metrics[0]
    }


def run_arima(
    train_masters: dict[str, pd.DataFrame],
    test_masters: dict[str, pd.DataFrame],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    """Run patient-wise ARIMA with KFold CV on training data and hold-out test evaluation."""
    model_cfg = cfg["models"].get("arima", {})
    eval_cfg = cfg.get("evaluation", {})
    order = tuple(model_cfg.get("order", [2, 1, 2]))
    n_folds = eval_cfg.get("n_folds", 5)
    shuffle = eval_cfg.get("kfold_shuffle", False)
    w = cfg["window"]
    n_forecast = w["forecast_minutes"] // w["freq_minutes"]

    for input_col, feature_tag in [("raw_glucose_level", "baseline")]:
        all_preds: list[np.ndarray] = []
        test_metrics_by_patient: list[dict[str, float]] = []
        cv_metrics_by_patient: list[dict[str, float]] = []

        for key in sorted(train_masters.keys()):
            if key not in test_masters:
                logger.warning("ARIMA: test data missing for %s, skipping", key)
                continue

            train_df = train_masters[key]
            test_df = test_masters[key]
            if input_col not in train_df.columns:
                logger.warning("ARIMA: %s missing for %s, skipping", input_col, key)
                continue

            tr_ser = train_df[input_col]
            tr_ser = ensure_monotonic_series(tr_ser)
            te_ser = test_df.get("raw_glucose_level", test_df["glucose_level"])
            te_ser = ensure_monotonic_series(te_ser)
            cv_metrics = _arima_cv_metrics(tr_ser, order, n_forecast, n_folds, shuffle)
            if cv_metrics is not None:
                cv_metrics_by_patient.append(cv_metrics)

            preds, actuals, artifact = _arima_patient_forecast(tr_ser, te_ser, order, n_forecast)
            if preds is None or actuals is None or artifact is None:
                continue

            test_metrics = compute_all_metrics(actuals, preds)
            test_metrics_by_patient.append(test_metrics)
            all_preds.append(preds)
            save_path = save_pickle_artifact(
                cfg,
                "ARIMA",
                feature_tag,
                f"{safe_name(key)}.pkl",
                {
                    **artifact,
                    "patient_key": key,
                    "model_name": "ARIMA",
                    "feature_tag": feature_tag,
                    "input_col": input_col,
                    "n_forecast": n_forecast,
                },
            )
            logger.info("Saved ARIMA (%s) artifact: %s", key, save_path)
            logger.info(
                "  ARIMA %s -> RMSE=%.3f R2=%.4f",
                key,
                test_metrics["rmse"],
                test_metrics["r2"],
            )

        if not test_metrics_by_patient:
            logger.error("ARIMA produced no results.")
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

        store.add("ARIMA", feature_tag, avg_cv, avg_test, np.vstack(all_preds))
        logger.info(
            "ARIMA avg -> CV RMSE=%.3f | Test RMSE=%.3f R2=%.4f",
            avg_cv.get("cv_rmse", float("nan")),
            avg_test["rmse"],
            avg_test["r2"],
        )
