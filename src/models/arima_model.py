"""
models/arima_model.py
---------------------
ARIMA-based multi-step forecasting applied patient-by-patient,
then averaged – matching the notebook's strategy exactly.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.metrics import ResultStore, compute_all_metrics
from src.models.time_series_utils import fill_series_gaps, is_finite_window

logger = logging.getLogger(__name__)


def _arima_patient_forecast(
    train_series: pd.Series,
    test_series:  pd.Series,
    order: tuple[int, int, int] = (2, 1, 2),
    n_forecast: int = 6,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Fit ARIMA on smoothed training series, then generate non-overlapping
    n_forecast-step-ahead windows over the test set.
    """
    train_model = fill_series_gaps(train_series)
    test_model = fill_series_gaps(test_series)
    if train_model.empty or test_model.empty:
        return None, None

    train_smooth = train_model.ewm(span=5, adjust=False).mean()

    try:
        fit = SARIMAX(
            train_smooth, order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            missing="none",
        ).fit(disp=False)
    except Exception as exc:
        logger.warning("ARIMA fit failed: %s", exc)
        return None, None

    all_model_series = pd.concat([train_smooth, test_model])
    all_actual_series = pd.concat([train_model, pd.to_numeric(test_series, errors="coerce")])
    preds_list, actuals_list = [], []
    n_train = len(train_smooth)
    n_test = min(len(test_model), len(test_series))

    for i in range(0, n_test - n_forecast + 1, n_forecast):
        start  = n_train + i
        actual = all_actual_series.iloc[start : start + n_forecast].values
        if len(actual) < n_forecast:
            break
        if not is_finite_window(actual):
            logger.debug("ARIMA window %d skipped because actuals contain NaN/inf", i)
            continue
        try:
            history = all_model_series.iloc[:start]
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


def run_arima(
    train_masters: dict[str, pd.DataFrame],
    test_masters:  dict[str, pd.DataFrame],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    """
    Run ARIMA per patient, collect RMSE / R², average across patients,
    and write results to the ResultStore.
    """
    model_cfg = cfg["models"].get("arima", {})
    order     = tuple(model_cfg.get("order", [2, 1, 2]))
    w         = cfg["window"]
    n_forecast = w["forecast_minutes"] // w["freq_minutes"]

    for input_col, smoothing_label in [
        ("glucose_level", "Savitzky-Golay"),
        ("raw_glucose_level", "no Savitzky-Golay"),
    ]:
        all_preds: list[np.ndarray] = []
        patient_metrics: list[dict] = []

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
            te_ser = test_df.get("raw_glucose_level", test_df["glucose_level"])

            preds, actuals = _arima_patient_forecast(tr_ser, te_ser, order, n_forecast)
            if preds is None:
                continue

            m = compute_all_metrics(actuals, preds)
            patient_metrics.append(m)
            all_preds.append(preds)
            logger.info(
                "  ARIMA %s (%s) -> RMSE=%.3f R2=%.4f",
                key,
                smoothing_label,
                m["rmse"],
                m["r2"],
            )

        if not patient_metrics:
            logger.error("ARIMA produced no results for %s.", smoothing_label)
            continue

        avg_metrics = {
            k: float(np.mean([m[k] for m in patient_metrics]))
            for k in patient_metrics[0]
        }
        cv_proxy = {f"cv_{k}": v for k, v in avg_metrics.items()}
        feature_tag = f"baseline + {smoothing_label}"

        store.add("ARIMA", feature_tag, cv_proxy, avg_metrics, np.vstack(all_preds))
        logger.info(
            "ARIMA avg (%s) -> RMSE=%.3f R2=%.4f",
            smoothing_label,
            avg_metrics["rmse"],
            avg_metrics["r2"],
        )

    return

    all_preds:   list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []
    patient_metrics: list[dict] = []

    for key in sorted(train_masters.keys()):
        if key not in test_masters:
            logger.warning("ARIMA: test data missing for %s, skipping", key)
            continue

        tr_ser = train_masters[key]["glucose_level"]
        te_ser = test_masters[key].get("raw_glucose_level", test_masters[key]["glucose_level"])

        preds, actuals = _arima_patient_forecast(tr_ser, te_ser, order, n_forecast)
        if preds is None:
            continue

        m = compute_all_metrics(actuals, preds)
        patient_metrics.append(m)
        all_preds.append(preds)
        all_actuals.append(actuals)
        logger.info("  ARIMA %s → RMSE=%.3f  R²=%.4f", key, m["rmse"], m["r2"])

    if not patient_metrics:
        logger.error("ARIMA produced no results.")
        return

    avg_metrics = {k: float(np.mean([m[k] for m in patient_metrics]))
                   for k in patient_metrics[0]}
    preds_flat  = np.vstack(all_preds)
    cv_proxy    = {f"cv_{k}": v for k, v in avg_metrics.items()}

    # ARIMA has no separate CV step – report test metrics in both slots
    store.add("ARIMA", "baseline",      cv_proxy, avg_metrics, preds_flat)
    logger.info("ARIMA avg → RMSE=%.3f  R²=%.4f", avg_metrics["rmse"], avg_metrics["r2"])
