"""
models/sarimax_model.py
-----------------------
Patient-wise SARIMAX/ARIMAX forecasting with engineered exogenous features.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.metrics import ResultStore, compute_all_metrics

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize exogenous variables with train statistics only."""
    train = train_exog.astype(float).ffill().fillna(0)
    test = test_exog.astype(float).ffill().fillna(0)
    train.attrs = {}
    test.attrs = {}
    mean = train.mean()
    std = train.std().replace(0, 1).fillna(1)
    return (train - mean) / std, (test - mean) / std


def _sarimax_patient_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    train_exog: pd.DataFrame,
    test_exog: pd.DataFrame,
    order: tuple[int, int, int] = (2, 1, 2),
    n_forecast: int = 6,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Fit SARIMAX on glucose plus exogenous feature columns, then generate
    non-overlapping n_forecast-step-ahead windows over the test set.
    """
    train_smooth = train_series.ewm(span=5, adjust=False).mean()
    train_x, test_x = _scale_exog(train_exog, test_exog)

    try:
        fit = SARIMAX(
            train_smooth,
            exog=train_x,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    except Exception as exc:
        logger.warning("SARIMAX fit failed: %s", exc)
        return None, None

    all_series = pd.concat([train_smooth, test_series])
    all_exog = pd.concat([train_x, test_x])
    preds_list, actuals_list = [], []
    n_train = len(train_smooth)
    n_test = len(test_series)

    for i in range(0, n_test - n_forecast + 1, n_forecast):
        start = n_train + i
        end = start + n_forecast
        actual = all_series.iloc[start:end].values
        future_exog = all_exog.iloc[start:end]
        if len(actual) < n_forecast or len(future_exog) < n_forecast:
            break
        try:
            history = all_series.iloc[:start]
            history_exog = all_exog.iloc[:start]
            fc = fit.apply(history, exog=history_exog, refit=False).forecast(
                n_forecast,
                exog=future_exog,
            )
            preds_list.append(fc.values)
            actuals_list.append(actual)
        except Exception as exc:
            logger.debug("SARIMAX window %d failed: %s", i, exc)

    if not preds_list:
        return None, None

    return np.array(preds_list), np.array(actuals_list)


def run_sarimax(
    train_masters: dict[str, pd.DataFrame],
    test_masters: dict[str, pd.DataFrame],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    """
    Run SARIMAX/ARIMAX per patient using engineered exogenous features.

    The target remains glucose; exogenous inputs are the aligned non-glucose
    feature columns from build_patient_master().
    """
    model_cfg = cfg["models"].get("sarimax", cfg["models"].get("arima", {}))
    order = tuple(model_cfg.get("order", [2, 1, 2]))
    w = cfg["window"]
    n_forecast = w["forecast_minutes"] // w["freq_minutes"]

    all_preds: list[np.ndarray] = []
    patient_metrics: list[dict] = []

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

        tr_ser = train_df["glucose_level"]
        te_ser = test_df.get("raw_glucose_level", test_df["glucose_level"])
        preds, actuals = _sarimax_patient_forecast(
            tr_ser,
            te_ser,
            train_df[exog_cols],
            test_df[exog_cols],
            order,
            n_forecast,
        )
        if preds is None:
            continue

        metrics = compute_all_metrics(actuals, preds)
        patient_metrics.append(metrics)
        all_preds.append(preds)
        logger.info(
            "  SARIMAX %s (%d exog) -> RMSE=%.3f R2=%.4f",
            key,
            len(exog_cols),
            metrics["rmse"],
            metrics["r2"],
        )

    if not patient_metrics:
        logger.error("SARIMAX produced no results.")
        return

    avg_metrics = {
        k: float(np.mean([m[k] for m in patient_metrics]))
        for k in patient_metrics[0]
    }
    cv_proxy = {f"cv_{k}": v for k, v in avg_metrics.items()}

    store.add(
        "SARIMAX",
        "with features",
        cv_proxy,
        avg_metrics,
        np.vstack(all_preds),
    )
    logger.info(
        "SARIMAX avg -> RMSE=%.3f R2=%.4f",
        avg_metrics["rmse"],
        avg_metrics["r2"],
    )
