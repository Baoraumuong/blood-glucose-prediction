"""
evaluation/diagnostics.py
-------------------------
Reusable EDA/statistical diagnostics for aligned OhioT1DM time series.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.data.parser import OhioT1DMParser

REGULAR_MISSING_FEATURES = [
    "acceleration",
    "basis_air_temperature",
    "basis_gsr",
    "basis_heart_rate",
    "basis_skin_temperature",
    "basis_sleep",
    "basis_steps",
    "glucose_level",
]


def collect_missing_stats(
    files: dict[str, str],
    features: Iterable[str] = REGULAR_MISSING_FEATURES,
    resample_freq: str = "5min",
) -> pd.DataFrame:
    """Build a missingness table only for regularly checked signals."""
    records = []
    for patient, path in sorted(files.items()):
        for feature in features:
            stats = OhioT1DMParser(path, feature, resample_freq).missing_stats()
            records.append({"patient": patient, "feature": feature, **stats})
    return pd.DataFrame(records)


def run_adf_test(series: pd.Series, autolag: str = "AIC") -> dict[str, float | int | bool]:
    """
    Run an Augmented Dickey-Fuller test after cleaning invalid observations.
    Constant or too-short series are reported as not testable.
    """
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 8 or clean.nunique() <= 1:
        return {
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "used_lag": 0,
            "n_obs": int(len(clean)),
            "is_stationary": False,
            "testable": False,
        }

    stat, p_value, used_lag, n_obs, *_ = adfuller(clean.to_numpy(), autolag=autolag)
    return {
        "adf_statistic": float(stat),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "is_stationary": bool(p_value < 0.05),
        "testable": True,
    }


def aligned_feature_correlations(
    xml_path: str | Path,
    features: Iterable[str],
    target_feature: str = "glucose_level",
    resample_freq: str = "5min",
) -> pd.Series:
    """
    Correlate each feature with glucose using only identical timestamps after
    snapping feature records to the nearest glucose timestamp.
    """
    target = OhioT1DMParser(xml_path, target_feature, resample_freq).make_dataframe()
    if target.empty:
        return pd.Series(dtype=float)

    glucose_index = target.index
    target_series = target["value"].rename(target_feature)
    out = {}
    for feature in features:
        if feature == target_feature:
            continue
        data = OhioT1DMParser(xml_path, feature, resample_freq).make_dataframe()
        if data.empty:
            out[feature] = np.nan
            continue
        positions = glucose_index.get_indexer(data.index, method="nearest")
        aligned = data[positions >= 0].copy()
        positions = positions[positions >= 0]
        if aligned.empty:
            out[feature] = np.nan
            continue
        aligned.index = glucose_index.take(positions)
        feature_series = aligned["value"].groupby(level=0).mean().rename(feature)
        joined = pd.concat([target_series, feature_series], axis=1, join="inner").dropna()
        out[feature] = (
            np.nan
            if len(joined) < 2 or joined[feature].nunique() <= 1
            else float(joined[target_feature].corr(joined[feature]))
        )
    return pd.Series(out, dtype=float)
