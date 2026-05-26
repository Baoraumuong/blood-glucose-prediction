"""
data/loader.py
--------------
Discover OhioT1DM XML files and build per-patient master DataFrames
with smoothed glucose and engineered features.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.data.parser import OhioT1DMParser

logger = logging.getLogger(__name__)

HIGH_MISSING_FEATURES = {
    "acceleration",
    "basis_steps",
    "basis_air_temperature",
    "basis_skin_temperature",
    "basis_heart_rate",
}
LOW_MISSING_SENSOR_FEATURES = ("basis_gsr", "basis_sleep")


def _causal_savgol(series: pd.Series, window_length: int, polyorder: int) -> pd.Series:
    """
    Apply Savitzky-Golay smoothing using only current and past values.

    scipy.signal.savgol_filter is centered by default, which would let future
    glucose readings influence lookback features.
    """
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder or len(series) < window_length:
        return series.copy()

    smoothed = series.copy()
    values = series.to_numpy(dtype=float)
    for i in range(window_length - 1, len(values)):
        window = values[i - window_length + 1 : i + 1]
        smoothed.iloc[i] = savgol_filter(
            window,
            window_length=window_length,
            polyorder=polyorder,
        )[-1]
    return smoothed


def _align_to_glucose_index(
    df: pd.DataFrame,
    glucose_index: pd.DatetimeIndex,
    column_name: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Snap feature timestamps to the nearest glucose timestamp, then aggregate
    duplicate snapped rows. This keeps every model input on the exact CGM grid.
    """
    if df.empty:
        return pd.DataFrame({column_name: np.nan}, index=glucose_index)

    aligned = df.copy()
    nearest_pos = glucose_index.get_indexer(aligned.index, method="nearest")
    aligned = aligned[nearest_pos >= 0]
    nearest_pos = nearest_pos[nearest_pos >= 0]
    if aligned.empty:
        return pd.DataFrame({column_name: np.nan}, index=glucose_index)

    aligned.index = glucose_index.take(nearest_pos)
    aligned = aligned.rename(columns={"value": column_name})
    if agg == "sum":
        aligned = aligned.groupby(level=0).sum()
    elif agg == "count":
        aligned = aligned.groupby(level=0).count()
    else:
        aligned = aligned.groupby(level=0).mean()
    return aligned.reindex(glucose_index)


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_xml_files(
    base_dir: str | Path,
    cohort_ids: dict[str, list[int]],
    train_suffix: str = "-ws-training.xml",
    test_suffix:  str = "-ws-testing.xml",
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Discover train and test XML files for all patients.

    Parameters
    ----------
    base_dir : str | Path
        Root of the OhioT1DM dataset (contains year sub-folders).
    cohort_ids : dict
        Mapping of year → list of patient IDs, e.g. {"2018": [559, ...], "2020": [...]}.
    train_suffix, test_suffix : str
        File-name suffixes for training and testing splits.

    Returns
    -------
    train_files, test_files : dict[str, str]
        Keys are '{patient_id}_{year}'; values are absolute paths.
    """
    base_dir = Path(base_dir)
    train_files: dict[str, str] = {}
    test_files:  dict[str, str] = {}

    for year, patient_ids in cohort_ids.items():
        for pid in patient_ids:
            key = f"{pid}_{year}"
            tr  = base_dir / year / "train" / f"{pid}{train_suffix}"
            te  = base_dir / year / "test"  / f"{pid}{test_suffix}"
            if tr.exists():
                train_files[key] = str(tr)
            else:
                logger.warning("Training file not found: %s", tr)
            if te.exists():
                test_files[key] = str(te)
            else:
                logger.warning("Testing file not found: %s", te)

    logger.info("Discovered %d train files, %d test files", len(train_files), len(test_files))
    return train_files, test_files


# ─────────────────────────────────────────────────────────────────────────────
# Master DataFrame builder
# ─────────────────────────────────────────────────────────────────────────────

def build_patient_master(
    xml_path: str | Path,
    resample_freq: str = "5min",
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
    interpolate_limit: int = 6,
    rolling_insulin_window: int = 36,
) -> pd.DataFrame:
    """
    Build a single master DataFrame for one patient file.

    Columns returned:
      - glucose_level      : causal Savitzky-Golay smoothed glucose input
      - raw_glucose_level  : unsmoothed glucose target for evaluation
      - basis_gsr          : GSR wearable signal aligned to the CGM timestamp grid
      - basis_sleep        : sleep signal aligned to the CGM timestamp grid
      - total_insulin      : bolus + basal dose in each 5-min window
      - insulin_count      : number of bolus events per window
      - insulin_3h_std     : 3-hour rolling std of total_insulin
      - daily_exercise     : cumulative exercise duration, reset at midnight

    Parameters
    ----------
    xml_path : str | Path
        Path to the patient XML file.
    resample_freq : str
        CGM resampling frequency (should match config).
    savgol_window, savgol_polyorder : int
        Savitzky-Golay filter parameters.
    interpolate_limit : int
        Max consecutive NaN steps filled by linear interpolation.
    rolling_insulin_window : int
        Rolling window size for insulin std (in 5-min steps).

    Returns
    -------
    pd.DataFrame
    """
    xml_path = Path(xml_path)

    # ── 1. Glucose ────────────────────────────────────────────────────────
    gl = OhioT1DMParser(xml_path, "glucose_level", resample_freq).make_dataframe()
    if gl.empty:
        return pd.DataFrame()
    glucose_index = pd.date_range(gl.index.min(), gl.index.max(), freq=resample_freq)
    gl = gl.groupby(level=0).mean().reindex(glucose_index)

    gl["raw_glucose_level"] = gl["value"]
    glucose_input = (
        gl["value"]
        .interpolate("linear", limit=interpolate_limit)
        .ffill()
        .bfill()
    )
    gl["glucose_level"] = _causal_savgol(
        glucose_input,
        window_length=savgol_window,
        polyorder=savgol_polyorder,
    )
    gl = gl.drop(columns="value")

    # ── 2. Bolus insulin ─────────────────────────────────────────────────
    bolus = OhioT1DMParser(xml_path, "bolus", resample_freq).make_dataframe()
    if not bolus.empty:
        bolus_sum = _align_to_glucose_index(bolus, gl.index, "dose_sum", agg="sum")
        bolus_count = _align_to_glucose_index(bolus, gl.index, "dose_count", agg="count")
        bolus = bolus_sum.join(bolus_count)
    else:
        bolus = pd.DataFrame(
            {"dose_sum": 0.0, "dose_count": 0}, index=gl.index
        )

    # ── 3. Basal insulin (rate × 5 min ÷ 60 → units) ────────────────────
    basal = OhioT1DMParser(xml_path, "basal", resample_freq).make_dataframe()
    if not basal.empty:
        basal = _align_to_glucose_index(basal, gl.index, "basal_dose") * (5 / 60)
    else:
        basal = pd.DataFrame({"basal_dose": 0.0}, index=gl.index)

    # ── 4. Exercise duration ──────────────────────────────────────────────
    exercise = OhioT1DMParser(xml_path, "exercise", resample_freq).make_dataframe()
    if not exercise.empty:
        exercise = _align_to_glucose_index(exercise, gl.index, "exercise", agg="sum")
    else:
        exercise = pd.DataFrame({"exercise": 0.0}, index=gl.index)

    # ── 5. Wearable sensors retained after missingness screening ─────────
    sensor_frames = []
    for feature in LOW_MISSING_SENSOR_FEATURES:
        sensor = OhioT1DMParser(xml_path, feature, resample_freq).make_dataframe()
        sensor_frames.append(_align_to_glucose_index(sensor, gl.index, feature))

    # ── 6. Merge ──────────────────────────────────────────────────────────
    master = gl.join(bolus[["dose_sum", "dose_count"]], how="left")
    master = master.join(basal, how="left")
    master = master.join(exercise, how="left")
    for sensor in sensor_frames:
        master = master.join(sensor, how="left")
    raw_target = master["raw_glucose_level"]
    event_cols = ["dose_sum", "dose_count", "basal_dose", "exercise"]
    master[event_cols] = master[event_cols].fillna(0)
    master[list(LOW_MISSING_SENSOR_FEATURES)] = (
        master[list(LOW_MISSING_SENSOR_FEATURES)].ffill().bfill().fillna(0)
    )
    master["raw_glucose_level"] = raw_target

    # ── 7. Engineered features ────────────────────────────────────────────
    master["total_insulin"]  = master["dose_sum"] + master["basal_dose"]
    master["insulin_count"]  = master["dose_count"]
    master["insulin_3h_std"] = (
        master["total_insulin"]
        .rolling(rolling_insulin_window, min_periods=1)
        .std()
        .fillna(0)
    )
    master["daily_exercise"] = (
        master["exercise"].groupby(master.index.date).cumsum()
    )
    master = master.drop(columns=["dose_sum", "dose_count", "basal_dose", "exercise"])
    master = master.drop(columns=[c for c in HIGH_MISSING_FEATURES if c in master.columns])
    return master


def build_all_masters(
    file_dict: dict[str, str],
    cfg: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    Build master DataFrames for all patients in file_dict.

    Parameters
    ----------
    file_dict : dict
        Mapping patient_key → xml_path (from find_xml_files).
    cfg : dict
        Full project config dict.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    fe_cfg = cfg.get("feature_engineering", {})
    masters: dict[str, pd.DataFrame] = {}
    for key, path in sorted(file_dict.items()):
        try:
            df = build_patient_master(
                path,
                resample_freq=cfg["dataset"]["resample_freq"],
                savgol_window=fe_cfg.get("savgol_window", 11),
                savgol_polyorder=fe_cfg.get("savgol_polyorder", 2),
                interpolate_limit=fe_cfg.get("interpolate_limit", 6),
                rolling_insulin_window=fe_cfg.get("rolling_insulin_window", 36),
            )
            masters[key] = df
            logger.info("  %-12s  rows=%d  cols=%s", key, len(df), list(df.columns))
        except Exception as exc:
            logger.error("Failed to build master for %s: %s", key, exc)
    return masters
