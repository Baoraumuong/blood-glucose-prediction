"""
data/loader.py
--------------
Discover OhioT1DM XML files and build per-patient master DataFrames
with smoothed glucose and engineered features.
"""
from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.data.parser import OhioT1DMParser, _parse_ts

logger = logging.getLogger(__name__)

HIGH_MISSING_FEATURES = {
    "acceleration",
    "basis_steps",
    "basis_air_temperature",
    "basis_heart_rate",
}
LOCF_FEATURES = ("basal", "basis_gsr", "finger_stick")
ZERO_IMPUTE_FEATURES = (
    "bolus",
    "temp_basal",
    "meal",
    "exercise",
    "basis_skin_temperature",
    "basis_sleep",
)
POINT_VALUE_ATTRS = {
    "bolus": "dose",
    "meal": "carbs",
    "finger_stick": "value",
    "basal": "value",
    "basis_gsr": "value",
    "basis_skin_temperature": "value",
}


def _align_glucose_to_grid(gl: pd.DataFrame, resample_freq: str) -> pd.DataFrame:
    """
    Put raw CGM readings on a dense grid without dropping phase-shifted values.

    Some OhioT1DM files contain long stretches sampled every 5 minutes but with
    different minute offsets, for example :02/:07 followed later by :00/:05.
    Reindexing against the first timestamp's phase drops the later readings.
    Keep a stable non-standard phase when one exists; otherwise round readings
    to the nearest configured grid timestamp and aggregate collisions.
    """
    if gl.empty:
        return gl

    grouped = gl.groupby(level=0).mean().sort_index()
    try:
        freq_delta = pd.Timedelta(resample_freq)
    except ValueError as exc:
        raise ValueError(f"resample_freq must be a fixed frequency, got {resample_freq!r}") from exc

    elapsed = grouped.index - grouped.index.min()
    has_single_phase = pd.Index(elapsed % freq_delta).nunique() <= 1

    if has_single_phase:
        glucose_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq=resample_freq)
        return grouped.reindex(glucose_index)

    rounded = grouped.copy()
    rounded.index = rounded.index.round(resample_freq)
    rounded = rounded.groupby(level=0).mean().sort_index()
    glucose_index = pd.date_range(rounded.index.min(), rounded.index.max(), freq=resample_freq)
    return rounded.reindex(glucose_index)


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
        if np.isnan(window).any():
            continue
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


def _empty_aligned(glucose_index: pd.DatetimeIndex, column_name: str) -> pd.DataFrame:
    return pd.DataFrame({column_name: np.nan}, index=glucose_index)


def _parse_events(xml_path: Path, feature: str) -> pd.DataFrame:
    """Parse point and interval event metadata for one XML feature."""
    columns = ["ts", "ts_begin", "ts_end", "value"]
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        logger.error("XML parse error for %s: %s", xml_path, exc)
        return pd.DataFrame(columns=columns)

    container = root.find(feature)
    if container is None:
        return pd.DataFrame(columns=columns)

    records = []
    for el in container:
        ts = el.get("ts")
        ts_begin = el.get("ts_begin") or el.get("tbegin")
        ts_end = el.get("ts_end") or el.get("tend")

        if feature == "basis_sleep":
            value = 1
        elif feature == "exercise":
            ts_begin = ts_begin or ts
            duration = pd.to_numeric(el.get("duration"), errors="coerce")
            begin = _parse_ts(ts_begin) if ts_begin else None
            if begin is not None and not pd.isna(duration):
                ts_end = begin + pd.Timedelta(minutes=float(duration))
            value = el.get("intensity") or 1
        elif feature == "temp_basal":
            value = el.get("value") or el.get("dose") or el.get("rate")
        else:
            value = el.get(POINT_VALUE_ATTRS.get(feature, "value"))

        records.append({"ts": ts, "ts_begin": ts_begin, "ts_end": ts_end, "value": value})

    df = pd.DataFrame(records, columns=columns)
    if df.empty:
        return df
    for col in ["ts", "ts_begin", "ts_end"]:
        df[col] = df[col].apply(_parse_ts)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def _align_point_feature(
    xml_path: Path,
    feature: str,
    glucose_index: pd.DatetimeIndex,
    column_name: str | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Round point feature timestamps to the nearest glucose timestamp."""
    column_name = column_name or feature
    events = _parse_events(xml_path, feature)
    if events.empty:
        return _empty_aligned(glucose_index, column_name)

    ts = events["ts"].where(events["ts"].notna(), events["ts_begin"])
    valid = ts.notna()
    if not valid.any():
        return _empty_aligned(glucose_index, column_name)

    df = events.loc[valid, ["value"]].copy()
    df.index = pd.DatetimeIndex(ts.loc[valid])
    return _align_to_glucose_index(df, glucose_index, column_name, agg=agg)


def _align_interval_feature(
    xml_path: Path,
    feature: str,
    glucose_index: pd.DatetimeIndex,
    column_name: str | None = None,
    default_value: float = 1.0,
) -> pd.DataFrame:
    """Apply begin/end events to every glucose timestamp they cover."""
    column_name = column_name or feature
    events = _parse_events(xml_path, feature)
    out = pd.Series(0.0, index=glucose_index, name=column_name)
    counts = pd.Series(0.0, index=glucose_index)
    if events.empty:
        return out.to_frame()

    for _, row in events.iterrows():
        begin = row["ts_begin"]
        end = row["ts_end"]
        if pd.isna(begin) or pd.isna(end):
            continue
        if end < begin:
            begin, end = end, begin
        value = row["value"] if not pd.isna(row["value"]) else default_value
        mask = (glucose_index >= begin) & (glucose_index <= end)
        out.loc[mask] += float(value)
        counts.loc[mask] += 1

    covered = counts > 0
    out.loc[covered] = out.loc[covered] / counts.loc[covered]
    return out.to_frame()


def feature_missing_percentages(master: pd.DataFrame) -> pd.DataFrame:
    """Missing percentage for aligned feature columns, with glucose rows as 100%."""
    denom = len(master)
    denom = max(denom, 1)
    rows = []
    for col in master.columns:
        if col in {"glucose_level", "raw_glucose_level"}:
            continue
        missing = int(master[col].isna().sum())
        rows.append(
            {
                "feature": col,
                "glucose_rows": denom,
                "missing": missing,
                "missing_pct_vs_glucose": round(missing / denom * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct_vs_glucose")


def plot_feature_missing_percentages(missing_df: pd.DataFrame, save_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = missing_df.sort_values("missing_pct_vs_glucose", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["missing_pct_vs_glucose"], color="#4c78a8")
    ax.set_xlabel("Missing % compared with glucose rows")
    ax.set_xlim(0, 100)
    ax.set_title("Aligned Feature Missingness vs Glucose")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


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
      - glucose_level      : causal Savitzky-Golay smoothed glucose input,
                             preserving missing CGM rows
      - raw_glucose_level  : unsmoothed glucose target for evaluation
      - basis_gsr          : GSR wearable signal aligned to the CGM timestamp grid
      - basis_sleep        : sleep signal aligned to the CGM timestamp grid
      - total_insulin      : bolus + basal dose in each 5-min window
      - insulin_count      : number of bolus events per window
      - insulin_3h_std     : 3-hour rolling std of total_insulin
      - daily_exercise     : 1 for every row on a day with exercise, else 0

    Parameters
    ----------
    xml_path : str | Path
        Path to the patient XML file.
    resample_freq : str
        CGM resampling frequency (should match config).
    savgol_window, savgol_polyorder : int
        Savitzky-Golay filter parameters.
    interpolate_limit : int
        Deprecated; glucose_level missing values are not filled.
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
    gl = _align_glucose_to_grid(gl, resample_freq)

    gl["raw_glucose_level"] = gl["value"]
    gl["glucose_level"] = _causal_savgol(
        gl["value"],
        window_length=savgol_window,
        polyorder=savgol_polyorder,
    )
    gl = gl.drop(columns="value")

    # ── 2. Bolus insulin ─────────────────────────────────────────────────
    bolus = _align_point_feature(xml_path, "bolus", gl.index, agg="sum")
    bolus_count = _align_point_feature(xml_path, "bolus", gl.index, "bolus_count", agg="count")

    # ── 3. Basal insulin (rate × 5 min ÷ 60 → units) ────────────────────
    basal = _align_point_feature(xml_path, "basal", gl.index)

    # ── 4. Exercise duration ──────────────────────────────────────────────
    exercise = _align_interval_feature(xml_path, "exercise", gl.index)

    # ── 5. Wearable sensors retained after missingness screening ─────────
    meal = _align_point_feature(xml_path, "meal", gl.index, agg="sum")
    temp_basal = _align_interval_feature(xml_path, "temp_basal", gl.index)
    basis_sleep = _align_interval_feature(xml_path, "basis_sleep", gl.index)
    basis_gsr = _align_point_feature(xml_path, "basis_gsr", gl.index)
    finger_stick = _align_point_feature(xml_path, "finger_stick", gl.index)
    basis_skin_temperature = _align_point_feature(xml_path, "basis_skin_temperature", gl.index)

    # ── 6. Merge ──────────────────────────────────────────────────────────
    master = gl.join(
        [
            bolus,
            bolus_count,
            meal,
            basal,
            temp_basal,
            exercise,
            basis_gsr,
            finger_stick,
            basis_skin_temperature,
            basis_sleep,
        ],
        how="left",
    )
    master.attrs["missing_percentages_before_imputation"] = feature_missing_percentages(master)
    raw_target = master["raw_glucose_level"]
    zero_cols = [c for c in ZERO_IMPUTE_FEATURES if c in master.columns]
    locf_cols = [c for c in LOCF_FEATURES if c in master.columns]
    master[zero_cols] = master[zero_cols].fillna(0)
    master[locf_cols] = master[locf_cols].ffill().fillna(0)
    master["bolus_count"] = master["bolus_count"].fillna(0)
    master["raw_glucose_level"] = raw_target

    # ── 7. Engineered features ────────────────────────────────────────────
    master["total_insulin"] = (
        master["bolus"] + master["basal"] * (5 / 60) + master["temp_basal"] * (5 / 60)
    )
    master["insulin_count"] = master["bolus_count"]
    master["insulin_3h_std"] = (
        master["total_insulin"]
        .rolling(rolling_insulin_window, min_periods=1)
        .std()
        .fillna(0)
    )
    master["daily_exercise"] = (
        master["exercise"].gt(0).groupby(master.index.date).transform("max").astype(int)
    )
    master = master.drop(columns=["bolus_count"])
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


def load_master_dataframes(
    split_name: str,
    processed_dir: str | Path,
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
) -> dict[str, pd.DataFrame]:
    """
    Load exported per-patient master DataFrames for a split.

    Expected file layout:
      processed_dir/train/{patient_key}_train_master.csv
      processed_dir/test/{patient_key}_test_master.csv
    """
    split_dir = Path(processed_dir) / split_name
    suffix = f"_{split_name}_master.csv"
    masters: dict[str, pd.DataFrame] = {}

    if not split_dir.exists():
        return masters

    for path in sorted(split_dir.glob(f"*{suffix}")):
        patient_key = path.name.removesuffix(suffix)
        df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
        df.index.name = None
        if {"raw_glucose_level", "glucose_level"}.issubset(df.columns):
            df["glucose_level"] = _causal_savgol(
                df["raw_glucose_level"],
                window_length=savgol_window,
                polyorder=savgol_polyorder,
            )
        masters[patient_key] = df.sort_index()
    return masters


def save_master_dataframes(
    split_name: str,
    masters: dict[str, pd.DataFrame],
    processed_dir: str | Path,
    save_missing_reports: bool = True,
) -> None:
    """Save per-patient master DataFrames to the cache layout used by main.py."""
    split_dir = Path(processed_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for patient_key, df in sorted(masters.items()):
        if df.empty:
            continue

        base = f"{patient_key}_{split_name}"
        df.to_csv(split_dir / f"{base}_master.csv", index_label="ts")

        missing_df = df.attrs.get("missing_percentages_before_imputation")
        if save_missing_reports and isinstance(missing_df, pd.DataFrame):
            missing_df.to_csv(
                split_dir / f"{base}_missing_pct_before_imputation.csv",
                index=False,
            )
