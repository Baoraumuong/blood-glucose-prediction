"""
features/windowing.py
---------------------
Create sliding-window (lookback → forecast) datasets for time-series models.
Handles both 3-D (LSTM/GRU) and 2-D (classical ML) representations,
plus feature scaling.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

TARGET_COL = "glucose_level"
RAW_TARGET_COL = "raw_glucose_level"
DEFAULT_EXCLUDED_FEATURES = {
    "acceleration",
    "basis_steps",
    "basis_air_temperature",
    "basis_skin_temperature",
    "basis_heart_rate",
}


def flatten_feature_names(feature_cols: list[str], lookback_steps: int) -> list[str]:
    """Return names matching flattened 2-D window columns."""
    return [
        f"{feature}_t-{lookback_steps - step}"
        for step in range(lookback_steps)
        for feature in feature_cols
    ]


def _replace_glucose_input(
    masters: dict[str, pd.DataFrame],
    use_savgol: bool,
) -> dict[str, pd.DataFrame]:
    """Return masters with glucose_level set to smoothed or raw CGM input."""
    if use_savgol:
        return masters

    out: dict[str, pd.DataFrame] = {}
    for key, df in masters.items():
        df_raw = df.copy()
        if RAW_TARGET_COL in df_raw.columns:
            df_raw[TARGET_COL] = df_raw[RAW_TARGET_COL]
        out[key] = df_raw
    return out


def _display_feature_cols(feature_cols: list[str], use_savgol: bool) -> list[str]:
    if use_savgol:
        return feature_cols
    return [RAW_TARGET_COL if c == TARGET_COL else c for c in feature_cols]


def _kalman_smooth_series(
    series: pd.Series,
    process_variance: float = 1e-3,
    measurement_variance: float = 4.0,
) -> pd.Series:
    """Smooth one glucose series with a local-level Kalman filter and RTS pass."""
    values = series.to_numpy(dtype=float)
    valid = ~np.isnan(values)
    if not valid.any():
        return series.copy()

    n = len(values)
    filtered = np.empty(n, dtype=float)
    filtered_var = np.empty(n, dtype=float)
    predicted = np.empty(n, dtype=float)
    predicted_var = np.empty(n, dtype=float)

    state = float(values[valid][0])
    variance = float(measurement_variance)

    for i, value in enumerate(values):
        pred_state = state
        pred_var = variance + process_variance
        predicted[i] = pred_state
        predicted_var[i] = pred_var

        if np.isnan(value):
            state = pred_state
            variance = pred_var
        else:
            gain = pred_var / (pred_var + measurement_variance)
            state = pred_state + gain * (value - pred_state)
            variance = (1.0 - gain) * pred_var

        filtered[i] = state
        filtered_var[i] = variance

    smoothed = filtered.copy()
    smoothed_var = filtered_var.copy()
    for i in range(n - 2, -1, -1):
        denom = predicted_var[i + 1]
        gain = filtered_var[i] / denom if denom > 0 else 0.0
        smoothed[i] = filtered[i] + gain * (smoothed[i + 1] - predicted[i + 1])
        smoothed_var[i] = filtered_var[i] + gain**2 * (smoothed_var[i + 1] - predicted_var[i + 1])

    return pd.Series(smoothed, index=series.index, name=series.name)


def _replace_glucose_with_kalman(
    masters: dict[str, pd.DataFrame],
    process_variance: float = 1e-3,
    measurement_variance: float = 4.0,
) -> dict[str, pd.DataFrame]:
    """Return per-patient masters with glucose_level Kalman-smoothed independently."""
    out: dict[str, pd.DataFrame] = {}
    for key, df in masters.items():
        kalman_df = df.copy()
        source_col = RAW_TARGET_COL if RAW_TARGET_COL in kalman_df.columns else TARGET_COL
        kalman_df[TARGET_COL] = _kalman_smooth_series(
            kalman_df[source_col],
            process_variance=process_variance,
            measurement_variance=measurement_variance,
        )
        out[key] = kalman_df
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core windowing
# ─────────────────────────────────────────────────────────────────────────────

def create_multistep_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lookback_steps: int = 6,
    forecast_steps: int = 6,
    feature_cols: list[str] | None = None,
    freq_minutes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding-window arrays preserving temporal order.

    Parameters
    ----------
    df : pd.DataFrame
        Patient master DataFrame (rows = time steps).
    target_col : str
        Column used as the prediction target.
    feature_cols : list[str] | None
        Columns used as model inputs. Defaults to all columns, preserving the
        original autoregressive behavior.
    lookback_steps : int
        Number of past steps fed to the model.
    forecast_steps : int
        Number of future steps to predict.
    freq_minutes : int | None
        Required spacing between consecutive timestamps. When provided for a
        DatetimeIndex, windows crossing a timestamp gap are rejected.

    Returns
    -------
    X : np.ndarray, shape (n_windows, lookback_steps, n_features)
    y : np.ndarray, shape (n_windows, forecast_steps)
    """
    if feature_cols is None:
        feature_cols = list(df.columns)

    feature_data = df[feature_cols].values.astype(np.float32)
    target_data = df[target_col].values.astype(np.float32)
    expected_delta = (
        pd.Timedelta(minutes=freq_minutes)
        if freq_minutes is not None and isinstance(df.index, pd.DatetimeIndex)
        else None
    )
    X, y = [], []

    for i in range(len(df) - lookback_steps - forecast_steps + 1):
        if expected_delta is not None:
            window_index = df.index[i : i + lookback_steps + forecast_steps]
            if not window_index.to_series().diff().iloc[1:].eq(expected_delta).all():
                continue

        x_window = feature_data[i : i + lookback_steps, :]
        y_window = target_data[i + lookback_steps : i + lookback_steps + forecast_steps]
        if np.isnan(x_window).any() or np.isnan(y_window).any():
            continue
        X.append(x_window)
        y.append(y_window)

    if not X:
        return (
            np.empty((0, lookback_steps, len(feature_cols)), dtype=np.float32),
            np.empty((0, forecast_steps), dtype=np.float32),
        )

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_global_dataset(
    patient_dfs: dict[str, pd.DataFrame],
    lookback_steps: int = 6,
    forecast_steps: int = 6,
    target_col: str = TARGET_COL,
    feature_cols: list[str] | None = None,
    freq_minutes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate windows from all patients without cross-patient contamination.

    Parameters
    ----------
    patient_dfs : dict
        Mapping patient_key → master DataFrame.
    lookback_steps, forecast_steps : int
        Window sizes.
    target_col : str
        Prediction target column name.
    feature_cols : list[str] | None
        Input columns to include in X.
    freq_minutes : int | None
        Required spacing between consecutive timestamps.

    Returns
    -------
    X : np.ndarray, shape (total_windows, lookback_steps, n_features)
    y : np.ndarray, shape (total_windows, forecast_steps)
    """
    all_X, all_y = [], []
    min_len = lookback_steps + forecast_steps

    for pid, df in patient_dfs.items():
        if df.empty or len(df) < min_len:
            logger.warning("Skipping %s: only %d rows (need %d)", pid, len(df), min_len)
            continue
        Xp, yp = create_multistep_dataset(
            df,
            target_col=target_col,
            lookback_steps=lookback_steps,
            forecast_steps=forecast_steps,
            feature_cols=feature_cols,
            freq_minutes=freq_minutes,
        )
        if len(Xp) == 0:
            logger.warning("Skipping %s: no complete target windows", pid)
            continue
        all_X.append(Xp)
        all_y.append(yp)
        logger.debug("  %s: %d windows", pid, len(Xp))

    if not all_X:
        raise ValueError("No valid patient data found to build dataset.")

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Scaling helpers
# ─────────────────────────────────────────────────────────────────────────────

class DatasetScaler:
    """
    Encapsulates MinMaxScalers for 2-D (classical ML) and 3-D (deep learning)
    feature representations as well as the target variable.

    Usage
    -----
    scaler = DatasetScaler()
    scaler.fit(X_train_3d, y_train)
    X_train_2d_s, X_test_2d_s = scaler.transform_2d(X_train_3d, X_test_3d)
    X_train_3d_s, X_test_3d_s = scaler.transform_3d(X_train_3d, X_test_3d)
    y_train_s, y_test_s       = scaler.transform_y(y_train, y_test)
    """

    def __init__(self) -> None:
        self._scaler_2d = MinMaxScaler()
        self._scaler_gl = MinMaxScaler()   # glucose-only for 3-D first channel
        self._scaler_y  = MinMaxScaler()
        self._fitted    = False

    def fit(self, X_train_3d: np.ndarray, y_train: np.ndarray) -> "DatasetScaler":
        """Fit all scalers on training data."""
        n_samples, lookback, n_features = X_train_3d.shape

        # 2-D scaler – flatten all features
        X_2d = X_train_3d.reshape(n_samples, -1)
        self._scaler_2d.fit(X_2d)

        # 3-D scaler – scale the glucose channel (index 0) only
        gl_flat = X_train_3d[:, :, 0].reshape(-1, 1)
        self._scaler_gl.fit(gl_flat)

        # Target scaler
        self._scaler_y.fit(y_train)

        self._fitted = True
        logger.debug("DatasetScaler fitted on X_train shape %s", X_train_3d.shape)
        return self

    def transform_2d(
        self, *arrays: np.ndarray
    ) -> list[np.ndarray]:
        """Flatten and scale 3-D arrays to 2-D."""
        self._check_fitted()
        out = []
        for X in arrays:
            n = X.shape[0]
            out.append(self._scaler_2d.transform(X.reshape(n, -1)))
        return out

    def transform_3d(
        self, *arrays: np.ndarray
    ) -> list[np.ndarray]:
        """Scale only the glucose channel (dim 0) of 3-D arrays in place."""
        self._check_fitted()
        out = []
        for X in arrays:
            X_s = X.copy()
            ns, ts, _ = X_s.shape
            X_s[:, :, 0] = self._scaler_gl.transform(
                X_s[:, :, 0].reshape(-1, 1)
            ).reshape(ns, ts)
            out.append(X_s)
        return out

    def transform_y(
        self, *arrays: np.ndarray
    ) -> list[np.ndarray]:
        """Scale target arrays."""
        self._check_fitted()
        return [self._scaler_y.transform(y) for y in arrays]

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Invert target scaling."""
        self._check_fitted()
        return self._scaler_y.inverse_transform(y_scaled)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("DatasetScaler must be fitted before transform.")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builder used by main.py
# ─────────────────────────────────────────────────────────────────────────────

def prepare_datasets(
    train_masters: dict[str, pd.DataFrame],
    test_masters:  dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Build and scale all dataset variants needed by the model suite.

    Returns a dict with keys:
      X_train_2d_full_s, X_test_2d_full_s  – scaled 2-D with all features
      X_train_2d_base_s, X_test_2d_base_s  – scaled 2-D glucose-only baseline
      X_train_3d,        X_test_3d          – scaled 3-D with all features
      X_train_3d_base,   X_test_3d_base     – scaled 3-D glucose-only baseline
      y_train,           y_test             – raw targets
      y_train_s,         y_test_s           – scaled targets
      scaler                                – fitted DatasetScaler
    """
    w = cfg["window"]
    lb_steps = w["lookback_minutes"] // w["freq_minutes"]
    fc_steps = w["forecast_minutes"] // w["freq_minutes"]

    logger.info("Building windowed datasets (lookback=%d, forecast=%d steps) …", lb_steps, fc_steps)

    # ── Full feature set ─────────────────────────────────────────────────
    all_master_dfs = [*train_masters.values(), *test_masters.values()]
    target_col = (
        RAW_TARGET_COL
        if all(RAW_TARGET_COL in df.columns for df in all_master_dfs)
        else TARGET_COL
    )
    sample_df = next(iter(train_masters.values()))
    excluded = set(cfg.get("feature_engineering", {}).get(
        "exclude_features",
        DEFAULT_EXCLUDED_FEATURES,
    ))
    full_feature_cols = [
        c for c in sample_df.columns
        if c != RAW_TARGET_COL and c not in excluded
    ]
    logger.info("Full model input features: %s", full_feature_cols)
    if excluded:
        logger.info("Excluded high-missing features: %s", sorted(excluded))

    base_feature_cols = [TARGET_COL]
    variants: list[dict[str, Any]] = []

    for use_savgol, smoothing_label in [
        (True, "Savitzky-Golay"),
        (False, "no Savitzky-Golay"),
    ]:
        variant_train = _replace_glucose_input(train_masters, use_savgol)
        variant_test = _replace_glucose_input(test_masters, use_savgol)

        for feature_label, feature_cols in [
            ("with features", full_feature_cols),
            ("baseline", base_feature_cols),
        ]:
            tag = f"{feature_label} + {smoothing_label}"
            X_train, y_train = build_global_dataset(
                variant_train,
                lb_steps,
                fc_steps,
                target_col=target_col,
                feature_cols=feature_cols,
                freq_minutes=w["freq_minutes"],
            )
            X_test, y_test = build_global_dataset(
                variant_test,
                lb_steps,
                fc_steps,
                target_col=target_col,
                feature_cols=feature_cols,
                freq_minutes=w["freq_minutes"],
            )

            scaler = DatasetScaler().fit(X_train, y_train)
            X_train_2d_s, X_test_2d_s = scaler.transform_2d(X_train, X_test)
            X_train_3d_s, X_test_3d_s = scaler.transform_3d(X_train, X_test)
            y_train_s, y_test_s = scaler.transform_y(y_train, y_test)
            display_cols = _display_feature_cols(feature_cols, use_savgol)

            variants.append(
                dict(
                    tag=tag,
                    feature_label=feature_label,
                    smoothing=smoothing_label,
                    use_savgol=use_savgol,
                    X_train_2d=X_train_2d_s,
                    X_test_2d=X_test_2d_s,
                    X_train_3d=X_train_3d_s,
                    X_test_3d=X_test_3d_s,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_s=y_train_s,
                    y_test_s=y_test_s,
                    scaler=scaler,
                    feature_cols=display_cols,
                    flat_feature_names=flatten_feature_names(display_cols, lb_steps),
                )
            )
            logger.info(
                "Dataset shape (%s): X_train=%s, X_test=%s",
                tag,
                X_train.shape,
                X_test.shape,
            )

    by_tag = {v["tag"]: v for v in variants}
    smoothed_full = by_tag["with features + Savitzky-Golay"]
    smoothed_base = by_tag["baseline + Savitzky-Golay"]

    return dict(
        variants=variants,
        classical_variants=variants,
        deep_variants=variants,
        X_train_2d_full_s=smoothed_full["X_train_2d"],
        X_test_2d_full_s=smoothed_full["X_test_2d"],
        X_train_2d_base_s=smoothed_base["X_train_2d"],
        X_test_2d_base_s=smoothed_base["X_test_2d"],
        X_train_3d=smoothed_full["X_train_3d"],
        X_test_3d=smoothed_full["X_test_3d"],
        X_train_3d_base=smoothed_base["X_train_3d"],
        X_test_3d_base=smoothed_base["X_test_3d"],
        y_train=smoothed_full["y_train"],
        y_test=smoothed_full["y_test"],
        y_train_base=smoothed_base["y_train"],
        y_test_base=smoothed_base["y_test"],
        y_train_s=smoothed_full["y_train_s"],
        y_test_s=smoothed_full["y_test_s"],
        scaler_full=smoothed_full["scaler"],
        scaler_base=smoothed_base["scaler"],
        full_feature_cols=full_feature_cols,
        base_feature_cols=base_feature_cols,
        full_flat_feature_names=smoothed_full["flat_feature_names"],
        base_flat_feature_names=smoothed_base["flat_feature_names"],
    )


def prepare_kalman_datasets(
    train_masters: dict[str, pd.DataFrame],
    test_masters: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build glucose-only and feature-rich variants after per-split Kalman smoothing."""
    w = cfg["window"]
    lb_steps = w["lookback_minutes"] // w["freq_minutes"]
    fc_steps = w["forecast_minutes"] // w["freq_minutes"]
    kalman_cfg = cfg.get("feature_engineering", {}).get("kalman", {})

    logger.info(
        "Building Kalman-smoothed datasets (lookback=%d, forecast=%d steps) ...",
        lb_steps,
        fc_steps,
    )
    kalman_train = _replace_glucose_with_kalman(
        train_masters,
        process_variance=kalman_cfg.get("process_variance", 1e-3),
        measurement_variance=kalman_cfg.get("measurement_variance", 4.0),
    )
    kalman_test = _replace_glucose_with_kalman(
        test_masters,
        process_variance=kalman_cfg.get("process_variance", 1e-3),
        measurement_variance=kalman_cfg.get("measurement_variance", 4.0),
    )

    all_master_dfs = [*train_masters.values(), *test_masters.values()]
    target_col = (
        RAW_TARGET_COL
        if all(RAW_TARGET_COL in df.columns for df in all_master_dfs)
        else TARGET_COL
    )
    sample_df = next(iter(train_masters.values()))
    excluded = set(cfg.get("feature_engineering", {}).get(
        "exclude_features",
        DEFAULT_EXCLUDED_FEATURES,
    ))
    full_feature_cols = [
        c for c in sample_df.columns
        if c != RAW_TARGET_COL and c not in excluded
    ]
    base_feature_cols = [TARGET_COL]
    variants: list[dict[str, Any]] = []

    for feature_label, feature_cols in [
        ("with features", full_feature_cols),
        ("baseline", base_feature_cols),
    ]:
        tag = f"{feature_label} + Kalman"
        X_train, y_train = build_global_dataset(
            kalman_train,
            lb_steps,
            fc_steps,
            target_col=target_col,
            feature_cols=feature_cols,
            freq_minutes=w["freq_minutes"],
        )
        X_test, y_test = build_global_dataset(
            kalman_test,
            lb_steps,
            fc_steps,
            target_col=target_col,
            feature_cols=feature_cols,
            freq_minutes=w["freq_minutes"],
        )
        scaler = DatasetScaler().fit(X_train, y_train)
        X_train_3d_s, X_test_3d_s = scaler.transform_3d(X_train, X_test)
        y_train_s, y_test_s = scaler.transform_y(y_train, y_test)

        variants.append(
            dict(
                tag=tag,
                feature_label=feature_label,
                smoothing="Kalman",
                use_savgol=False,
                use_kalman=True,
                X_train_3d=X_train_3d_s,
                X_test_3d=X_test_3d_s,
                y_train=y_train,
                y_test=y_test,
                y_train_s=y_train_s,
                y_test_s=y_test_s,
                scaler=scaler,
                feature_cols=feature_cols,
                flat_feature_names=flatten_feature_names(feature_cols, lb_steps),
            )
        )
        logger.info(
            "Kalman dataset shape (%s): X_train=%s, X_test=%s",
            tag,
            X_train.shape,
            X_test.shape,
        )

    return {"deep_variants": variants}
