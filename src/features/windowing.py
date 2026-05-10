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


# ─────────────────────────────────────────────────────────────────────────────
# Core windowing
# ─────────────────────────────────────────────────────────────────────────────

def create_multistep_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lookback_steps: int = 6,
    forecast_steps: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding-window arrays preserving temporal order.

    Parameters
    ----------
    df : pd.DataFrame
        Patient master DataFrame (rows = time steps).
    target_col : str
        Column used as the prediction target.
    lookback_steps : int
        Number of past steps fed to the model.
    forecast_steps : int
        Number of future steps to predict.

    Returns
    -------
    X : np.ndarray, shape (n_windows, lookback_steps, n_features)
    y : np.ndarray, shape (n_windows, forecast_steps)
    """
    data = df.values.astype(np.float32)
    tidx = df.columns.get_loc(target_col)
    X, y = [], []

    for i in range(len(df) - lookback_steps - forecast_steps + 1):
        X.append(data[i : i + lookback_steps, :])
        y.append(data[i + lookback_steps : i + lookback_steps + forecast_steps, tidx])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_global_dataset(
    patient_dfs: dict[str, pd.DataFrame],
    lookback_steps: int = 6,
    forecast_steps: int = 6,
    target_col: str = TARGET_COL,
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
        Xp, yp = create_multistep_dataset(df, target_col, lookback_steps, forecast_steps)
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
    X_train_full, y_train = build_global_dataset(train_masters, lb_steps, fc_steps)
    X_test_full,  y_test  = build_global_dataset(test_masters,  lb_steps, fc_steps)

    # ── Glucose-only baseline ─────────────────────────────────────────────
    train_gl = {k: df[[TARGET_COL]] for k, df in train_masters.items()}
    test_gl  = {k: df[[TARGET_COL]] for k, df in test_masters.items()}
    X_train_base, y_train_base = build_global_dataset(train_gl, lb_steps, fc_steps)
    X_test_base,  y_test_base  = build_global_dataset(test_gl,  lb_steps, fc_steps)

    logger.info(
        "Dataset shapes – full: X_train=%s, baseline: X_train=%s",
        X_train_full.shape,
        X_train_base.shape,
    )

    # ── Scaling ───────────────────────────────────────────────────────────
    scaler_full = DatasetScaler().fit(X_train_full, y_train)
    scaler_base = DatasetScaler().fit(X_train_base, y_train_base)

    X_train_2d_full_s, X_test_2d_full_s = scaler_full.transform_2d(X_train_full, X_test_full)
    X_train_2d_base_s, X_test_2d_base_s = scaler_base.transform_2d(X_train_base, X_test_base)

    X_train_3d, X_test_3d               = scaler_full.transform_3d(X_train_full, X_test_full)
    X_train_3d_base, X_test_3d_base     = scaler_base.transform_3d(X_train_base, X_test_base)

    y_train_s, y_test_s = scaler_full.transform_y(y_train, y_test)

    return dict(
        X_train_2d_full_s=X_train_2d_full_s,
        X_test_2d_full_s=X_test_2d_full_s,
        X_train_2d_base_s=X_train_2d_base_s,
        X_test_2d_base_s=X_test_2d_base_s,
        X_train_3d=X_train_3d,
        X_test_3d=X_test_3d,
        X_train_3d_base=X_train_3d_base,
        X_test_3d_base=X_test_3d_base,
        y_train=y_train,
        y_test=y_test,
        y_train_base=y_train_base,
        y_test_base=y_test_base,
        y_train_s=y_train_s,
        y_test_s=y_test_s,
        scaler_full=scaler_full,
        scaler_base=scaler_base,
    )
