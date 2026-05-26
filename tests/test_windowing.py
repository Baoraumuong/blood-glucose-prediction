"""
tests/test_windowing.py
-----------------------
Unit tests for sliding-window dataset creation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.windowing import create_multistep_dataset, DatasetScaler, prepare_datasets


def _make_df(n_rows: int = 50, n_features: int = 3) -> pd.DataFrame:
    data = np.random.default_rng(0).random((n_rows, n_features)).astype(np.float32)
    cols = ["glucose_level"] + [f"feat_{i}" for i in range(1, n_features)]
    return pd.DataFrame(data, columns=cols)


def test_create_multistep_shapes():
    df = _make_df(50)
    lb, fc = 6, 6
    X, y = create_multistep_dataset(df, lookback_steps=lb, forecast_steps=fc)
    n_windows = 50 - lb - fc + 1
    assert X.shape == (n_windows, lb, 3)
    assert y.shape == (n_windows, fc)


def test_create_multistep_target_values():
    """y[i] should equal the glucose column of df shifted by lookback_steps."""
    df = _make_df(30, n_features=1)
    lb, fc = 5, 3
    X, y = create_multistep_dataset(df, lookback_steps=lb, forecast_steps=fc)
    # Window 0: X uses rows 0:5, y uses rows 5:8
    expected_y0 = df["glucose_level"].values[lb : lb + fc]
    np.testing.assert_allclose(y[0], expected_y0, rtol=1e-5)


def test_prepare_datasets_uses_raw_glucose_as_target():
    idx = pd.date_range("2022-01-01", periods=20, freq="5min")
    df = pd.DataFrame(
        {
            "raw_glucose_level": np.arange(20, dtype=np.float32),
            "glucose_level": np.arange(20, dtype=np.float32) + 100,
            "total_insulin": np.zeros(20, dtype=np.float32),
            "insulin_count": np.zeros(20, dtype=np.float32),
            "insulin_3h_std": np.zeros(20, dtype=np.float32),
            "daily_exercise": np.zeros(20, dtype=np.float32),
        },
        index=idx,
    )
    cfg = {"window": {"lookback_minutes": 10, "forecast_minutes": 10, "freq_minutes": 5}}

    datasets = prepare_datasets({"p": df}, {"p": df}, cfg)

    np.testing.assert_allclose(datasets["y_train"][0], [2, 3])
    assert datasets["X_train_3d"].shape[-1] == 5


def test_dataset_scaler_fit_transform_shapes():
    df = _make_df(100, n_features=3)
    lb, fc = 6, 6
    X, y = create_multistep_dataset(df, lookback_steps=lb, forecast_steps=fc)

    scaler = DatasetScaler()
    scaler.fit(X, y)

    (X_2d,) = scaler.transform_2d(X)
    (X_3d,) = scaler.transform_3d(X)
    (y_s,)  = scaler.transform_y(y)

    assert X_2d.shape == (len(X), lb * 3)
    assert X_3d.shape == X.shape
    assert y_s.shape  == y.shape


def test_dataset_scaler_inverse():
    df = _make_df(100)
    lb, fc = 6, 6
    X, y = create_multistep_dataset(df, lookback_steps=lb, forecast_steps=fc)
    scaler = DatasetScaler().fit(X, y)
    (y_s,) = scaler.transform_y(y)
    y_back = scaler.inverse_transform_y(y_s)
    np.testing.assert_allclose(y_back, y, atol=1e-5)


def test_scaler_not_fitted_raises():
    scaler = DatasetScaler()
    X = np.zeros((10, 6, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.transform_2d(X)
