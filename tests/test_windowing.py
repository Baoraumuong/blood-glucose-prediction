"""
tests/test_windowing.py
-----------------------
Unit tests for sliding-window dataset creation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.windowing import (
    create_multistep_dataset,
    DatasetScaler,
    prepare_datasets,
)


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


def test_create_multistep_rejects_timestamp_gaps():
    idx = pd.to_datetime(
        [
            "2022-01-01 00:00:00",
            "2022-01-01 00:05:00",
            "2022-01-01 00:20:00",
            "2022-01-01 00:25:00",
            "2022-01-01 00:30:00",
        ]
    )
    df = pd.DataFrame({"glucose_level": np.arange(5, dtype=np.float32)}, index=idx)

    X, y = create_multistep_dataset(
        df,
        lookback_steps=2,
        forecast_steps=2,
        freq_minutes=5,
    )

    assert X.shape == (0, 2, 1)
    assert y.shape == (0, 2)


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


def test_prepare_datasets_builds_savgol_and_raw_input_variants():
    idx = pd.date_range("2022-01-01", periods=20, freq="5min")
    df = pd.DataFrame(
        {
            "raw_glucose_level": np.arange(20, dtype=np.float32),
            "glucose_level": np.arange(20, dtype=np.float32) + 100,
            "total_insulin": np.zeros(20, dtype=np.float32),
        },
        index=idx,
    )
    cfg = {"window": {"lookback_minutes": 10, "forecast_minutes": 10, "freq_minutes": 5}}

    datasets = prepare_datasets({"p": df}, {"p": df}, cfg)

    assert [v["tag"] for v in datasets["variants"]] == [
        "with features + Savitzky-Golay",
        "baseline + Savitzky-Golay",
        "with features + no Savitzky-Golay",
        "baseline + no Savitzky-Golay",
    ]
    raw_variant = next(v for v in datasets["variants"] if v["tag"] == "baseline + no Savitzky-Golay")
    assert raw_variant["feature_cols"] == ["raw_glucose_level"]


def test_prepare_datasets_excludes_high_missing_features():
    idx = pd.date_range("2022-01-01", periods=20, freq="5min")
    df = pd.DataFrame(
        {
            "raw_glucose_level": np.arange(20, dtype=np.float32),
            "glucose_level": np.arange(20, dtype=np.float32),
            "basis_gsr": np.ones(20, dtype=np.float32),
            "basis_sleep": np.zeros(20, dtype=np.float32),
            "basis_heart_rate": np.full(20, 80, dtype=np.float32),
            "acceleration": np.full(20, 1, dtype=np.float32),
        },
        index=idx,
    )
    cfg = {
        "window": {"lookback_minutes": 10, "forecast_minutes": 10, "freq_minutes": 5},
        "feature_engineering": {
            "exclude_features": ["basis_heart_rate", "acceleration"],
        },
    }

    datasets = prepare_datasets({"p": df}, {"p": df}, cfg)

    assert datasets["full_feature_cols"] == ["glucose_level", "basis_gsr", "basis_sleep"]
    assert datasets["X_train_3d"].shape[-1] == 3


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
