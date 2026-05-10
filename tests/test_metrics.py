"""
tests/test_metrics.py
---------------------
Unit tests for evaluation metrics.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import rmse, mae, r2, mard, compute_all_metrics, ResultStore


def test_rmse_perfect():
    y = np.array([[100.0, 110.0, 120.0]])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([[0.0, 0.0]])
    y_pred = np.array([[3.0, 4.0]])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(25 / 2))


def test_mae_known():
    y_true = np.array([[0.0, 0.0]])
    y_pred = np.array([[3.0, 4.0]])
    assert mae(y_true, y_pred) == pytest.approx(3.5)


def test_r2_perfect():
    y = np.array([[100.0, 110.0, 120.0]])
    assert r2(y, y) == pytest.approx(1.0)


def test_mard_known():
    y_true = np.array([[100.0, 200.0]])
    y_pred = np.array([[110.0, 220.0]])
    # |100-110|/100 = 0.10, |200-220|/200 = 0.10 → mean = 10 %
    assert mard(y_true, y_pred) == pytest.approx(10.0)


def test_compute_all_metrics_keys():
    y = np.array([[100.0, 110.0]])
    metrics = compute_all_metrics(y, y)
    assert set(metrics.keys()) == {"rmse", "mae", "r2", "mard"}


def test_result_store_add_and_dataframe():
    store = ResultStore()
    cv_m  = {"cv_rmse": 5.0, "cv_r2": 0.9}
    te_m  = {"rmse": 4.5, "mae": 3.0, "r2": 0.92, "mard": 2.5}
    preds = np.zeros((10, 6))
    store.add("TestModel", "with features", cv_m, te_m, preds)
    df = store.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["model"] == "TestModel"
    assert df.iloc[0]["test_rmse"] == pytest.approx(4.5)


def test_result_store_get_predictions():
    store = ResultStore()
    preds = np.ones((5, 6))
    store.add("M", "baseline", {"cv_rmse": 1.0}, {"rmse": 1.0, "mae": 1.0, "r2": 0.5, "mard": 1.0}, preds)
    retrieved = store.get_predictions("M", "baseline")
    assert retrieved is preds
    assert store.get_predictions("M", "with features") is None
