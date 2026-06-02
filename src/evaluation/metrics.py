"""
evaluation/metrics.py
---------------------
Metrics and cross-validation helpers used across all models.
Mirrors the evaluation strategy from the original notebook:
  - 5-Fold KFold (no shuffle) for classical models
  - Best validation-loss proxy for deep learning
  - Hold-out test-set evaluation for all models
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual metrics
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true.ravel(), y_pred.ravel()))


def mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Relative Difference (%)."""
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    mask = y_true_flat != 0
    return float(
        np.mean(np.abs(y_true_flat[mask] - y_pred_flat[mask]) / np.abs(y_true_flat[mask])) * 100
    )


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return dict(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        r2=r2(y_true, y_pred),
        mard=mard(y_true, y_pred),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def cv_evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    shuffle: bool = False,
) -> dict[str, float]:
    """
    KFold cross-validation returning mean ± std of RMSE and R².

    Parameters
    ----------
    model : sklearn-compatible estimator (must have fit / predict).
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples, forecast_steps)
    n_folds : int
    shuffle : bool
        False preserves temporal ordering (recommended for time-series).

    Returns
    -------
    dict with keys: cv_rmse, cv_rmse_std, cv_r2, cv_r2_std
    """
    kf = KFold(n_splits=n_folds, shuffle=shuffle)
    rmse_list, r2_list = [], []

    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        preds = model.predict(X[val_idx])
        rmse_list.append(rmse(y[val_idx], preds))
        r2_list.append(r2(y[val_idx], preds))

    return dict(
        cv_rmse=float(np.mean(rmse_list)),
        cv_rmse_std=float(np.std(rmse_list)),
        cv_r2=float(np.mean(r2_list)),
        cv_r2_std=float(np.std(r2_list)),
    )


def test_evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Fit on the full training set and evaluate on the hold-out test set.

    Returns
    -------
    predictions : np.ndarray
    metrics     : dict with rmse, mae, r2, mard
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, compute_all_metrics(y_test, preds)


# ─────────────────────────────────────────────────────────────────────────────
# Result store
# ─────────────────────────────────────────────────────────────────────────────

class ResultStore:
    """Accumulate per-model evaluation results and export them."""

    def __init__(self) -> None:
        self._records: list[dict] = []
        self._preds:   dict[str, np.ndarray] = {}

    def add(
        self,
        model_name: str,
        feature_tag: str,
        cv_metrics: dict[str, float],
        test_metrics: dict[str, float],
        predictions: np.ndarray,
    ) -> None:
        key = f"{model_name} ({feature_tag})"
        normalized_cv_metrics = {
            metric_name if metric_name.startswith("cv_") else f"cv_{metric_name}": round(
                value,
                4,
            )
            for metric_name, value in cv_metrics.items()
        }
        record = dict(
            model=model_name,
            features=feature_tag,
            **normalized_cv_metrics,
            **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()},
        )
        self._records.append(record)
        self._preds[key] = predictions
        logger.info(
            "[%-35s] CV RMSE=%.3f | Test RMSE=%.3f  R²=%.4f",
            key,
            cv_metrics.get("cv_rmse", float("nan")),
            test_metrics["rmse"],
            test_metrics["r2"],
        )

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._records)
        if df.empty:
            return df
        return df.sort_values("test_rmse").reset_index(drop=True)

    def get_predictions(self, model_name: str, feature_tag: str) -> np.ndarray | None:
        return self._preds.get(f"{model_name} ({feature_tag})")

    def save_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)
        logger.info("Results saved to %s", path)
