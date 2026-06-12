"""
models/stacking.py
-------------------
Out-of-Fold (OOF) stacking implementation.

This module generates true OOF predictions for base models (no leakage)
and trains a meta-learner on the OOF features. Current base models
supported: XGBoost (classical, 2-D inputs) and LSTM (deep, 3-D inputs).

Notes
-----
- OOF predictions for each base model are generated using K-fold CV
    (TimeSeriesSplit by default to preserve temporal order).
- Test-set predictions are produced per-fold and averaged to create the
    meta-test features.
- Scaling for classical models is done per-fold using a fresh MinMaxScaler
    fitted on the fold training data to reduce leakage. Deep scaling uses
    the provided target scaler for inverse-transforming predictions.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, TimeSeriesSplit

from src.evaluation.metrics import ResultStore, compute_all_metrics
from src.models.persistence import save_pickle_artifact, artifact_dir
from src.models.deep_learning import _build_rnn

logger = logging.getLogger(__name__)


def _generate_oof_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: dict[str, Any],
    n_folds: int = 5,
    time_series: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate OOF train preds and averaged test preds for XGBoost base model."""
    n_train = X_train.shape[0]
    horizon = y_train.shape[1]
    oof = np.zeros((n_train, horizon), dtype=float)
    test_fold_preds = []

    splitter = TimeSeriesSplit(n_splits=n_folds) if time_series else KFold(n_splits=n_folds, shuffle=True)

    for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train), start=1):
        xtr, xval = X_train[tr_idx], X_train[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        # per-fold scaling to reduce leakage
        scaler = MinMaxScaler()
        xtr_s = scaler.fit_transform(xtr)
        xval_s = scaler.transform(xval)
        xtest_s = scaler.transform(X_test)

        model = xgb.XGBRegressor(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", 4),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            verbosity=0,
            n_jobs=-1,
            random_state=cfg.get("seed", 42),
        )
        model.fit(xtr_s, ytr)
        oof[val_idx] = model.predict(xval_s)
        test_fold_preds.append(model.predict(xtest_s))

    test_mean = np.mean(np.stack(test_fold_preds, axis=0), axis=0)
    return oof, test_mean


def _generate_oof_lstm(
    X3d_train: np.ndarray,
    y_train_s: np.ndarray,
    X3d_test: np.ndarray,
    target_scaler: Any,
    cfg: dict[str, Any],
    n_folds: int = 5,
    time_series: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate OOF train preds and averaged test preds for LSTM base model.

    Note: this trains K separate LSTM models (one per fold) which can be slow.
    """
    from keras.callbacks import EarlyStopping

    n_train = X3d_train.shape[0]
    horizon = target_scaler.inverse_transform_y(y_train_s[:1]).shape[1]
    oof = np.zeros((n_train, horizon), dtype=float)
    test_fold_preds = []

    splitter = TimeSeriesSplit(n_splits=n_folds) if time_series else KFold(n_splits=n_folds, shuffle=True)
    dl_cfg = cfg.get("deep_learning", {})
    model_cfg = dl_cfg.get("lstm", {})

    for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X3d_train), start=1):
        Xtr, Xval = X3d_train[tr_idx], X3d_train[val_idx]
        ytr_s, yval_s = y_train_s[tr_idx], y_train_s[val_idx]

        model = _build_rnn(
            rnn_type="lstm",
            input_shape=(Xtr.shape[1], Xtr.shape[2]),
            output_steps=ytr_s.shape[1],
            units=model_cfg.get("units", [128, 64]),
            dropout=model_cfg.get("dropout", 0.2),
            dense_units=model_cfg.get("dense_units", 32),
            learning_rate=dl_cfg.get("learning_rate", 1e-3),
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=dl_cfg.get("early_stopping_patience", 5), restore_best_weights=True)
        ]

        model.fit(
            Xtr,
            ytr_s,
            validation_data=(Xval, yval_s),
            epochs=dl_cfg.get("epochs", 30),
            batch_size=dl_cfg.get("batch_size", 256),
            callbacks=callbacks,
            verbose=0,
        )

        val_preds_s = model.predict(Xval, verbose=0)
        val_preds = target_scaler.inverse_transform_y(val_preds_s)
        oof[val_idx] = val_preds

        test_preds_s = model.predict(X3d_test, verbose=0)
        test_preds = target_scaler.inverse_transform_y(test_preds_s)
        test_fold_preds.append(test_preds)

        import tensorflow as tf
        tf.keras.backend.clear_session()

    test_mean = np.mean(np.stack(test_fold_preds, axis=0), axis=0)
    return oof, test_mean


def _evaluate_meta_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    time_series: bool,
) -> dict[str, float]:
    """Cross-validate the meta learner on the meta-training features."""
    cv = TimeSeriesSplit(n_splits=n_folds) if time_series else KFold(n_splits=n_folds, shuffle=True)
    metrics_list = []
    for tr_idx, val_idx in cv.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        preds = model.predict(X[val_idx])
        metrics_list.append(compute_all_metrics(y[val_idx], preds))
    return {f"cv_{k}": float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}


def run_stacking(datasets: dict[str, Any], store: ResultStore, cfg: dict[str, Any]) -> None:
    """Run OOF stacking for each available dataset variant (tags).

    Iterates through `datasets["classical_variants"]` and runs OOF
    generation for the configured base models for each variant. Results
    are added to `store` under the corresponding tag.
    """
    stk_cfg = cfg.get("stacking", {})
    if not stk_cfg.get("enabled", True):
        logger.info("Stacking disabled in config.")
        return

    n_folds = stk_cfg.get("n_folds", cfg.get("evaluation", {}).get("n_folds", 5))
    time_series = stk_cfg.get("time_series", True)
    base_models = stk_cfg.get("base_models", ["xgboost", "lstm"])

    variants = datasets.get("classical_variants") or datasets.get("variants") or []
    if not variants:
        logger.warning("No dataset variants found for stacking; skipping.")
        return

    for variant in variants:
        tag = variant.get("tag")
        logger.info("Running stacking for variant: %s", tag)

        X2d_tr = variant.get("X_train_2d")
        X2d_te = variant.get("X_test_2d")
        X3d_tr = variant.get("X_train_3d")
        X3d_te = variant.get("X_test_3d")
        y_tr = variant.get("y_train")
        y_te = variant.get("y_test")
        y_train_s = variant.get("y_train_s")
        target_scaler = variant.get("scaler")

        if X2d_tr is None or y_tr is None:
            logger.warning(" Variant %s missing data; skipping.", tag)
            continue

        meta_train_parts = [X2d_tr]
        meta_test_parts = [X2d_te]

        for bm in base_models:
            if bm.lower() == "xgboost":
                logger.info("  Generating OOF for XGBoost (%s)", tag)
                xgb_cfg = cfg.get("models", {}).get("xgboost", {})
                oof_tr, test_mean = _generate_oof_xgb(X2d_tr, y_tr, X2d_te, xgb_cfg, n_folds=n_folds, time_series=time_series)
                meta_train_parts.append(oof_tr)
                meta_test_parts.append(test_mean)
                if stk_cfg.get("save_oof", False):
                    save_pickle_artifact(cfg, "Stacking", f"{tag}_xgboost", "oof_xgb.pkl", {"oof": oof_tr})

            elif bm.lower() == "lstm":
                if X3d_tr is None or y_train_s is None:
                    logger.warning("  LSTM OOF skipped for %s (missing 3D or scaled y)", tag)
                    continue
                logger.info("  Generating OOF for LSTM (%s)", tag)
                oof_tr, test_mean = _generate_oof_lstm(X3d_tr, y_train_s, X3d_te, target_scaler, cfg, n_folds=n_folds, time_series=time_series)
                meta_train_parts.append(oof_tr)
                meta_test_parts.append(test_mean)
                if stk_cfg.get("save_oof", False):
                    save_pickle_artifact(cfg, "Stacking", f"{tag}_lstm", "oof_lstm.pkl", {"oof": oof_tr})

            else:
                logger.warning("  Unknown base model '%s' for stacking; skipping.", bm)

        # concatenate meta features (flatten any multi-output parts along columns)
        X_meta_tr = np.concatenate([p if p.ndim == 2 else p.reshape(p.shape[0], -1) for p in meta_train_parts], axis=1)
        X_meta_te = np.concatenate([p if p.ndim == 2 else p.reshape(p.shape[0], -1) for p in meta_test_parts], axis=1)

        # train meta-learner
        meta_cfg = stk_cfg.get("meta_model", {})
        base_xgb = xgb.XGBRegressor(
            n_estimators=meta_cfg.get("n_estimators", 200),
            max_depth=meta_cfg.get("max_depth", 6),
            learning_rate=meta_cfg.get("learning_rate", 0.05),
            subsample=meta_cfg.get("subsample", 0.8),
            colsample_bytree=meta_cfg.get("colsample_bytree", 0.8),
            verbosity=0,
            n_jobs=-1,
            random_state=cfg.get("seed", 42),
        )
        meta = MultiOutputRegressor(base_xgb, n_jobs=-1)
        logger.info("  Training meta-learner for %s ...", tag)

        cv_metrics = _evaluate_meta_cv(meta, X_meta_tr, y_tr, n_folds=n_folds, time_series=time_series)
        meta.fit(X_meta_tr, y_tr)
        preds = meta.predict(X_meta_te)
        metrics = compute_all_metrics(y_te, preds)

        store.add("Stacking-OOF", tag, cv_metrics, metrics, preds)
        logger.info("  Completed stacking for %s — CV RMSE=%.4f | Test RMSE=%.4f", tag, cv_metrics.get("cv_rmse", float("nan")), metrics.get("rmse", float("nan")))
