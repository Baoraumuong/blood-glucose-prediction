"""
models/deep_learning.py
-----------------------
LSTM and GRU models built with Keras / TensorFlow.
Architecture and training hyperparameters are read from config.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.evaluation.metrics import ResultStore, compute_all_metrics

logger = logging.getLogger(__name__)


def _build_rnn(
    rnn_type: str,
    input_shape: tuple[int, int],
    output_steps: int,
    units: list[int],
    dropout: float,
    dense_units: int,
    learning_rate: float,
) -> Any:
    """
    Construct a two-layer LSTM or GRU with dropout and dense head.

    Parameters
    ----------
    rnn_type : str   'lstm' | 'gru'
    input_shape : tuple  (lookback_steps, n_features)
    output_steps : int   number of forecast steps
    units : list[int]   hidden units for each RNN layer
    dropout : float      dropout rate between layers
    dense_units : int    units in the intermediate Dense layer
    learning_rate : float

    Returns
    -------
    compiled keras.Sequential
    """
    import tensorflow as tf
    from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    RNNLayer = LSTM if rnn_type.lower() == "lstm" else GRU

    model = Sequential(
        [
            Input(shape=input_shape),
            RNNLayer(units[0], return_sequences=(len(units) > 1)),
            Dropout(dropout),
            RNNLayer(units[1]),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(output_steps),
        ]
    )
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model


def _run_deep_model(
    rnn_type: str,
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    dl_cfg    = cfg.get("deep_learning", {})
    model_cfg = dl_cfg.get(rnn_type.lower(), {})
    w         = cfg["window"]
    fc_steps  = w["forecast_minutes"] // w["freq_minutes"]
    name      = rnn_type.upper()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=dl_cfg.get("early_stopping_patience", 10),
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=dl_cfg.get("reduce_lr_factor", 0.5),
            patience=dl_cfg.get("reduce_lr_patience", 5),
            min_lr=dl_cfg.get("min_lr", 1e-6),
        ),
    ]

    deep_iter = [
        (
            v["tag"],
            v["X_train_3d"],
            v["X_test_3d"],
            v["y_train_s"],
            v["y_test"],
            v["scaler"],
        )
        for v in datasets.get("deep_variants", [])
    ]

    for tag, X3d_tr, X3d_te, ytr_s, yte_orig, target_scaler in deep_iter or [
        ("with features",
         datasets["X_train_3d"],      datasets["X_test_3d"],
         datasets["y_train_s"],       datasets["y_test"], datasets["scaler_full"]),
        ("baseline",
         datasets["X_train_3d_base"], datasets["X_test_3d_base"],
         datasets["y_train_s"],       datasets["y_test"], datasets["scaler_base"]),
    ]:
        logger.info("Training %s (%s) …", name, tag)

        model = _build_rnn(
            rnn_type    = rnn_type,
            input_shape = (X3d_tr.shape[1], X3d_tr.shape[2]),
            output_steps= fc_steps,
            units       = model_cfg.get("units", [128, 64]),
            dropout     = model_cfg.get("dropout", 0.2),
            dense_units = model_cfg.get("dense_units", 32),
            learning_rate=dl_cfg.get("learning_rate", 1e-3),
        )

        model.fit(
            X3d_tr, ytr_s,
            epochs=dl_cfg.get("epochs", 60),
            batch_size=dl_cfg.get("batch_size", 256),
            validation_split=dl_cfg.get("validation_split", 0.1),
            callbacks=callbacks,
            verbose=0,
        )

        hist     = model.history.history
        best_val = min(hist["val_loss"])
        # Approximate CV metric from best validation loss (scaled → original scale proxy)
        y_range  = float(yte_orig.max() - yte_orig.min())
        cv_rmse_approx = float(np.sqrt(best_val) * y_range)

        # Predict and inverse-transform
        preds_s = model.predict(X3d_te, verbose=0)
        preds   = target_scaler.inverse_transform_y(preds_s)

        test_metrics = compute_all_metrics(yte_orig, preds)
        cv_metrics   = dict(
            cv_rmse=cv_rmse_approx,
            cv_r2=float(1 - best_val),
        )
        store.add(name, tag, cv_metrics, test_metrics, preds)

        # Free GPU memory
        import tensorflow as tf
        tf.keras.backend.clear_session()


def run_lstm(
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    _run_deep_model("lstm", datasets, store, cfg)


def run_gru(
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    _run_deep_model("gru", datasets, store, cfg)
