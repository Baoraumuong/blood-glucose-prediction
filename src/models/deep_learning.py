"""
LSTM and GRU models built with Keras / TensorFlow.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from src.evaluation.metrics import ResultStore, compute_all_metrics
from src.models.persistence import artifact_dir, save_pickle_artifact

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
    """Construct and compile a two-layer LSTM or GRU."""
    from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    rnn_layer = LSTM if rnn_type.lower() == "lstm" else GRU
    model = Sequential(
        [
            Input(shape=input_shape),
            rnn_layer(units[0], return_sequences=(len(units) > 1)),
            Dropout(dropout),
            rnn_layer(units[1]),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(output_steps),
        ]
    )
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model


def _build_stacked_lstm(
    input_shape: tuple[int, int],
    units: int,
    dropout: float,
    dense_units: list[int],
    learning_rate: float,
) -> Any:
    """Construct the requested two-layer stacked LSTM."""
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.losses import Poisson
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units),
            Dense(dense_units[0], activation="relu"),
            Dense(dense_units[1], activation="relu"),
            Dense(1, activation="exponential"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate), loss=Poisson(name="negative_log_likelihood"))
    return model


def _last_horizon(y: np.ndarray) -> np.ndarray:
    return y[:, -1:].astype(np.float32)


def _inverse_last_horizon(target_scaler: Any, y_last_scaled: np.ndarray, steps: int) -> np.ndarray:
    padded = np.zeros((len(y_last_scaled), steps), dtype=np.float32)
    padded[:, -1:] = y_last_scaled
    return target_scaler.inverse_transform_y(padded)[:, -1:]


def _run_deep_model(
    rnn_type: str,
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    dl_cfg = cfg.get("deep_learning", {})
    model_cfg = dl_cfg.get(rnn_type.lower(), {})
    w = cfg["window"]
    fc_steps = w["forecast_minutes"] // w["freq_minutes"]
    name = rnn_type.upper()

    def make_callbacks() -> list[Any]:
        return [
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
            v.get("feature_cols"),
            v.get("use_savgol"),
        )
        for v in datasets.get("deep_variants", [])
    ]

    if not deep_iter:
        deep_iter = [
            (
                "with features",
                datasets["X_train_3d"],
                datasets["X_test_3d"],
                datasets["y_train_s"],
                datasets["y_test"],
                datasets["scaler_full"],
                datasets.get("full_feature_cols"),
                True,
            ),
            (
                "baseline",
                datasets["X_train_3d_base"],
                datasets["X_test_3d_base"],
                datasets["y_train_s"],
                datasets["y_test"],
                datasets["scaler_base"],
                datasets.get("base_feature_cols"),
                True,
            ),
        ]

    for tag, X3d_tr, X3d_te, ytr_s, yte_orig, target_scaler, feature_cols, use_savgol in deep_iter:
        logger.info("Training %s (%s) ...", name, tag)
        n_folds = cfg.get("evaluation", {}).get("n_folds", 5)
        shuffle = cfg.get("evaluation", {}).get("kfold_shuffle", False)
        fold_metrics: list[dict[str, float]] = []
        kf = KFold(n_splits=n_folds, shuffle=shuffle)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X3d_tr), start=1):
            logger.info("  %s %s CV fold %d/%d", name, tag, fold_idx, n_folds)
            fold_model = _build_rnn(
                rnn_type=rnn_type,
                input_shape=(X3d_tr.shape[1], X3d_tr.shape[2]),
                output_steps=fc_steps,
                units=model_cfg.get("units", [128, 64]),
                dropout=model_cfg.get("dropout", 0.2),
                dense_units=model_cfg.get("dense_units", 32),
                learning_rate=dl_cfg.get("learning_rate", 1e-3),
            )
            fold_model.fit(
                X3d_tr[tr_idx],
                ytr_s[tr_idx],
                epochs=dl_cfg.get("epochs", 60),
                batch_size=dl_cfg.get("batch_size", 256),
                validation_data=(X3d_tr[val_idx], ytr_s[val_idx]),
                callbacks=make_callbacks(),
                verbose=0,
            )
            val_preds_s = fold_model.predict(X3d_tr[val_idx], verbose=0)
            val_preds = target_scaler.inverse_transform_y(val_preds_s)
            val_actual = target_scaler.inverse_transform_y(ytr_s[val_idx])
            fold_metrics.append(compute_all_metrics(val_actual, val_preds))

            import tensorflow as tf

            tf.keras.backend.clear_session()

        cv_metrics = {
            f"cv_{metric}": float(np.mean([m[metric] for m in fold_metrics]))
            for metric in fold_metrics[0]
        }

        model = _build_rnn(
            rnn_type=rnn_type,
            input_shape=(X3d_tr.shape[1], X3d_tr.shape[2]),
            output_steps=fc_steps,
            units=model_cfg.get("units", [128, 64]),
            dropout=model_cfg.get("dropout", 0.2),
            dense_units=model_cfg.get("dense_units", 32),
            learning_rate=dl_cfg.get("learning_rate", 1e-3),
        )
        model.fit(
            X3d_tr,
            ytr_s,
            epochs=dl_cfg.get("epochs", 60),
            batch_size=dl_cfg.get("batch_size", 256),
            validation_split=dl_cfg.get("validation_split", 0.1),
            callbacks=make_callbacks(),
            verbose=0,
        )

        preds_s = model.predict(X3d_te, verbose=0)
        preds = target_scaler.inverse_transform_y(preds_s)
        test_metrics = compute_all_metrics(yte_orig, preds)

        model_dir = artifact_dir(cfg, name, tag)
        model_path = model_dir / "model.keras"
        model.save(model_path)
        metadata_path = save_pickle_artifact(
            cfg,
            name,
            tag,
            "metadata.pkl",
            {
                "model_name": name,
                "feature_tag": tag,
                "scaler": target_scaler,
                "feature_cols": feature_cols,
                "use_savgol": use_savgol,
                "input_shape": tuple(X3d_tr.shape[1:]),
            },
        )
        logger.info("Saved %s (%s) model: %s", name, tag, model_path)
        logger.info("Saved %s (%s) metadata: %s", name, tag, metadata_path)
        store.add(name, tag, cv_metrics, test_metrics, preds)

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


def run_stacked_lstm(
    train_masters: dict[str, Any],
    test_masters: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.models import load_model

    from src.features.windowing import prepare_kalman_datasets

    dl_cfg = cfg.get("deep_learning", {})
    model_cfg = dl_cfg.get("stacked_lstm", {})
    w = cfg["window"]
    fc_steps = w["forecast_minutes"] // w["freq_minutes"]
    name = "STACKED_LSTM"

    kalman_datasets = prepare_kalman_datasets(train_masters, test_masters, cfg)
    deep_iter = [
        (
            v["tag"],
            v["X_train_3d"],
            v["X_test_3d"],
            v["y_train_s"],
            v["y_train"],
            v["y_test"],
            v["scaler"],
            v.get("feature_cols"),
        )
        for v in kalman_datasets.get("deep_variants", [])
    ]

    def build_model(input_shape: tuple[int, int]) -> Any:
        return _build_stacked_lstm(
            input_shape=input_shape,
            units=model_cfg.get("units", 128),
            dropout=model_cfg.get("dropout", 0.2),
            dense_units=model_cfg.get("dense_units", [512, 128]),
            learning_rate=dl_cfg.get("learning_rate", 1e-3),
        )

    for tag, X3d_tr, X3d_te, ytr_s, ytr_orig, yte_orig, target_scaler, feature_cols in deep_iter:
        logger.info("Training %s (%s) ...", name, tag)
        n_folds = cfg.get("evaluation", {}).get("n_folds", 5)
        shuffle = cfg.get("evaluation", {}).get("kfold_shuffle", False)
        fold_metrics: list[dict[str, float]] = []
        kf = KFold(n_splits=n_folds, shuffle=shuffle)
        ytr_last_s = _last_horizon(ytr_s)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X3d_tr), start=1):
            logger.info("  %s %s CV fold %d/%d", name, tag, fold_idx, n_folds)
            fold_model = build_model((X3d_tr.shape[1], X3d_tr.shape[2]))
            fold_model.fit(
                X3d_tr[tr_idx],
                ytr_last_s[tr_idx],
                epochs=model_cfg.get("epochs", dl_cfg.get("epochs", 6000)),
                batch_size=model_cfg.get("batch_size", dl_cfg.get("batch_size", 128)),
                validation_data=(X3d_tr[val_idx], ytr_last_s[val_idx]),
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=model_cfg.get(
                            "early_stopping_patience",
                            dl_cfg.get("early_stopping_patience", 128),
                        ),
                        restore_best_weights=True,
                    ),
                ],
                verbose=0,
            )
            val_preds_s = fold_model.predict(X3d_tr[val_idx], verbose=0)
            val_preds = _inverse_last_horizon(target_scaler, val_preds_s, fc_steps)
            val_actual = ytr_orig[val_idx, -1:]
            fold_metrics.append(compute_all_metrics(val_actual, val_preds))

            import tensorflow as tf

            tf.keras.backend.clear_session()

        cv_metrics = {
            f"cv_{metric}": float(np.mean([m[metric] for m in fold_metrics]))
            for metric in fold_metrics[0]
        }

        model_dir = artifact_dir(cfg, name, tag)
        model_path = model_dir / "model.keras"
        if model_path.exists():
            try:
                model = load_model(model_path)
                logger.info("Loaded existing %s (%s) checkpoint: %s", name, tag, model_path)
            except Exception as exc:
                logger.warning("Could not load %s; starting a fresh model: %s", model_path, exc)
                model = build_model((X3d_tr.shape[1], X3d_tr.shape[2]))
        else:
            model = build_model((X3d_tr.shape[1], X3d_tr.shape[2]))

        model.fit(
            X3d_tr,
            ytr_last_s,
            epochs=model_cfg.get("epochs", dl_cfg.get("epochs", 6000)),
            batch_size=model_cfg.get("batch_size", dl_cfg.get("batch_size", 128)),
            validation_split=model_cfg.get("validation_split", dl_cfg.get("validation_split", 0.1)),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=model_cfg.get(
                        "early_stopping_patience",
                        dl_cfg.get("early_stopping_patience", 128),
                    ),
                    restore_best_weights=True,
                ),
                ModelCheckpoint(model_path, monitor="val_loss", save_best_only=False),
            ],
            verbose=0,
        )

        preds_s = model.predict(X3d_te, verbose=0)
        preds = _inverse_last_horizon(target_scaler, preds_s, fc_steps)
        test_actual = yte_orig[:, -1:]
        test_metrics = compute_all_metrics(test_actual, preds)

        model.save(model_path)
        metadata_path = save_pickle_artifact(
            cfg,
            name,
            tag,
            "metadata.pkl",
            {
                "model_name": name,
                "feature_tag": tag,
                "scaler": target_scaler,
                "feature_cols": feature_cols,
                "use_kalman": True,
                "input_shape": tuple(X3d_tr.shape[1:]),
                "target_horizon_step": fc_steps,
            },
        )
        logger.info("Saved %s (%s) model: %s", name, tag, model_path)
        logger.info("Saved %s (%s) metadata: %s", name, tag, metadata_path)
        store.add(name, tag, cv_metrics, test_metrics, preds)

        import tensorflow as tf

        tf.keras.backend.clear_session()
