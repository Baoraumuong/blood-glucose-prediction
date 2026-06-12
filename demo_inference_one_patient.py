#!/usr/bin/env python
"""
Run demo inference for one saved model artifact on one patient's test data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_all_metrics
from src.features.windowing import RAW_TARGET_COL, TARGET_COL, create_multistep_dataset
from src.models.persistence import load_pickle_artifact, safe_name
from src.utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo inference on one patient's test split.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--patient", required=True, help="Patient key, e.g. 540_2020")
    parser.add_argument("--model", required=True, help="Model name, e.g. RandomForest, LSTM, ARIMA")
    parser.add_argument("--features", required=True, help='Feature tag, e.g. "baseline" or "with features"')
    parser.add_argument(
        "--masters-dir",
        default=None,
        help="Directory containing processed train/test master CSVs; defaults to output/processed.",
    )
    return parser.parse_args()


def _load_test_master(cfg: dict, patient: str, masters_dir: str | None) -> pd.DataFrame:
    processed_dir = Path(
        masters_dir
        or cfg["paths"].get("processed_dir")
        or Path(cfg["paths"].get("output_dir", "output")) / "processed"
    )
    path = processed_dir / "test" / f"{patient}_test_master.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test master not found: {path}")
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    df.index.name = None
    return df.sort_index()


def _window_patient(df: pd.DataFrame, cfg: dict, feature_cols: list[str]) -> tuple:
    w = cfg["window"]
    lookback_steps = w["lookback_minutes"] // w["freq_minutes"]
    forecast_steps = w["forecast_minutes"] // w["freq_minutes"]
    target_col = RAW_TARGET_COL if RAW_TARGET_COL in df.columns else TARGET_COL
    return create_multistep_dataset(
        df,
        target_col=target_col,
        lookback_steps=lookback_steps,
        forecast_steps=forecast_steps,
        feature_cols=feature_cols,
        freq_minutes=w["freq_minutes"],
    )


def _artifact_base(cfg: dict, model_name: str, feature_tag: str) -> Path:
    return (
        Path(cfg["paths"].get("checkpoints_dir", "output/checkpoints"))
        / safe_name(model_name)
        / safe_name(feature_tag)
    )


def _run_classical(cfg: dict, model_name: str, feature_tag: str, df: pd.DataFrame) -> tuple:
    artifact = load_pickle_artifact(_artifact_base(cfg, model_name, feature_tag) / "model.pkl")
    X, y = _window_patient(df, cfg, artifact["feature_cols"])
    X_s = artifact["scaler"].transform_2d(X)[0]
    preds = artifact["model"].predict(X_s)
    return preds, y


def _run_deep(cfg: dict, model_name: str, feature_tag: str, df: pd.DataFrame) -> tuple:
    from tensorflow.keras.models import load_model

    base = _artifact_base(cfg, model_name, feature_tag)
    metadata = load_pickle_artifact(base / "metadata.pkl")
    model = load_model(base / "model.keras")
    X, y = _window_patient(df, cfg, metadata["feature_cols"])
    X_s = metadata["scaler"].transform_3d(X)[0]
    preds_s = model.predict(X_s, verbose=0)
    preds = metadata["scaler"].inverse_transform_y(preds_s)
    return preds, y


def _run_arima(cfg: dict, feature_tag: str, patient: str, df: pd.DataFrame) -> tuple:
    from src.models.arima_model import _forecast_windows

    artifact = load_pickle_artifact(_artifact_base(cfg, "ARIMA", feature_tag) / f"{safe_name(patient)}.pkl")
    test_series = df.get(RAW_TARGET_COL, df[TARGET_COL])
    return _forecast_windows(
        artifact["fit"],
        artifact["train_history"],
        test_series,
        artifact["n_forecast"],
    )


def _run_sarimax(cfg: dict, feature_tag: str, patient: str, df: pd.DataFrame) -> tuple:
    from src.models.sarimax_model import _forecast_windows

    artifact = load_pickle_artifact(_artifact_base(cfg, "SARIMAX", feature_tag) / f"{safe_name(patient)}.pkl")
    test_series = df.get(RAW_TARGET_COL, df[TARGET_COL])
    return _forecast_windows(
        artifact["fit"],
        artifact["train_history"],
        artifact["train_x"],
        test_series,
        df[artifact["exog_cols"]],
        artifact["exog_mean"],
        artifact["exog_std"],
        artifact["n_forecast"],
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    df = _load_test_master(cfg, args.patient, args.masters_dir)
    model_name = args.model
    feature_tag = args.features

    if model_name.upper() == "ARIMA":
        preds, actuals = _run_arima(cfg, feature_tag, args.patient, df)
    elif model_name.upper() == "SARIMAX":
        preds, actuals = _run_sarimax(cfg, feature_tag, args.patient, df)
    elif model_name.upper() in {"LSTM", "GRU"}:
        preds, actuals = _run_deep(cfg, model_name.upper(), feature_tag, df)
    else:
        preds, actuals = _run_classical(cfg, model_name, feature_tag, df)

    if preds is None or actuals is None:
        raise RuntimeError("No valid prediction windows were produced.")

    metrics = compute_all_metrics(actuals, preds)
    print(f"Model: {model_name} ({feature_tag})")
    print(f"Patient: {args.patient}")
    print(f"Windows: {len(preds)}")
    print(
        "RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} MARD={mard:.4f}".format(
            **metrics,
        )
    )


if __name__ == "__main__":
    main()
