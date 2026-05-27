#!/usr/bin/env python
"""
main.py
-------
Orchestration entry point for the blood-glucose prediction pipeline.

Usage
-----
    python main.py                          # run all enabled models
    python main.py --models lr rf xgb      # run subset of models
    python main.py --config configs/custom.yaml
    python main.py --skip-deep             # skip LSTM/GRU (faster dev run)
    python main.py --no-plots              # skip figure generation
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make sure the project root is on sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config, get_window_params
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.data.loader import find_xml_files, build_all_masters
from src.features.windowing import prepare_datasets
from src.evaluation.metrics import ResultStore
from src.evaluation.visualizer import (
    plot_rmse_r2_bars,
    plot_feature_impact,
    plot_best_model_predictions,
    plot_horizon_rmse,
    plot_summary_heatmap,
)
from src.models.classical import (
    run_linear_regression,
    run_svr,
    run_xgboost,
    run_decision_tree,
    run_random_forest,
    run_knn,
)
from src.models.arima_model import run_arima
from src.models.sarimax_model import run_sarimax
from src.models.deep_learning import run_lstm, run_gru


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood Glucose Prediction Pipeline")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["lr", "svr", "xgb", "dt", "rf", "knn", "arima", "sarimax", "lstm", "gru"],
        help="Run only the specified models (default: all enabled in config)",
    )
    parser.add_argument(
        "--skip-deep", action="store_true",
        help="Skip LSTM and GRU (useful for quick experiments)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots",
    )
    return parser.parse_args()


# ─── Pipeline ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Config & logging ──────────────────────────────────────────────────
    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    logger  = get_logger(
        "main",
        log_dir=cfg["paths"]["logs_dir"],
        level=log_cfg.get("level", "INFO"),
        log_to_file=log_cfg.get("log_to_file", True),
        filename_prefix=log_cfg.get("filename_prefix", "bg_prediction"),
    )

    set_global_seed(cfg.get("seed", 42))
    logger.info("=" * 70)
    logger.info("Blood Glucose Prediction Pipeline  –  OhioT1DM")
    logger.info("Config: %s", args.config)
    logger.info("=" * 70)

    t0 = time.time()

    # ── 1. Data loading ───────────────────────────────────────────────────
    logger.info("[1/5] Discovering XML files …")
    ds_cfg = cfg["dataset"]
    train_files, test_files = find_xml_files(
        cfg["paths"]["data_dir"],
        ds_cfg["cohorts"],
        ds_cfg.get("train_suffix", "-ws-training.xml"),
        ds_cfg.get("test_suffix",  "-ws-testing.xml"),
    )

    logger.info("[1/5] Building patient master DataFrames …")
    train_masters = build_all_masters(train_files, cfg)
    test_masters  = build_all_masters(test_files,  cfg)

    if not train_masters:
        logger.error("No training data loaded. Check data_dir in config.")
        sys.exit(1)

    # ── 2. Feature engineering & windowing ───────────────────────────────
    logger.info("[2/5] Preparing windowed datasets …")
    datasets = prepare_datasets(train_masters, test_masters, cfg)
    lb_steps, fc_steps, freq = get_window_params(cfg)

    # ── 3. Model training & evaluation ───────────────────────────────────
    logger.info("[3/5] Training models …")
    store    = ResultStore()
    models_cfg = cfg.get("models", {})

    # Determine which models to run
    run_all   = args.models is None
    requested = set(args.models or [])

    def should_run(short_name: str, config_key: str) -> bool:
        enabled_in_cfg = models_cfg.get(config_key, {}).get("enabled", True)
        if not enabled_in_cfg:
            return False
        if run_all:
            return True
        return short_name in requested

    if should_run("lr",    "linear_regression"):
        run_linear_regression(datasets, store, cfg)

    if should_run("svr",   "svr"):
        run_svr(datasets, store, cfg)

    if should_run("xgb",  "xgboost"):
        run_xgboost(datasets, store, cfg)

    if should_run("dt",    "decision_tree"):
        run_decision_tree(datasets, store, cfg)

    if should_run("rf",    "random_forest"):
        run_random_forest(datasets, store, cfg)

    if should_run("knn",   "knn"):
        run_knn(datasets, store, cfg)

    if should_run("arima", "arima"):
        run_arima(train_masters, test_masters, store, cfg)

    if should_run("sarimax", "sarimax"):
        run_sarimax(train_masters, test_masters, store, cfg)

    if not args.skip_deep:
        if should_run("lstm", "lstm"):
            run_lstm(datasets, store, cfg)
        if should_run("gru",  "gru"):
            run_gru(datasets, store, cfg)
    else:
        logger.info("Skipping LSTM / GRU (--skip-deep flag set)")

    # ── 4. Save results ───────────────────────────────────────────────────
    logger.info("[4/5] Saving results …")
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    res_df = store.to_dataframe()
    csv_path = os.path.join(results_dir, "model_results.csv")
    store.save_csv(csv_path)

    logger.info("\n%s", res_df.to_string(index=False))

    # ── 5. Plots ──────────────────────────────────────────────────────────
    if not args.no_plots and not res_df.empty:
        logger.info("[5/5] Generating plots …")
        plots_dir = cfg["paths"]["plots_dir"]
        os.makedirs(plots_dir, exist_ok=True)

        plot_rmse_r2_bars(res_df, save_path=os.path.join(plots_dir, "rmse_r2_bars.png"))
        plot_feature_impact(res_df, save_path=os.path.join(plots_dir, "feature_impact.png"))
        plot_summary_heatmap(res_df, save_path=os.path.join(plots_dir, "summary_heatmap.png"))
        plot_best_model_predictions(
            res_df, store, datasets["y_test"],
            save_path=os.path.join(plots_dir, "best_model_predictions.png"),
        )
        plot_horizon_rmse(
            res_df, store, datasets["y_test"],
            forecast_steps=fc_steps, freq_minutes=freq,
            save_path=os.path.join(plots_dir, "horizon_rmse.png"),
        )
        logger.info("All plots saved to %s", plots_dir)
    else:
        logger.info("[5/5] Skipping plots (--no-plots or empty results)")

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("Pipeline completed in %.1f s", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
