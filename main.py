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
from src.data.loader import find_xml_files, build_all_masters, load_master_dataframes
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
from src.models.stacking import run_stacking
from src.models.deep_learning import run_lstm, run_gru, run_stacked_lstm


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood Glucose Prediction Pipeline")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=[
            "lr",
            "svr",
            "xgb",
            "dt",
            "rf",
            "knn",
            "arima",
            "sarimax",
            "lstm",
            "gru",
            "stacked_lstm",
            "slstm",
            "stack"
        ],
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
    parser.add_argument(
        "--masters-dir",
        help="Directory containing exported train/test master CSVs (default: paths.processed_dir or output/processed)",
    )
    parser.add_argument(
        "--rebuild-masters",
        action="store_true",
        help="Rebuild patient master DataFrames from XML instead of loading exported CSVs",
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

    processed_dir = Path(
        args.masters_dir
        or cfg["paths"].get("processed_dir")
        or Path(cfg["paths"].get("output_dir", "output")) / "processed"
    )
    expected_keys = {
        f"{pid}_{year}"
        for year, patient_ids in ds_cfg["cohorts"].items()
        for pid in patient_ids
    }
    expected_train = expected_keys
    expected_test = expected_keys
    train_masters = {}
    test_masters = {}

    if not args.rebuild_masters:
        logger.info("[1/5] Loading patient master DataFrames from %s ...", processed_dir)
        train_masters = load_master_dataframes(
            "train",
            processed_dir,
        )
        test_masters = load_master_dataframes(
            "test",
            processed_dir,
        )

        missing_train = expected_train - set(train_masters)
        missing_test = expected_test - set(test_masters)
        if missing_train or missing_test:
            logger.warning(
                "Cached master DataFrames are incomplete (missing train=%s, test=%s). Rebuilding from XML.",
                sorted(missing_train),
                sorted(missing_test),
            )
            train_masters = {}
            test_masters = {}

    if args.rebuild_masters or not train_masters or not test_masters:
        logger.info("[1/5] Building patient master DataFrames from XML ...")
        train_masters = build_all_masters(train_files, cfg)
        test_masters  = build_all_masters(test_files,  cfg)

    if not train_masters:
        logger.error("No training data loaded. Check data_dir in config.")
        sys.exit(1)

    # ── 2. Feature engineering & windowing ───────────────────────────────
    logger.info("[2/5] Preparing windowed datasets …")
    datasets = prepare_datasets(train_masters, test_masters, cfg)
    lb_steps, fc_steps, freq = get_window_params(cfg)
    logger.info(
        "[2/5] Windowed datasets ready: train_windows=%d, test_windows=%d",
        len(datasets["y_train"]),
        len(datasets["y_test"]),
    )

    # ── 3. Model training & evaluation ───────────────────────────────────
    logger.info("[3/5] Training models …")
    store    = ResultStore()
    models_cfg = cfg.get("models", {})

    # Determine which models to run
    run_all   = args.models is None
    requested = set(args.models or [])

    def should_run(short_name: str, config_key: str) -> bool:
        model_section = models_cfg.get(config_key)
        if model_section is None:
            model_section = cfg.get("deep_learning", {}).get(config_key, {})
        enabled_in_cfg = model_section.get("enabled", True)
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
        if should_run("stack", "stacking"):
            run_stacking(datasets, store, cfg)
        if should_run("stacked_lstm", "stacked_lstm") or should_run("slstm", "stacked_lstm"):
            run_stacked_lstm(train_masters, test_masters, store, cfg)
    else:
        logger.info("Skipping LSTM / GRU / STACKED_LSTM (--skip-deep flag set)")

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
