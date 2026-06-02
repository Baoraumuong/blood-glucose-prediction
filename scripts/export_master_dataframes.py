#!/usr/bin/env python
"""
Export aligned per-patient master DataFrames used by the modeling pipeline.

This script intentionally uses the same loader functions as main.py, so the
CSV outputs match the DataFrames that are fed into prepare_datasets().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import build_all_masters, find_xml_files, save_master_dataframes  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export model-ready OhioT1DM master DataFrames to CSV."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/processed",
        help="Directory where final master CSV files are saved.",
    )
    return parser.parse_args()


def _export_split(
    split_name: str,
    masters: dict[str, object],
    output_dir: Path,
) -> None:
    save_master_dataframes(split_name, masters, output_dir)

    for patient_key, df in sorted(masters.items()):
        if df.empty:
            continue
        print(f"{split_name}/{patient_key}: rows={len(df)} cols={len(df.columns)}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)

    ds_cfg = cfg["dataset"]
    train_files, test_files = find_xml_files(
        cfg["paths"]["data_dir"],
        ds_cfg["cohorts"],
        ds_cfg.get("train_suffix", "-ws-training.xml"),
        ds_cfg.get("test_suffix", "-ws-testing.xml"),
    )

    train_masters = build_all_masters(train_files, cfg)
    test_masters = build_all_masters(test_files, cfg)

    _export_split("train", train_masters, output_dir)
    _export_split("test", test_masters, output_dir)


if __name__ == "__main__":
    main()
