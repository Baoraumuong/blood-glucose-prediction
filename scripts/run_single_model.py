#!/usr/bin/env python
"""
scripts/run_single_model.py
---------------------------
Run a single specified model by short name.

Usage
-----
    python scripts/run_single_model.py --model rf
    python scripts/run_single_model.py --model xgb --no-plots
    python scripts/run_single_model.py --model lstm
"""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

VALID_MODELS = ["lr", "svr", "xgb", "dt", "rf", "knn", "arima", "lstm", "gru"]

parser = argparse.ArgumentParser()
parser.add_argument("--model",    required=True, choices=VALID_MODELS, help="Model short name")
parser.add_argument("--config",   default="configs/config.yaml")
parser.add_argument("--no-plots", action="store_true")
args = parser.parse_args()

cmd = [
    sys.executable, "main.py",
    "--config", args.config,
    "--models", args.model,
]
if args.model not in ("lstm", "gru"):
    cmd.append("--skip-deep")
if args.no_plots:
    cmd.append("--no-plots")

sys.exit(subprocess.call(cmd))
