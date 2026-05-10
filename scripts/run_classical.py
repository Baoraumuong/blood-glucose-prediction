#!/usr/bin/env python
"""
scripts/run_classical.py
------------------------
Run all classical ML models (LR, SVR, XGBoost, DT, RF, KNN) only.
Equivalent to:
    python main.py --models lr svr xgb dt rf knn --skip-deep

Usage
-----
    python scripts/run_classical.py
    python scripts/run_classical.py --config configs/config.yaml
    python scripts/run_classical.py --no-plots
"""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config",   default="configs/config.yaml")
parser.add_argument("--no-plots", action="store_true")
args = parser.parse_args()

cmd = [
    sys.executable, "main.py",
    "--config",  args.config,
    "--models",  "lr", "svr", "xgb", "dt", "rf", "knn",
    "--skip-deep",
]
if args.no_plots:
    cmd.append("--no-plots")

sys.exit(subprocess.call(cmd))
