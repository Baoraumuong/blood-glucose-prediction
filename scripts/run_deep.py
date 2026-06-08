#!/usr/bin/env python
"""
scripts/run_deep.py
-------------------
Run LSTM, GRU, and STACKED_LSTM models only.

Usage
-----
    python scripts/run_deep.py
    python scripts/run_deep.py --config configs/config.yaml
    python scripts/run_deep.py --no-plots
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
    "--config", args.config,
    "--models", "lstm", "gru", "stacked_lstm",
]
if args.no_plots:
    cmd.append("--no-plots")

sys.exit(subprocess.call(cmd))
