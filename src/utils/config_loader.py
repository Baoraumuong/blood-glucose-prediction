"""
utils/config_loader.py
----------------------
Load and validate the YAML configuration file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    _ensure_output_dirs(cfg)
    return cfg


def _ensure_output_dirs(cfg: dict) -> None:
    """Create output directories declared in config if they don't exist."""
    paths_cfg = cfg.get("paths", {})
    for key in ("output_dir", "results_dir", "logs_dir", "plots_dir", "checkpoints_dir"):
        path = paths_cfg.get(key)
        if path:
            os.makedirs(path, exist_ok=True)


def get_window_params(cfg: dict) -> tuple[int, int, int]:
    """
    Return (lookback_steps, forecast_steps, freq_minutes) derived from config.
    """
    w = cfg["window"]
    lb_min = w["lookback_minutes"]
    fc_min = w["forecast_minutes"]
    freq   = w["freq_minutes"]
    return lb_min // freq, fc_min // freq, freq
