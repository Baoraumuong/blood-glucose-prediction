"""
utils/logger.py
---------------
Centralised logging setup – writes to console and optionally a dated log file.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(
    name: str,
    log_dir: str | Path = "output/logs",
    level: str = "INFO",
    log_to_file: bool = True,
    filename_prefix: str = "bg_prediction",
) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    log_dir : str | Path
        Directory where log files are written.
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_to_file : bool
        Whether to write logs to a file in addition to stdout.
    filename_prefix : str
        Prefix used for the log file name.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers if called multiple times
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    if not root_logger.handlers:
        root_logger.addHandler(ch)

    # ── File handler ─────────────────────────────────────────────────────
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{filename_prefix}_{timestamp}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        root_logger.addHandler(fh)

    return logger
