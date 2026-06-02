"""
Utilities for statistical time-series models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fill_series_gaps(series: pd.Series) -> pd.Series:
    """
    Return a finite numeric series for statsmodels endog inputs.

    Master DataFrames intentionally preserve missing CGM rows after alignment.
    SARIMAX can be brittle when those gaps are passed through repeated
    apply/forecast calls, so statistical models use interpolated history while
    evaluation still uses the original raw target windows.
    """
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = clean.interpolate(method="time" if isinstance(clean.index, pd.DatetimeIndex) else "linear")
    clean = clean.ffill().bfill()
    if isinstance(clean.index, pd.DatetimeIndex) and clean.index.freq is None:
        try:
            clean.index.freq = pd.infer_freq(clean.index)
        except ValueError:
            pass
    return clean.dropna()


def is_finite_window(values: np.ndarray) -> bool:
    """True when every value in a forecast/evaluation window is finite."""
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())
