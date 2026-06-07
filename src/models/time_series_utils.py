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
    clean = ensure_supported_series(clean)
    return clean.dropna()


def ensure_supported_series(series: pd.Series) -> pd.Series:
    """Normalize series indexes for statsmodels time-series prediction."""
    s = ensure_monotonic_series(series)
    if not isinstance(s.index, pd.DatetimeIndex):
        return s

    if s.index.freq is None:
        freq = pd.infer_freq(s.index)
        if freq is not None:
            try:
                s.index.freq = freq
            except Exception:
                pass

    if s.index.freq is None:
        s.index = pd.RangeIndex(len(s))
    return s


def normalize_series_and_exog(
    series: pd.Series,
    exog: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame | None]:
    """Normalize endogenous and exogenous indexes for statsmodels."""
    series = ensure_supported_series(series)
    if exog is None:
        return series, None

    if isinstance(series.index, pd.RangeIndex):
        exog = exog.reset_index(drop=True)
    else:
        exog = exog.reindex(series.index)
    return series, exog


def is_finite_window(values: np.ndarray) -> bool:
    """True when every value in a forecast/evaluation window is finite."""
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def ensure_monotonic_series(series: pd.Series) -> pd.Series:
    """Ensure a Series has a monotonic DatetimeIndex.

    - Converts index to DatetimeIndex when possible.
    - Drops NaT index entries.
    - Sorts by index if not monotonic increasing.
    - Drops duplicate index entries, keeping the first.

    The function returns the (possibly) modified Series.
    """
    if series is None:
        return series

    s = series.copy()
    # Try to coerce index to datetime if it's not already
    try:
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
    except Exception:
        # If coercion fails, return original
        return series

    # Drop rows with NaT index
    if s.index.hasnans:
        s = s[s.index.notna()]

    # Sort and drop duplicates
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    if s.index.duplicated().any():
        s = s[~s.index.duplicated(keep="first")]
    return s
