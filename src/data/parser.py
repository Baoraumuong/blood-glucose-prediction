"""
data/parser.py
--------------
XML parser for the OhioT1DM dataset.
Supports all standard physiological features and both 2018 / 2020 cohorts.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Timestamp formats encountered across both cohort releases
_TS_FORMATS = (
    "%d-%m-%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
)


def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Try each known timestamp format; return None on failure."""
    if isinstance(ts_str, datetime):
        return ts_str
    if not ts_str or pd.isna(ts_str):
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(str(ts_str).strip(), fmt)
        except ValueError:
            continue
    logger.warning("Cannot parse timestamp: %s", ts_str)
    return None


class OhioT1DMParser:
    """
    Parse a single physiological feature from an OhioT1DM XML file.

    Parameters
    ----------
    xml_path : str | Path
        Path to the patient XML file.
    feature : str
        Feature name matching the XML element tag (e.g. 'glucose_level').
    resample_freq : str
        Pandas offset alias used for resampling (default '5min').
    """

    def __init__(
        self,
        xml_path: str | Path,
        feature: str,
        resample_freq: str = "5min",
    ) -> None:
        self.xml_path = Path(xml_path)
        self.feature = feature
        self.resample_freq = resample_freq
        self._df: Optional[pd.DataFrame] = None

    # ── Public API ───────────────────────────────────────────────────────

    def make_dataframe(self) -> pd.DataFrame:
        """
        Parse the XML and return a DatetimeIndex DataFrame with a 'value' column.
        Result is cached after first call.
        """
        if self._df is not None:
            return self._df
        self._df = self._parse()
        return self._df

    def missing_stats(self) -> dict:
        """
        Return a dict with keys: actual, expected, missing, missing_pct.
        'expected' is derived from resampling the date-range at resample_freq.
        """
        df = self.make_dataframe()
        if df.empty:
            return dict(actual=0, expected=0, missing=0, missing_pct=0.0)
        if self.feature == "exercise":
            return dict(actual=len(df), expected=len(df), missing=0, missing_pct=0.0)
        df_r = df.resample(self.resample_freq).mean()
        miss = int(df_r["value"].isna().sum())
        exp  = len(df_r)
        pct  = miss / exp * 100 if exp > 0 else 0.0
        return dict(
            actual=len(df),
            expected=exp,
            missing=miss,
            missing_pct=round(pct, 2),
        )

    # ── Private helpers ──────────────────────────────────────────────────

    def _parse(self) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([]))

        if not self.xml_path.exists():
            logger.warning("File not found: %s", self.xml_path)
            return empty

        try:
            root = ET.parse(self.xml_path).getroot()
        except ET.ParseError as exc:
            logger.error("XML parse error for %s: %s", self.xml_path, exc)
            return empty

        container = root.find(self.feature)
        if container is None:
            logger.debug("Feature '%s' not found in %s", self.feature, self.xml_path.name)
            return empty

        records = []
        for el in container:
            if self.feature == "bolus":
                ts  = el.get("ts_begin") or el.get("ts")
                val = el.get("dose")
            elif self.feature == "temp_basal":
                ts  = el.get("ts_begin") or el.get("ts")
                val = el.get("value")
            elif self.feature == "exercise":
                ts  = el.get("ts")
                val = el.get("duration")
            else:
                ts  = el.get("ts")
                val = el.get("value")
            records.append({"ts": ts, "value": val})

        if not records:
            return empty

        df = pd.DataFrame(records)
        df = df.dropna(subset=["ts", "value"])
        df["ts"]    = df["ts"].apply(_parse_ts)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df
