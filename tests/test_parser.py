"""
tests/test_parser.py
--------------------
Unit tests for the OhioT1DM XML parser.
"""
from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from src.data.parser import OhioT1DMParser, _parse_ts


# ─── _parse_ts ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ts_str,expected_year", [
    ("01-01-2018 08:00:00", 2018),
    ("2018-01-01 08:00:00", 2018),
    ("01/01/2018 08:00:00", 2018),
    ("01-01-2018 08:00",    2018),
])
def test_parse_ts_formats(ts_str, expected_year):
    result = _parse_ts(ts_str)
    assert result is not None
    assert result.year == expected_year


def test_parse_ts_none():
    assert _parse_ts("") is None
    assert _parse_ts(None) is None


# ─── OhioT1DMParser ──────────────────────────────────────────────────────────

def _make_xml(feature: str, records: list[dict]) -> str:
    """Helper: build a minimal OhioT1DM-style XML string."""
    root = ET.Element("patient")
    container = ET.SubElement(root, feature)
    for rec in records:
        el = ET.SubElement(container, "event")
        for k, v in rec.items():
            el.set(k, str(v))
    return ET.tostring(root, encoding="unicode")


def _write_temp_xml(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def test_glucose_parser_basic():
    xml_content = _make_xml("glucose_level", [
        {"ts": "01-01-2018 08:00:00", "value": "120"},
        {"ts": "01-01-2018 08:05:00", "value": "125"},
        {"ts": "01-01-2018 08:10:00", "value": "130"},
    ])
    path = _write_temp_xml(xml_content)
    try:
        parser = OhioT1DMParser(path, "glucose_level")
        df = parser.make_dataframe()
        assert len(df) == 3
        assert "value" in df.columns
        assert df["value"].iloc[0] == pytest.approx(120.0)
    finally:
        os.remove(path)


def test_missing_feature_returns_empty():
    xml_content = _make_xml("glucose_level", [
        {"ts": "01-01-2018 08:00:00", "value": "100"},
    ])
    path = _write_temp_xml(xml_content)
    try:
        parser = OhioT1DMParser(path, "bolus")  # bolus not present
        df = parser.make_dataframe()
        assert df.empty
    finally:
        os.remove(path)


def test_missing_stats_unavailable_feature_returns_na():
    xml_content = _make_xml("glucose_level", [
        {"ts": "01-01-2018 08:00:00", "value": "100"},
    ])
    path = _write_temp_xml(xml_content)
    try:
        stats = OhioT1DMParser(path, "basis_gsr").missing_stats()
        assert stats["actual"] == "NA"
        assert stats["missing_pct"] == "NA"
    finally:
        os.remove(path)


def test_missing_stats():
    # 3 readings at 5-min intervals → 0 missing
    xml_content = _make_xml("glucose_level", [
        {"ts": "01-01-2018 08:00:00", "value": "110"},
        {"ts": "01-01-2018 08:05:00", "value": "115"},
        {"ts": "01-01-2018 08:10:00", "value": "120"},
    ])
    path = _write_temp_xml(xml_content)
    try:
        stats = OhioT1DMParser(path, "glucose_level").missing_stats()
        assert stats["actual"] == 3
        assert stats["missing"] == 0
    finally:
        os.remove(path)


def test_caching():
    """make_dataframe() should return the same object on repeated calls."""
    xml_content = _make_xml("glucose_level", [
        {"ts": "01-01-2018 08:00:00", "value": "100"},
    ])
    path = _write_temp_xml(xml_content)
    try:
        parser = OhioT1DMParser(path, "glucose_level")
        df1 = parser.make_dataframe()
        df2 = parser.make_dataframe()
        assert df1 is df2
    finally:
        os.remove(path)
