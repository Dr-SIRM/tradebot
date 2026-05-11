"""Tests for data.download — checks the pure-Python helpers without network I/O.

Network-backed paths (download_crypto, download_equity) are smoke-tested through
the CLI entry point in the end-to-end backtest, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.download import _ensure_utc, _normalize_frame, _resume_window


def test_ensure_utc_naive_string():
    out = _ensure_utc("2024-01-01")
    assert out.tz is not None
    assert out.tz.zone == "UTC" if hasattr(out.tz, "zone") else str(out.tz) == "UTC"
    assert out.year == 2024 and out.month == 1 and out.day == 1


def test_ensure_utc_already_tz_aware():
    # Use fixed-offset string to avoid depending on system tzdata
    ts = pd.Timestamp("2024-01-01 00:00:00-05:00")
    out = _ensure_utc(ts)
    assert str(out.tz) == "UTC"
    assert out.hour == 5  # midnight at UTC-5 → 5am UTC


def test_normalize_frame_strips_extras_and_sorts():
    idx = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-01", "2024-01-01"], name="timestamp"
    )
    df = pd.DataFrame({
        "open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
        "close": [1, 2, 3], "volume": [10, 20, 30], "junk": ["a", "b", "c"],
    }, index=idx)
    out = _normalize_frame(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.is_monotonic_increasing
    # Duplicate timestamps kept last; 3 rows → 2 unique
    assert len(out) == 2
    assert str(out.index.tz) == "UTC"


def test_normalize_frame_rejects_missing_columns():
    idx = pd.DatetimeIndex(["2024-01-01"], name="timestamp", tz="UTC")
    df = pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]}, index=idx)
    with pytest.raises(ValueError, match="missing column"):
        _normalize_frame(df)


def test_resume_window_no_file(tmp_path):
    out = tmp_path / "nope.csv"
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-02-01", tz="UTC")
    new_start, existing = _resume_window(out, start, end)
    assert new_start == start
    assert existing.empty


def test_resume_window_continues_from_last(tmp_path):
    out = tmp_path / "data.csv"
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    pd.DataFrame({
        "open": [1] * 5, "high": [1] * 5, "low": [1] * 5,
        "close": [1] * 5, "volume": [1] * 5,
    }, index=idx).reset_index(names="timestamp").to_csv(out, index=False)

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-02-01", tz="UTC")
    new_start, existing = _resume_window(out, start, end)
    # Should resume just after the last bar
    assert new_start > idx[-1]
    assert len(existing) == 5
