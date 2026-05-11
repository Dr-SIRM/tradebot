"""Tests for run_asset_sweep — AssetSpec parsing and the verdict logic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_asset_sweep import AssetSpec, _verdict


def test_asset_spec_from_cli_parses():
    spec = AssetSpec.from_cli("data/raw/btc.csv:BTC/USDT:crypto:1h:4h")
    assert spec.data == "data/raw/btc.csv"
    assert spec.symbol == "BTC/USDT"
    assert spec.asset_class == "crypto"
    assert spec.tf_low == "1h"
    assert spec.tf_high == "4h"


def test_asset_spec_from_cli_rejects_bad_format():
    with pytest.raises(ValueError, match="must be"):
        AssetSpec.from_cli("only:three:parts")


def test_verdict_promising():
    row = {"dsr": 0.97, "pbo": 0.05, "wf_sharpe_mean": 0.8}
    assert _verdict(row) == "PROMISING"


def test_verdict_investigate():
    row = {"dsr": 0.85, "pbo": 0.20, "wf_sharpe_mean": 0.3}
    assert _verdict(row) == "INVESTIGATE"


def test_verdict_no_edge_on_low_dsr():
    row = {"dsr": 0.30, "pbo": 0.10, "wf_sharpe_mean": 0.5}
    assert _verdict(row) == "NO_EDGE"


def test_verdict_no_edge_on_negative_wf():
    row = {"dsr": 0.90, "pbo": 0.20, "wf_sharpe_mean": -0.5}
    assert _verdict(row) == "NO_EDGE"


def test_verdict_no_edge_on_high_pbo():
    row = {"dsr": 0.90, "pbo": 0.60, "wf_sharpe_mean": 0.5}
    assert _verdict(row) == "NO_EDGE"


def test_verdict_error_when_failed():
    assert _verdict({"error": "boom"}) == "ERROR"


def test_verdict_no_data_when_dsr_missing():
    assert _verdict({"symbol": "FOO"}) == "NO_DATA"


def test_verdict_promising_works_without_pbo_or_wf():
    # Sensitivity disabled → no PBO, no walkforward; high DSR alone still PROMISING
    row = {"dsr": 0.98}
    assert _verdict(row) == "PROMISING"
