"""Tests for the risk gates: R:R, max concurrent, drawdown halts, sizing."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk.manager import RiskManager
from utils.types import Regime, Side, Signal


def _cfg(**overrides):
    base = {
        "risk": {
            "max_risk_per_trade_pct": 0.01,
            "min_reward_risk_ratio": 2.5,
            "atr_stop_mult_min": 1.5,
            "atr_stop_mult_max": 2.0,
            "daily_drawdown_halt_pct": 0.04,
            "weekly_drawdown_halt_pct": 0.08,
            "max_concurrent_positions": 3,
            "trailing_stop_atr_mult": 1.0,
            "trailing_activate_R": 1.5,
            "conviction_scale_min": 0.5,
            "conviction_scale_max": 1.0,
            "news_blackout_minutes": 15,
        },
        "correlation": {
            "clusters": [["BTC/USDT", "ETH/USDT"]],
            "asset_class_caps": {"crypto": 2, "equity": 2, "forex": 2, "commodity": 2},
        },
    }
    base["risk"].update(overrides.get("risk", {}))
    return base


def _signal(symbol="BTC/USDT", asset="crypto", side=Side.LONG,
            entry=100.0, stop=99.0, tp=104.0, conv=0.7):
    return Signal(
        symbol=symbol, asset_class=asset, side=side, strategy="momentum",
        entry=entry, stop=stop, take_profit=tp, conviction=conv,
        timestamp=datetime.now(timezone.utc), timeframe="5m",
        regime=Regime.TRENDING,
    )


class TestRewardRisk:
    def test_low_rr_rejected(self):
        rm = RiskManager(_cfg(), 100_000)
        # entry=100, stop=99 → risk=1; tp=101.5 → reward=1.5; RR=1.5 < 2.5
        sig = _signal(tp=101.5)
        ok, reason, _ = rm.approve_trade(sig, datetime.now(timezone.utc))
        assert not ok
        assert "reward:risk" in reason

    def test_good_rr_accepted(self):
        rm = RiskManager(_cfg(), 100_000)
        sig = _signal(tp=104.0)  # RR = 4
        ok, reason, plan = rm.approve_trade(sig, datetime.now(timezone.utc))
        assert ok, f"unexpected reject: {reason}"
        assert plan["quantity"] > 0


class TestStopSanity:
    def test_long_stop_above_entry_rejected(self):
        rm = RiskManager(_cfg(), 100_000)
        sig = _signal(side=Side.LONG, entry=100, stop=101, tp=110)
        ok, reason, _ = rm.approve_trade(sig, datetime.now(timezone.utc))
        assert not ok
        assert "stop" in reason.lower()

    def test_short_stop_below_entry_rejected(self):
        rm = RiskManager(_cfg(), 100_000)
        sig = _signal(side=Side.SHORT, entry=100, stop=99, tp=90)
        ok, reason, _ = rm.approve_trade(sig, datetime.now(timezone.utc))
        assert not ok
        assert "stop" in reason.lower()


class TestMaxConcurrent:
    def test_blocks_at_limit(self):
        rm = RiskManager(_cfg(risk={"max_concurrent_positions": 2}), 100_000)
        now = datetime.now(timezone.utc)
        # Open 2 positions in different non-correlated clusters
        s1 = _signal(symbol="BTC/USDT", asset="crypto")
        ok, _, plan = rm.approve_trade(s1, now)
        assert ok
        rm.open_position(s1, plan["quantity"], 100.0, plan["risk_amount"], "paper", now)

        s2 = _signal(symbol="SPY", asset="equity", entry=400, stop=395, tp=420)
        ok, _, plan = rm.approve_trade(s2, now)
        assert ok
        rm.open_position(s2, plan["quantity"], 400.0, plan["risk_amount"], "paper", now)

        # Third should be rejected
        s3 = _signal(symbol="EUR_USD", asset="forex", entry=1.10, stop=1.09, tp=1.13)
        ok, reason, _ = rm.approve_trade(s3, now)
        assert not ok
        assert "max concurrent" in reason


class TestPositionSizing:
    def test_dollar_risk_within_cap(self):
        equity = 100_000
        rm = RiskManager(_cfg(), equity)
        sig = _signal(entry=100, stop=99, tp=104, conv=1.0)
        ok, _, plan = rm.approve_trade(sig, datetime.now(timezone.utc))
        assert ok
        # 1% of 100k = $1000 risk cap; conviction_mult=1.0 so dollar_risk ≈ $1000
        assert plan["risk_amount"] <= 1000.01

    def test_lower_conviction_smaller_size(self):
        rm_hi = RiskManager(_cfg(), 100_000)
        rm_lo = RiskManager(_cfg(), 100_000)
        sig_hi = _signal(conv=1.0)
        sig_lo = _signal(conv=0.5)
        _, _, p_hi = rm_hi.approve_trade(sig_hi, datetime.now(timezone.utc))
        _, _, p_lo = rm_lo.approve_trade(sig_lo, datetime.now(timezone.utc))
        assert p_lo["quantity"] < p_hi["quantity"]


class TestCorrelation:
    def test_cluster_blocks_second_entry(self):
        rm = RiskManager(_cfg(), 100_000)
        now = datetime.now(timezone.utc)
        s1 = _signal(symbol="BTC/USDT", asset="crypto")
        ok, _, plan = rm.approve_trade(s1, now)
        assert ok
        rm.open_position(s1, plan["quantity"], 100.0, plan["risk_amount"], "paper", now)

        # ETH is in same cluster
        s2 = _signal(symbol="ETH/USDT", asset="crypto", entry=2000, stop=1980, tp=2080)
        ok, reason, _ = rm.approve_trade(s2, now)
        assert not ok


class TestDrawdownHalt:
    def test_daily_halt_triggers(self):
        rm = RiskManager(_cfg(), 100_000)
        now = datetime.now(timezone.utc)
        # Establish the day anchor
        rm.update_equity_mark({}, now)
        assert not rm.daily_halted

        # Inject a closed trade representing a >4% loss
        from utils.types import Side, Trade
        rm.closed_trades.append(Trade(
            symbol="BTC/USDT", asset_class="crypto", side=Side.LONG,
            strategy="momentum", entry_time=now, exit_time=now,
            entry_price=100, exit_price=95, quantity=1000, pnl=-5_000,
            pnl_pct=-0.05, r_multiple=-5.0, exit_reason="stop",
        ))
        # Re-mark; equity should drop and halt should trigger
        rm.update_equity_mark({}, now)
        assert rm.daily_halted, f"expected halt, equity={rm.equity}"

        sig = _signal()
        ok, reason, _ = rm.approve_trade(sig, now)
        assert not ok
        assert "halt" in reason.lower()


class TestExitCheck:
    def test_long_stop_hit(self):
        rm = RiskManager(_cfg(), 100_000)
        now = datetime.now(timezone.utc)
        sig = _signal(entry=100, stop=99, tp=104, conv=1.0)
        ok, _, plan = rm.approve_trade(sig, now)
        assert ok
        rm.open_position(sig, plan["quantity"], 100.0, plan["risk_amount"], "paper", now)
        # Bar where low pierces the stop
        should_exit, reason, exit_px = rm.check_exits("BTC/USDT", 101, 98, 99.5)
        assert should_exit
        assert reason == "stop"
        assert exit_px == pytest.approx(99.0)

    def test_long_tp_hit(self):
        rm = RiskManager(_cfg(), 100_000)
        now = datetime.now(timezone.utc)
        sig = _signal(entry=100, stop=99, tp=104, conv=1.0)
        _, _, plan = rm.approve_trade(sig, now)
        rm.open_position(sig, plan["quantity"], 100.0, plan["risk_amount"], "paper", now)
        should_exit, reason, exit_px = rm.check_exits("BTC/USDT", 105, 100, 104.5)
        assert should_exit
        assert reason == "take_profit"
