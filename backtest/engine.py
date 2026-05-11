"""Backtest engine — bar-by-bar replay through the same strategy/risk pipeline
as live trading.

Designed to be fast (vectorized indicator pre-computation) but step-wise on the
event loop (so the same risk manager, sizer, regime detector run in test as in
prod). This is the only sane way to avoid look-ahead bugs.

Fill model:
  - Entry: at next bar's open + slippage_bps (configurable). This is realistic
    for market orders; for limit-only strategies use the LimitFillBacktestEngine
    variant if you need it.
  - Stop/TP exits: checked on each bar's high/low. If both hit on the same bar,
    we conservatively assume the stop hit first (worst case).
"""
from __future__ import annotations
from datetime import timezone
import numpy as np
import pandas as pd

from data.indicators import add_all_indicators
from strategy.selector import StrategySelector
from risk.manager import RiskManager
from utils.types import Trade, Side, Signal
from utils.logging import get_logger
from utils.time import utc_now

log = get_logger(__name__)


class Backtester:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.starting_equity = cfg["backtest"]["initial_equity"]
        self.fee_bps = cfg["brokers"]["paper"]["fee_bps"]
        self.slippage_bps = cfg["execution"]["slippage_bps_assumed"]

    def run(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame, symbol: str,
             asset_class: str, timeframe_low: str,
             start_date: str | None = None, end_date: str | None = None) -> tuple[list[Trade], pd.Series]:
        """Run a single-symbol backtest. Returns (trades, equity_curve)."""
        if start_date:
            ltf_df = ltf_df[ltf_df.index >= pd.Timestamp(start_date, tz="UTC")]
            htf_df = htf_df[htf_df.index >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            ltf_df = ltf_df[ltf_df.index <= pd.Timestamp(end_date, tz="UTC")]
            htf_df = htf_df[htf_df.index <= pd.Timestamp(end_date, tz="UTC")]

        ltf_ind = add_all_indicators(ltf_df, self.cfg)
        htf_ind = add_all_indicators(htf_df, self.cfg)

        selector = StrategySelector(self.cfg)
        risk = RiskManager(self.cfg, self.starting_equity)

        warmup = self.cfg["data"]["warmup_bars"]
        if len(ltf_ind) < warmup + 10:
            log.warning("not enough bars to backtest: %d", len(ltf_ind))
            return [], pd.Series([self.starting_equity])

        # Pending entry: filled at next bar's open
        pending_entry: tuple[Signal, dict] | None = None

        eq_records: list[tuple[pd.Timestamp, float]] = []

        for i in range(warmup, len(ltf_ind) - 1):
            bar = ltf_ind.iloc[i]
            next_bar = ltf_ind.iloc[i + 1]
            ts = ltf_ind.index[i].to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            # 1. Process pending entry → fill at next bar open with slippage
            if pending_entry is not None:
                signal, plan = pending_entry
                pending_entry = None
                slip = next_bar["open"] * (self.slippage_bps / 10000.0)
                fill_price = next_bar["open"] + slip if signal.side == Side.LONG else next_bar["open"] - slip
                # Recheck: signal stop must still be valid relative to fill
                if (signal.side == Side.LONG and signal.stop < fill_price) or \
                   (signal.side == Side.SHORT and signal.stop > fill_price):
                    risk.open_position(signal, plan["quantity"], fill_price,
                                        plan["risk_amount"], "backtest", ts)

            # 2. Check exits on this bar (stop, TP, trailing update)
            for sym in list(risk.positions.keys()):
                # Update trailing stop based on current bar close
                risk.update_trailing(sym, float(bar["close"]))
                should_exit, reason, exit_price = risk.check_exits(
                    sym, float(bar["high"]), float(bar["low"]), float(bar["close"])
                )
                if should_exit:
                    notional = exit_price * risk.positions[sym].quantity
                    fee = notional * (self.fee_bps / 10000.0)
                    risk.close_position(sym, exit_price, ts, reason, fees=fee)

            # 3. Mark equity to current close
            risk.update_equity_mark({sym: float(bar["close"]) for sym in risk.positions}, ts)
            eq_records.append((ltf_ind.index[i], risk.equity))

            # 4. Generate new signal (only if no pending entry and not halted)
            if pending_entry is None and not risk.daily_halted and not risk.weekly_halted:
                # Align HTF: use HTF bars up through this LTF timestamp
                htf_slice = htf_ind.loc[: ltf_ind.index[i]]
                if len(htf_slice) >= 60:
                    ltf_slice = ltf_ind.iloc[: i + 1]
                    regime, signal = selector.select(htf_slice, ltf_slice, symbol,
                                                      asset_class, timeframe_low)
                    if signal is not None:
                        ok, reason, plan = risk.approve_trade(signal, ts)
                        if ok:
                            pending_entry = (signal, plan)
                        else:
                            log.debug("rejected signal: %s @ %s", reason, ts)

        # Close any remaining positions at last close
        if risk.positions:
            last_bar = ltf_ind.iloc[-1]
            last_ts = ltf_ind.index[-1].to_pydatetime()
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            for sym in list(risk.positions.keys()):
                price = float(last_bar["close"])
                fee = price * risk.positions[sym].quantity * (self.fee_bps / 10000.0)
                risk.close_position(sym, price, last_ts, "end_of_data", fees=fee)

        eq_df = pd.Series([e[1] for e in eq_records],
                           index=pd.DatetimeIndex([e[0] for e in eq_records]),
                           name="equity")
        return risk.closed_trades, eq_df
