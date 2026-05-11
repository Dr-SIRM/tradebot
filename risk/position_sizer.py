"""Position sizing.

Given equity, risk %, ATR, and a Signal, compute the quantity such that the
loss at the proposed stop equals (risk_pct × equity × conviction_scale).

The conviction scale linearly maps the signal's conviction (0..1) to
[conviction_scale_min, conviction_scale_max], so a 0.5 conviction trade
with [0.5, 1.0] bounds gets 75% of full size.
"""
from __future__ import annotations
from utils.types import Signal


class PositionSizer:
    def __init__(self, risk_cfg: dict):
        self.cfg = risk_cfg

    def size(self, signal: Signal, equity: float) -> tuple[float, float, float]:
        """Return (quantity, dollar_risk, conviction_multiplier).

        quantity is in base units (shares for stocks, coins for crypto, units
        for forex — caller is responsible for any contract-size conversion).

        dollar_risk is the actual $ at risk after sizing (== risk_pct × equity ×
        conviction_mult, bounded by stop distance).
        """
        risk_pct = self.cfg["max_risk_per_trade_pct"]
        cmin = self.cfg["conviction_scale_min"]
        cmax = self.cfg["conviction_scale_max"]
        conviction_mult = cmin + (cmax - cmin) * signal.conviction

        target_risk = risk_pct * equity * conviction_mult
        stop_distance = abs(signal.entry - signal.stop)
        if stop_distance <= 0:
            return 0.0, 0.0, conviction_mult

        quantity = target_risk / stop_distance

        # Sanity: never exceed 100% of equity in notional (prevents leverage blowups
        # on very tight stops).
        max_notional = equity * 1.0
        if quantity * signal.entry > max_notional:
            quantity = max_notional / signal.entry

        actual_risk = quantity * stop_distance
        return quantity, actual_risk, conviction_mult
