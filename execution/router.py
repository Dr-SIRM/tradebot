"""Smart order router.

Behavior:
  - Default: place a LIMIT at mid ± offset_bps. If not filled within
    limit_timeout_seconds, cancel and (for momentum/breakout strategies) submit
    a MARKET to ensure entry.
  - Mean-reversion strategies stay limit-only — if the limit doesn't fill, the
    setup is gone.
  - Logs every order with submit_time, fill_time, and slippage in bps for
    post-trade analysis.

Maintains a registry of broker instances keyed by name so the orchestrator can
look one up per symbol.
"""
from __future__ import annotations
import asyncio
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from execution.broker_base import BrokerBase
from utils.types import Order, OrderStatus, OrderType, Side, Signal
from utils.logging import get_logger
from utils.time import utc_now

log = get_logger(__name__)


class OrderRouter:
    def __init__(self, cfg: dict, brokers: dict[str, BrokerBase]):
        self.cfg = cfg["execution"]
        self.brokers = brokers
        self.market_fallback_strategies = set(self.cfg["market_fallback_strategies"])
        self.trade_log_path = Path(cfg["logging"]["trade_log_csv"])
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_trade_log()

    def _init_trade_log(self):
        if not self.trade_log_path.exists():
            with self.trade_log_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "submit_ts", "fill_ts", "symbol", "side", "strategy",
                    "qty", "intended_price", "fill_price", "slippage_bps",
                    "order_type", "status", "broker", "latency_ms",
                ])

    def _log_fill(self, o: Order, intended_price: float, strategy: str):
        slip_bps = 0.0
        if intended_price and o.avg_fill_price:
            diff = (o.avg_fill_price - intended_price) if o.side == Side.LONG \
                   else (intended_price - o.avg_fill_price)
            slip_bps = (diff / intended_price) * 10000.0
        latency_ms = ""
        if o.submit_time and o.fill_time:
            latency_ms = f"{(o.fill_time - o.submit_time).total_seconds() * 1000:.0f}"
        with self.trade_log_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                o.submit_time.isoformat() if o.submit_time else "",
                o.fill_time.isoformat() if o.fill_time else "",
                o.symbol, o.side.value, strategy, f"{o.quantity:.8f}",
                f"{intended_price:.8f}" if intended_price else "",
                f"{o.avg_fill_price:.8f}" if o.avg_fill_price else "",
                f"{slip_bps:.2f}", o.order_type.value, o.status.value,
                o.broker, latency_ms,
            ])

    async def execute_entry(self, signal: Signal, qty: float,
                             broker_name: str) -> Optional[Order]:
        """Submit an entry order per the smart-routing rules."""
        broker = self.brokers.get(broker_name)
        if broker is None:
            log.error("router: unknown broker %s", broker_name)
            return None

        if not self.cfg["prefer_limit"]:
            return await self._submit_market(broker, signal, qty)

        # Try LIMIT first
        bid, ask = await broker.get_quote(signal.symbol)
        if bid <= 0 or ask <= 0:
            log.warning("router: invalid quote for %s, falling back to market", signal.symbol)
            return await self._submit_market(broker, signal, qty)

        mid = (bid + ask) / 2
        offset = mid * (self.cfg["limit_offset_bps"] / 10000.0)
        limit_price = mid - offset if signal.side == Side.LONG else mid + offset

        order = await broker.submit_order(
            signal.symbol, signal.side, qty, OrderType.LIMIT,
            limit_price=limit_price,
        )
        if order.status == OrderStatus.REJECTED:
            log.warning("router: limit rejected for %s", signal.symbol)
            return order

        # Wait up to timeout for fill
        timeout = self.cfg["limit_timeout_seconds"]
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                cur = await broker.get_order(order.id)
            except Exception:
                cur = order
            if cur.status == OrderStatus.FILLED:
                self._log_fill(cur, limit_price, signal.strategy)
                return cur
            if cur.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                break
            await asyncio.sleep(min(2.0, max(0.5, timeout / 10)))

        # Timeout: cancel + maybe market fallback
        await broker.cancel_order(order.id)
        if signal.strategy in self.market_fallback_strategies:
            log.info("router: limit timeout on %s, falling back to MARKET", signal.symbol)
            return await self._submit_market(broker, signal, qty, intended_price=mid)
        log.info("router: limit timeout on %s and no market fallback for strategy=%s",
                 signal.symbol, signal.strategy)
        return None

    async def _submit_market(self, broker: BrokerBase, signal: Signal, qty: float,
                              intended_price: float | None = None) -> Order:
        if intended_price is None:
            bid, ask = await broker.get_quote(signal.symbol)
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else signal.entry
            intended_price = mid
        o = await broker.submit_order(signal.symbol, signal.side, qty, OrderType.MARKET)
        self._log_fill(o, intended_price, signal.strategy)
        return o

    async def execute_exit(self, symbol: str, side_to_close: Side, qty: float,
                            broker_name: str, strategy: str,
                            reason: str) -> Optional[Order]:
        """Close an existing position. Always uses MARKET to guarantee exit."""
        broker = self.brokers.get(broker_name)
        if broker is None:
            log.error("router: unknown broker %s", broker_name)
            return None
        # To close a long, sell; to close a short, buy
        close_side = Side.SHORT if side_to_close == Side.LONG else Side.LONG
        bid, ask = await broker.get_quote(symbol)
        intended = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0.0
        o = await broker.submit_order(symbol, close_side, qty, OrderType.MARKET)
        self._log_fill(o, intended, f"{strategy}-exit:{reason}")
        return o
