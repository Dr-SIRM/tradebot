"""In-memory paper broker. Simulates fills with configurable slippage and fees.

For market orders: fills at next bar's open + slippage.
For limit orders: fills if next bar's range crosses the limit price.

The orchestrator drives this by calling on_bar() with new bar data per symbol.
"""
from __future__ import annotations
import asyncio
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Optional

from execution.broker_base import BrokerBase
from utils.types import Order, OrderStatus, OrderType, Side
from utils.logging import get_logger
from utils.time import utc_now

log = get_logger(__name__)


class PaperBroker(BrokerBase):
    name = "paper"

    def __init__(self, cfg: dict, starting_equity: float):
        self.fee_bps = cfg.get("brokers", {}).get("paper", {}).get("fee_bps", 5)
        self.slippage_bps = cfg.get("brokers", {}).get("paper", {}).get("slippage_bps", 3)
        self._equity = starting_equity
        self._cash = starting_equity
        self._positions: dict[str, dict] = {}        # symbol -> {qty, avg_entry, side}
        self._orders: dict[str, Order] = {}          # order_id -> Order
        self._open_orders: dict[str, Order] = {}     # only PENDING/OPEN
        self._last_quote: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        log.info("paper broker ready (equity=$%.2f)", self._equity)

    async def disconnect(self) -> None:
        pass

    async def get_equity(self) -> float:
        return self._equity

    async def get_position(self, symbol: str) -> dict | None:
        return self._positions.get(symbol)

    async def get_quote(self, symbol: str) -> tuple[float, float]:
        return self._last_quote.get(symbol, (0.0, 0.0))

    async def submit_order(self, symbol: str, side: Side, qty: float,
                            order_type: OrderType, limit_price: Optional[float] = None,
                            stop_price: Optional[float] = None,
                            client_order_id: str = "") -> Order:
        async with self._lock:
            oid = client_order_id or str(uuid.uuid4())
            o = Order(
                id=oid, symbol=symbol, side=side, quantity=qty,
                order_type=order_type, limit_price=limit_price, stop_price=stop_price,
                status=OrderStatus.OPEN if order_type == OrderType.LIMIT else OrderStatus.PENDING,
                submit_time=utc_now(), broker=self.name,
            )
            self._orders[oid] = o
            self._open_orders[oid] = o
            log.debug("paper submit: %s %s %s qty=%.6f type=%s lim=%s",
                      symbol, side.value, order_type.value, qty, order_type.value, limit_price)
            return o

    async def cancel_order(self, order_id: str) -> bool:
        async with self._lock:
            o = self._open_orders.pop(order_id, None)
            if o is None:
                return False
            o.status = OrderStatus.CANCELLED
            return True

    async def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    # ---- driven by orchestrator on each bar -----------------------------

    def update_quote(self, symbol: str, bid: float, ask: float) -> None:
        self._last_quote[symbol] = (bid, ask)

    async def on_bar(self, symbol: str, bar_open: float, bar_high: float,
                     bar_low: float, bar_close: float, ts: datetime) -> list[Order]:
        """Try to fill open orders against this bar. Returns the orders that filled."""
        filled: list[Order] = []
        async with self._lock:
            for oid, o in list(self._open_orders.items()):
                if o.symbol != symbol:
                    continue
                fill_price = self._try_fill(o, bar_open, bar_high, bar_low, bar_close)
                if fill_price is None:
                    continue
                # Apply slippage on the side that hurts us
                slip = fill_price * (self.slippage_bps / 10000.0)
                if o.side == Side.LONG:
                    fill_price += slip
                else:
                    fill_price -= slip

                o.avg_fill_price = fill_price
                o.filled_qty = o.quantity
                o.fill_time = ts
                o.status = OrderStatus.FILLED
                self._open_orders.pop(oid, None)
                self._apply_fill(o, fill_price)
                filled.append(o)
                log.debug("paper fill: %s %s @ %.4f", o.symbol, o.side.value, fill_price)
        return filled

    def _try_fill(self, o: Order, bo: float, bh: float, bl: float, bc: float) -> Optional[float]:
        if o.order_type == OrderType.MARKET:
            return bo  # next-bar-open assumption
        if o.order_type == OrderType.LIMIT:
            if o.side == Side.LONG and o.limit_price is not None and bl <= o.limit_price:
                return min(o.limit_price, bo)  # filled at limit or better
            if o.side == Side.SHORT and o.limit_price is not None and bh >= o.limit_price:
                return max(o.limit_price, bo)
        if o.order_type == OrderType.STOP:
            if o.side == Side.LONG and o.stop_price is not None and bh >= o.stop_price:
                return max(o.stop_price, bo)
            if o.side == Side.SHORT and o.stop_price is not None and bl <= o.stop_price:
                return min(o.stop_price, bo)
        return None

    def _apply_fill(self, o: Order, price: float) -> None:
        notional = price * o.quantity
        fee = notional * (self.fee_bps / 10000.0)
        existing = self._positions.get(o.symbol)

        if existing is None:
            # Opening
            self._positions[o.symbol] = {
                "symbol": o.symbol, "qty": o.quantity, "avg_entry": price,
                "side": o.side.value,
            }
            self._cash -= notional if o.side == Side.LONG else -notional  # short adds cash
            self._cash -= fee
        else:
            # If same side -> increase. If opposite -> reduce/close.
            same_side = existing["side"] == o.side.value
            if same_side:
                total_qty = existing["qty"] + o.quantity
                existing["avg_entry"] = ((existing["avg_entry"] * existing["qty"]) +
                                          (price * o.quantity)) / total_qty
                existing["qty"] = total_qty
                self._cash -= notional if o.side == Side.LONG else -notional
                self._cash -= fee
            else:
                # Closing or reducing
                close_qty = min(existing["qty"], o.quantity)
                if existing["side"] == "long":
                    pnl = (price - existing["avg_entry"]) * close_qty
                else:
                    pnl = (existing["avg_entry"] - price) * close_qty
                self._cash += pnl - fee
                existing["qty"] -= close_qty
                if existing["qty"] <= 1e-12:
                    self._positions.pop(o.symbol, None)

        # Mark equity to last seen quote
        self._equity = self._cash + sum(
            (q[0] if (s := p.get("side")) == "long" else -q[0]) * p["qty"]
            if (q := self._last_quote.get(sym)) else 0.0
            for sym, p in self._positions.items()
        )
