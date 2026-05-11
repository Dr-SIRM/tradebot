"""Alpaca live broker (equities/ETFs).

Wraps alpaca-py's TradingClient. The SDK is sync; we offload to the executor.

⚠️ TEST IN PAPER MODE FIRST. The Alpaca API has nuances around order types,
extended hours, fractional shares, and PDT rules that this thin wrapper does
NOT handle. Use it as a starting point, not a finished product.
"""
from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone

from execution.broker_base import BrokerBase
from utils.types import Order, OrderStatus, OrderType, Side
from utils.logging import get_logger

log = get_logger(__name__)


class AlpacaBroker(BrokerBase):
    name = "alpaca"

    def __init__(self, cfg: dict):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        bcfg = cfg["brokers"]["alpaca"]
        self.client = TradingClient(bcfg["api_key"], bcfg["api_secret"],
                                     paper=bcfg.get("paper", True))
        self.data_client = StockHistoricalDataClient(bcfg["api_key"], bcfg["api_secret"])

    async def connect(self) -> None:
        loop = asyncio.get_event_loop()
        acc = await loop.run_in_executor(None, self.client.get_account)
        log.info("alpaca connected: equity=$%s status=%s", acc.equity, acc.status)

    async def disconnect(self) -> None:
        return

    async def get_equity(self) -> float:
        loop = asyncio.get_event_loop()
        acc = await loop.run_in_executor(None, self.client.get_account)
        return float(acc.equity)

    async def get_position(self, symbol: str) -> dict | None:
        from alpaca.common.exceptions import APIError
        loop = asyncio.get_event_loop()
        try:
            pos = await loop.run_in_executor(None, self.client.get_open_position, symbol)
        except APIError:
            return None
        return {
            "symbol": pos.symbol, "qty": float(pos.qty),
            "avg_entry": float(pos.avg_entry_price),
            "side": "long" if float(pos.qty) > 0 else "short",
        }

    async def get_quote(self, symbol: str) -> tuple[float, float]:
        from alpaca.data.requests import StockLatestQuoteRequest
        loop = asyncio.get_event_loop()
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        q = await loop.run_in_executor(None, self.data_client.get_stock_latest_quote, req)
        quote = q[symbol]
        return float(quote.bid_price), float(quote.ask_price)

    async def submit_order(self, symbol: str, side: Side, qty: float,
                            order_type: OrderType, limit_price: float | None = None,
                            stop_price: float | None = None,
                            client_order_id: str = "") -> Order:
        from alpaca.trading.requests import (
            MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce

        ap_side = OrderSide.BUY if side == Side.LONG else OrderSide.SELL
        coid = client_order_id or f"tb-{uuid.uuid4().hex[:16]}"

        if order_type == OrderType.MARKET:
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=ap_side,
                                      time_in_force=TimeInForce.DAY,
                                      client_order_id=coid)
        elif order_type == OrderType.LIMIT:
            req = LimitOrderRequest(symbol=symbol, qty=qty, side=ap_side,
                                     time_in_force=TimeInForce.DAY,
                                     limit_price=limit_price, client_order_id=coid)
        elif order_type == OrderType.STOP:
            req = StopOrderRequest(symbol=symbol, qty=qty, side=ap_side,
                                    time_in_force=TimeInForce.DAY,
                                    stop_price=stop_price, client_order_id=coid)
        else:
            raise ValueError(f"unsupported order type: {order_type}")

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, self.client.submit_order, req)
        return self._to_order(resp)

    async def cancel_order(self, order_id: str) -> bool:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self.client.cancel_order_by_id, order_id)
            return True
        except Exception as e:
            log.warning("alpaca cancel failed %s: %s", order_id, e)
            return False

    async def get_order(self, order_id: str) -> Order:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, self.client.get_order_by_id, order_id)
        return self._to_order(resp)

    @staticmethod
    def _to_order(r) -> Order:
        status_map = {
            "new": OrderStatus.OPEN, "accepted": OrderStatus.OPEN,
            "pending_new": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED, "partially_filled": OrderStatus.PARTIAL,
            "canceled": OrderStatus.CANCELLED, "expired": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        side = Side.LONG if r.side == "buy" else Side.SHORT
        return Order(
            id=str(r.id), symbol=r.symbol, side=side, quantity=float(r.qty or 0),
            order_type=OrderType(r.order_type) if r.order_type in [e.value for e in OrderType] else OrderType.MARKET,
            limit_price=float(r.limit_price) if r.limit_price else None,
            stop_price=float(r.stop_price) if r.stop_price else None,
            status=status_map.get(r.status, OrderStatus.OPEN),
            filled_qty=float(r.filled_qty or 0),
            avg_fill_price=float(r.filled_avg_price or 0),
            submit_time=r.submitted_at if isinstance(r.submitted_at, datetime) else None,
            fill_time=r.filled_at if isinstance(r.filled_at, datetime) else None,
            broker="alpaca",
        )
