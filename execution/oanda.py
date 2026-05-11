"""OANDA live broker (forex + CFDs including XAU/XAG, oil).

Wraps oandapyV20 (sync, dispatched to executor).

⚠️ TEST IN PRACTICE MODE FIRST. Watch out for:
  - Pip vs price units on JPY pairs.
  - Margin requirements per pair.
  - Trailing stop server-side support (not used here — we trail in-process).
"""
from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone

from execution.broker_base import BrokerBase
from utils.types import Order, OrderStatus, OrderType, Side
from utils.logging import get_logger

log = get_logger(__name__)


class OandaBroker(BrokerBase):
    name = "oanda"

    def __init__(self, cfg: dict):
        from oandapyV20 import API
        bcfg = cfg["brokers"]["oanda"]
        env = "practice" if bcfg.get("practice", True) else "live"
        self.api = API(access_token=bcfg["api_key"], environment=env)
        self.account_id = bcfg["account_id"]

    async def connect(self) -> None:
        from oandapyV20.endpoints.accounts import AccountSummary
        loop = asyncio.get_event_loop()
        r = AccountSummary(accountID=self.account_id)
        resp = await loop.run_in_executor(None, self.api.request, r)
        log.info("oanda connected: balance=%s", resp.get("account", {}).get("balance"))

    async def disconnect(self) -> None:
        return

    async def get_equity(self) -> float:
        from oandapyV20.endpoints.accounts import AccountSummary
        loop = asyncio.get_event_loop()
        r = AccountSummary(accountID=self.account_id)
        resp = await loop.run_in_executor(None, self.api.request, r)
        return float(resp["account"]["NAV"])

    async def get_position(self, symbol: str) -> dict | None:
        from oandapyV20.endpoints.positions import PositionDetails
        loop = asyncio.get_event_loop()
        try:
            r = PositionDetails(accountID=self.account_id, instrument=symbol)
            resp = await loop.run_in_executor(None, self.api.request, r)
            p = resp.get("position", {})
            long_units = float(p.get("long", {}).get("units", 0) or 0)
            short_units = float(p.get("short", {}).get("units", 0) or 0)
            if long_units > 0:
                return {"symbol": symbol, "qty": long_units,
                        "avg_entry": float(p["long"].get("averagePrice", 0)),
                        "side": "long"}
            if short_units < 0:
                return {"symbol": symbol, "qty": abs(short_units),
                        "avg_entry": float(p["short"].get("averagePrice", 0)),
                        "side": "short"}
            return None
        except Exception:
            return None

    async def get_quote(self, symbol: str) -> tuple[float, float]:
        from oandapyV20.endpoints.pricing import PricingInfo
        loop = asyncio.get_event_loop()
        params = {"instruments": symbol}
        r = PricingInfo(accountID=self.account_id, params=params)
        resp = await loop.run_in_executor(None, self.api.request, r)
        prices = resp.get("prices", [])
        if not prices:
            return 0.0, 0.0
        p = prices[0]
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        return bid, ask

    async def submit_order(self, symbol: str, side: Side, qty: float,
                            order_type: OrderType, limit_price: float | None = None,
                            stop_price: float | None = None,
                            client_order_id: str = "") -> Order:
        from oandapyV20.endpoints.orders import OrderCreate
        units = int(qty if side == Side.LONG else -qty)  # OANDA: signed integer units
        coid = client_order_id or f"tb-{uuid.uuid4().hex[:16]}"

        if order_type == OrderType.MARKET:
            order = {"type": "MARKET", "instrument": symbol, "units": str(units),
                     "timeInForce": "FOK", "clientExtensions": {"id": coid}}
        elif order_type == OrderType.LIMIT:
            order = {"type": "LIMIT", "instrument": symbol, "units": str(units),
                     "timeInForce": "GTC", "price": str(limit_price),
                     "clientExtensions": {"id": coid}}
        elif order_type == OrderType.STOP:
            order = {"type": "STOP", "instrument": symbol, "units": str(units),
                     "timeInForce": "GTC", "price": str(stop_price),
                     "clientExtensions": {"id": coid}}
        else:
            raise ValueError(f"unsupported order type: {order_type}")

        loop = asyncio.get_event_loop()
        try:
            r = OrderCreate(accountID=self.account_id, data={"order": order})
            resp = await loop.run_in_executor(None, self.api.request, r)
            return self._to_order(resp, symbol, side, qty, order_type, limit_price, stop_price, coid)
        except Exception as e:
            log.error("oanda order rejected: %s", e)
            return Order(id=coid, symbol=symbol, side=side, quantity=qty,
                         order_type=order_type, status=OrderStatus.REJECTED,
                         broker=self.name, metadata={"error": str(e)})

    async def cancel_order(self, order_id: str) -> bool:
        from oandapyV20.endpoints.orders import OrderCancel
        loop = asyncio.get_event_loop()
        try:
            r = OrderCancel(accountID=self.account_id, orderID=order_id)
            await loop.run_in_executor(None, self.api.request, r)
            return True
        except Exception as e:
            log.warning("oanda cancel failed %s: %s", order_id, e)
            return False

    async def get_order(self, order_id: str) -> Order:
        from oandapyV20.endpoints.orders import OrderDetails
        loop = asyncio.get_event_loop()
        r = OrderDetails(accountID=self.account_id, orderID=order_id)
        resp = await loop.run_in_executor(None, self.api.request, r)
        o = resp.get("order", {})
        return self._to_order(resp, o.get("instrument", ""),
                              Side.LONG if int(o.get("units", 0)) > 0 else Side.SHORT,
                              abs(float(o.get("units", 0))),
                              OrderType(o.get("type", "MARKET").lower()),
                              float(o.get("price")) if o.get("price") else None,
                              None, str(o.get("id", "")))

    @staticmethod
    def _to_order(resp: dict, symbol: str, side: Side, qty: float,
                  order_type: OrderType, lim, stp, coid: str) -> Order:
        # Pull the created-or-filled transaction
        ot = resp.get("orderFillTransaction") or resp.get("orderCreateTransaction") or {}
        oid = ot.get("orderID") or ot.get("id") or coid
        filled_qty = abs(float(ot.get("units", 0))) if "units" in ot else 0.0
        avg = float(ot.get("price", 0)) if ot.get("price") else 0.0
        if "orderFillTransaction" in resp:
            status = OrderStatus.FILLED
        elif resp.get("orderCancelTransaction"):
            status = OrderStatus.CANCELLED
        else:
            status = OrderStatus.OPEN
        ts = ot.get("time")
        submit = None
        if ts:
            try:
                submit = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                submit = None
        return Order(id=str(oid), symbol=symbol, side=side, quantity=qty,
                     order_type=order_type, limit_price=lim, stop_price=stp,
                     status=status, filled_qty=filled_qty, avg_fill_price=avg,
                     submit_time=submit, broker="oanda")
