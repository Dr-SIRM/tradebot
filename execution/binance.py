"""Binance live broker (crypto spot; futures requires changing the ccxt market type).

Uses ccxt's async API for native asyncio support.

⚠️ TEST IN TESTNET FIRST. Real considerations not handled here:
  - Tick size and lot step rounding (use exchange.markets[symbol]['precision']).
  - Minimum notional thresholds.
  - Reduce-only flags for futures.
  - Margin mode (isolated vs cross).
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone

from execution.broker_base import BrokerBase
from utils.types import Order, OrderStatus, OrderType, Side
from utils.logging import get_logger

log = get_logger(__name__)


class BinanceBroker(BrokerBase):
    name = "binance"

    def __init__(self, cfg: dict):
        import ccxt.async_support as ccxt
        bcfg = cfg["brokers"]["binance"]
        self.ex = ccxt.binance({
            "apiKey": bcfg.get("api_key", ""),
            "secret": bcfg.get("api_secret", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if bcfg.get("testnet", True):
            self.ex.set_sandbox_mode(True)

    async def connect(self) -> None:
        await self.ex.load_markets()
        bal = await self.ex.fetch_balance()
        usdt = bal.get("USDT", {}).get("free", 0)
        log.info("binance connected: USDT free=%s", usdt)

    async def disconnect(self) -> None:
        await self.ex.close()

    async def get_equity(self) -> float:
        bal = await self.ex.fetch_balance()
        # Best-effort total USD value; real impls should sum across all assets at last price.
        return float(bal.get("USDT", {}).get("total", 0))

    async def get_position(self, symbol: str) -> dict | None:
        # Spot has no "positions" — derive from balance of base asset.
        base = symbol.split("/")[0]
        bal = await self.ex.fetch_balance()
        amt = float(bal.get(base, {}).get("total", 0))
        if amt <= 0:
            return None
        ticker = await self.ex.fetch_ticker(symbol)
        return {"symbol": symbol, "qty": amt,
                "avg_entry": float(ticker["last"]), "side": "long"}

    async def get_quote(self, symbol: str) -> tuple[float, float]:
        ob = await self.ex.fetch_order_book(symbol, limit=5)
        bid = ob["bids"][0][0] if ob["bids"] else 0.0
        ask = ob["asks"][0][0] if ob["asks"] else 0.0
        return float(bid), float(ask)

    async def submit_order(self, symbol: str, side: Side, qty: float,
                            order_type: OrderType, limit_price: float | None = None,
                            stop_price: float | None = None,
                            client_order_id: str = "") -> Order:
        ccxt_side = "buy" if side == Side.LONG else "sell"
        ccxt_type = order_type.value
        params = {}
        if client_order_id:
            params["clientOrderId"] = client_order_id

        # ccxt expects price for limit, none for market
        price = limit_price if order_type == OrderType.LIMIT else None
        try:
            r = await self.ex.create_order(symbol, ccxt_type, ccxt_side, qty, price, params)
        except Exception as e:
            log.error("binance order rejected: %s", e)
            return Order(id=client_order_id or str(uuid.uuid4()), symbol=symbol,
                         side=side, quantity=qty, order_type=order_type,
                         status=OrderStatus.REJECTED, broker=self.name,
                         metadata={"error": str(e)})
        return self._to_order(r)

    async def cancel_order(self, order_id: str) -> bool:
        try:
            # ccxt requires symbol; we don't always have it — caller should track.
            # The router stores symbol alongside order_id, so this is a known limitation
            # for the bare interface. Implementation below tries common patterns.
            await self.ex.cancel_order(order_id)
            return True
        except Exception as e:
            log.warning("binance cancel failed %s: %s", order_id, e)
            return False

    async def get_order(self, order_id: str) -> Order:
        # Same caveat as cancel re: needing symbol. Real router calls this with symbol.
        r = await self.ex.fetch_order(order_id)
        return self._to_order(r)

    @staticmethod
    def _to_order(r: dict) -> Order:
        status_map = {
            "open": OrderStatus.OPEN, "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED, "expired": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        side = Side.LONG if r.get("side") == "buy" else Side.SHORT
        ts_ms = r.get("timestamp")
        submit = (datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                   if ts_ms else None)
        return Order(
            id=str(r.get("id", "")), symbol=r.get("symbol", ""), side=side,
            quantity=float(r.get("amount", 0) or 0),
            order_type=OrderType.LIMIT if r.get("type") == "limit" else OrderType.MARKET,
            limit_price=float(r.get("price")) if r.get("price") else None,
            status=status_map.get(r.get("status", "open"), OrderStatus.OPEN),
            filled_qty=float(r.get("filled", 0) or 0),
            avg_fill_price=float(r.get("average", 0) or 0),
            submit_time=submit, broker="binance",
        )
