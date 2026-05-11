"""Abstract broker interface. Every broker adapter (paper, alpaca, binance,
oanda) implements this contract. The router and orchestrator only depend on
this — strategy code never imports a specific broker.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from utils.types import Order, OrderType, Side


class BrokerBase(ABC):
    """Async broker interface."""

    name: str = "base"

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def get_equity(self) -> float: ...

    @abstractmethod
    async def get_position(self, symbol: str) -> dict | None:
        """Returns {symbol, qty, avg_entry, side} or None."""

    @abstractmethod
    async def submit_order(self, symbol: str, side: Side, qty: float,
                            order_type: OrderType, limit_price: float | None = None,
                            stop_price: float | None = None,
                            client_order_id: str = "") -> Order:
        """Submit an order. Returns an Order object with broker-assigned ID."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    async def get_order(self, order_id: str) -> Order: ...

    @abstractmethod
    async def get_quote(self, symbol: str) -> tuple[float, float]:
        """Returns (bid, ask). Used for limit-order pricing."""
