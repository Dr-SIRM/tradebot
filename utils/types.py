"""Core type definitions used across the bot."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"


class Regime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class Bar:
    """One OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    """A trade setup proposed by a strategy. Not yet risk-checked."""
    symbol: str
    asset_class: str
    side: Side
    strategy: str
    entry: float
    stop: float
    take_profit: float
    conviction: float                 # 0..1
    timestamp: datetime
    timeframe: str
    regime: Regime = Regime.UNKNOWN
    ml_score: Optional[float] = None  # filled later if ML enabled
    metadata: dict = field(default_factory=dict)

    def reward_risk_ratio(self) -> float:
        risk = abs(self.entry - self.stop)
        if risk <= 0:
            return 0.0
        reward = abs(self.take_profit - self.entry)
        return reward / risk


@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    submit_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    broker: str = ""
    strategy: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    asset_class: str
    side: Side
    quantity: float
    entry_price: float
    stop_price: float
    take_profit: float
    initial_stop: float
    open_time: datetime
    strategy: str
    broker: str
    risk_amount: float                # dollar amount risked at entry
    high_water_mark: float = 0.0      # for trailing stop (best price seen)
    realized_pnl: float = 0.0
    metadata: dict = field(default_factory=dict)

    def unrealized_pnl(self, mark: float) -> float:
        if self.side == Side.LONG:
            return (mark - self.entry_price) * self.quantity
        return (self.entry_price - mark) * self.quantity

    def r_multiple(self, mark: float) -> float:
        """How many R the trade is currently at (R = initial risk per unit)."""
        risk_per_unit = abs(self.entry_price - self.initial_stop)
        if risk_per_unit <= 0:
            return 0.0
        if self.side == Side.LONG:
            return (mark - self.entry_price) / risk_per_unit
        return (self.entry_price - mark) / risk_per_unit


@dataclass
class Trade:
    """A closed round-trip trade. Used for logging and metrics."""
    symbol: str
    asset_class: str
    side: Side
    strategy: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str
    fees: float = 0.0
    slippage: float = 0.0
    metadata: dict = field(default_factory=dict)
