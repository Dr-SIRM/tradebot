"""RiskManager: the central gate every trade goes through.

Owns:
  - account equity & high-water marks (daily/weekly)
  - open positions
  - drawdown monitoring + halts
  - the approve_trade() pipeline that runs every check before submitting an order
  - position lifecycle (open/update/close) including trailing stops

Every method that mutates state is synchronous; the orchestrator calls them
serially per bar event so we don't need locks.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional

from utils.types import Position, Signal, Side, Trade
from utils.logging import get_logger
from risk.position_sizer import PositionSizer
from risk.correlation import CorrelationManager

log = get_logger(__name__)


class RiskManager:
    def __init__(self, cfg: dict, starting_equity: float):
        self.cfg = cfg["risk"]
        self.sizer = PositionSizer(self.cfg)
        self.corr = CorrelationManager(cfg.get("correlation", {}))

        self.starting_equity = starting_equity
        self.equity = starting_equity
        self.peak_equity = starting_equity

        # Daily / weekly P&L tracking
        self._day_anchor: Optional[datetime] = None
        self._day_start_equity = starting_equity
        self._week_anchor: Optional[datetime] = None
        self._week_start_equity = starting_equity

        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.closed_trades: list[Trade] = []

        self.daily_halted = False
        self.weekly_halted = False
        self.halt_reason = ""

    # --- equity bookkeeping --------------------------------------------------

    def update_equity_mark(self, marks: dict[str, float], now: datetime) -> None:
        """Refresh equity using current mark prices for open positions."""
        unrealized = sum(
            pos.unrealized_pnl(marks.get(sym, pos.entry_price))
            for sym, pos in self.positions.items()
        )
        realized_today = sum(
            t.pnl for t in self.closed_trades
            if self._day_anchor and t.exit_time >= self._day_anchor
        )
        # Equity = starting + all realized + open unrealized
        # (Realized total includes all-time, but day delta is what triggers halts.)
        all_time_realized = sum(t.pnl for t in self.closed_trades)
        self.equity = self.starting_equity + all_time_realized + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        self._roll_periods(now)

        # Drawdown halts
        day_dd = (self._day_start_equity - self.equity) / self._day_start_equity
        if day_dd >= self.cfg["daily_drawdown_halt_pct"] and not self.daily_halted:
            self.daily_halted = True
            self.halt_reason = f"daily drawdown {day_dd:.2%} >= {self.cfg['daily_drawdown_halt_pct']:.0%}"
            log.warning("DAILY HALT: %s (today realized: %.2f)", self.halt_reason, realized_today)

        week_dd = (self._week_start_equity - self.equity) / self._week_start_equity
        if week_dd >= self.cfg["weekly_drawdown_halt_pct"] and not self.weekly_halted:
            self.weekly_halted = True
            self.halt_reason = f"weekly drawdown {week_dd:.2%} >= {self.cfg['weekly_drawdown_halt_pct']:.0%}"
            log.error("WEEKLY HALT: %s", self.halt_reason)

    def _roll_periods(self, now: datetime) -> None:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        # Daily roll at UTC midnight
        if self._day_anchor is None or now.date() != self._day_anchor.date():
            self._day_anchor = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            self._day_start_equity = self.equity
            self.daily_halted = False
        # Weekly roll on Monday UTC
        iso = now.isocalendar()
        if self._week_anchor is None:
            self._week_anchor = now
            self._week_start_equity = self.equity
        else:
            wk_iso = self._week_anchor.isocalendar()
            if (iso.year, iso.week) != (wk_iso.year, wk_iso.week):
                self._week_anchor = now
                self._week_start_equity = self.equity
                self.weekly_halted = False

    # --- trade gates ---------------------------------------------------------

    def approve_trade(self, signal: Signal, now: datetime) -> tuple[bool, str, Optional[dict]]:
        """Run every pre-trade gate. Returns (ok, reason, plan_or_None).

        plan dict on success contains: {quantity, risk_amount, conviction_mult}.
        """
        self._roll_periods(now)

        if self.weekly_halted:
            return False, f"halt: {self.halt_reason}", None
        if self.daily_halted:
            return False, f"halt: {self.halt_reason}", None

        # Already in this symbol?
        if signal.symbol in self.positions:
            return False, f"already in position on {signal.symbol}", None

        # Max concurrent positions
        if len(self.positions) >= self.cfg["max_concurrent_positions"]:
            return False, (f"max concurrent positions reached: "
                           f"{len(self.positions)}/{self.cfg['max_concurrent_positions']}"), None

        # Reward:risk
        rr = signal.reward_risk_ratio()
        if rr < self.cfg["min_reward_risk_ratio"]:
            return False, f"reward:risk {rr:.2f} < {self.cfg['min_reward_risk_ratio']}", None

        # Stop sanity
        if signal.side == Side.LONG and signal.stop >= signal.entry:
            return False, "long signal has stop >= entry", None
        if signal.side == Side.SHORT and signal.stop <= signal.entry:
            return False, "short signal has stop <= entry", None

        # Correlation / asset-class caps
        ok, reason = self.corr.can_open(signal.symbol, signal.asset_class,
                                         list(self.positions.values()))
        if not ok:
            return False, reason, None

        qty, risk_amount, conviction_mult = self.sizer.size(signal, self.equity)
        if qty <= 0:
            return False, "computed quantity is zero", None

        return True, "approved", {
            "quantity": qty,
            "risk_amount": risk_amount,
            "conviction_mult": conviction_mult,
        }

    # --- position lifecycle --------------------------------------------------

    def open_position(self, signal: Signal, qty: float, fill_price: float,
                      risk_amount: float, broker: str, now: datetime) -> Position:
        pos = Position(
            symbol=signal.symbol, asset_class=signal.asset_class, side=signal.side,
            quantity=qty, entry_price=fill_price,
            stop_price=signal.stop, take_profit=signal.take_profit,
            initial_stop=signal.stop, open_time=now,
            strategy=signal.strategy, broker=broker, risk_amount=risk_amount,
            high_water_mark=fill_price,
            metadata={"signal_meta": signal.metadata, "regime": signal.regime.value,
                      "ml_score": signal.ml_score},
        )
        self.positions[signal.symbol] = pos
        log.info("OPEN %s %s qty=%.6f entry=%.4f stop=%.4f tp=%.4f risk=$%.2f strat=%s",
                 pos.side.value.upper(), pos.symbol, qty, fill_price, pos.stop_price,
                 pos.take_profit, risk_amount, pos.strategy)
        return pos

    def update_trailing(self, symbol: str, mark: float) -> None:
        """Tighten stop using trailing logic, once trade is past activate_R."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        # Update HWM
        if pos.side == Side.LONG:
            pos.high_water_mark = max(pos.high_water_mark, mark)
        else:
            pos.high_water_mark = min(pos.high_water_mark, mark)

        r_mult = pos.r_multiple(mark)
        if r_mult < self.cfg["trailing_activate_R"]:
            return

        # Use signal-time ATR if stored; else estimate from initial risk distance
        initial_risk = abs(pos.entry_price - pos.initial_stop)
        trail_dist = self.cfg["trailing_stop_atr_mult"] * initial_risk

        if pos.side == Side.LONG:
            new_stop = pos.high_water_mark - trail_dist
            if new_stop > pos.stop_price:
                log.info("TRAIL %s long: stop %.4f -> %.4f (R=%.2f)",
                         pos.symbol, pos.stop_price, new_stop, r_mult)
                pos.stop_price = new_stop
        else:
            new_stop = pos.high_water_mark + trail_dist
            if new_stop < pos.stop_price:
                log.info("TRAIL %s short: stop %.4f -> %.4f (R=%.2f)",
                         pos.symbol, pos.stop_price, new_stop, r_mult)
                pos.stop_price = new_stop

    def check_exits(self, symbol: str, bar_high: float, bar_low: float,
                    bar_close: float) -> tuple[bool, str, float]:
        """Check stop/TP touches on the latest bar. Returns (should_exit,
        reason, exit_price). Stop is checked first (conservative)."""
        pos = self.positions.get(symbol)
        if pos is None:
            return False, "", 0.0

        if pos.side == Side.LONG:
            if bar_low <= pos.stop_price:
                return True, "stop", pos.stop_price
            if bar_high >= pos.take_profit:
                return True, "take_profit", pos.take_profit
        else:
            if bar_high >= pos.stop_price:
                return True, "stop", pos.stop_price
            if bar_low <= pos.take_profit:
                return True, "take_profit", pos.take_profit
        return False, "", 0.0

    def close_position(self, symbol: str, exit_price: float, now: datetime,
                       reason: str, fees: float = 0.0) -> Optional[Trade]:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        if pos.side == Side.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity - fees
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity - fees

        pnl_pct = pnl / (pos.entry_price * pos.quantity) if pos.quantity > 0 else 0.0
        risk_per_unit = abs(pos.entry_price - pos.initial_stop)
        r_mult = ((exit_price - pos.entry_price) if pos.side == Side.LONG
                  else (pos.entry_price - exit_price)) / risk_per_unit if risk_per_unit > 0 else 0.0

        trade = Trade(
            symbol=pos.symbol, asset_class=pos.asset_class, side=pos.side,
            strategy=pos.strategy, entry_time=pos.open_time, exit_time=now,
            entry_price=pos.entry_price, exit_price=exit_price,
            quantity=pos.quantity, pnl=pnl, pnl_pct=pnl_pct, r_multiple=r_mult,
            exit_reason=reason, fees=fees,
            metadata=pos.metadata,
        )
        self.closed_trades.append(trade)
        log.info("CLOSE %s %s qty=%.6f entry=%.4f exit=%.4f pnl=%.2f R=%.2f reason=%s",
                 pos.side.value.upper(), pos.symbol, pos.quantity, pos.entry_price,
                 exit_price, pnl, r_mult, reason)
        return trade

    # --- info ---------------------------------------------------------------

    def stats(self) -> dict:
        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]
        return {
            "equity": round(self.equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "open_positions": len(self.positions),
            "trades_total": len(self.closed_trades),
            "trades_won": len(wins),
            "trades_lost": len(losses),
            "win_rate": (len(wins) / len(self.closed_trades)) if self.closed_trades else 0.0,
            "daily_halted": self.daily_halted,
            "weekly_halted": self.weekly_halted,
            "halt_reason": self.halt_reason,
        }
