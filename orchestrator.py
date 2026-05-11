"""Orchestrator: the main async event loop.

Per symbol, runs two LiveFeeds (HTF + LTF). On each closed LTF bar:
  1. Update mark prices and equity in RiskManager
  2. Run trailing stop updates
  3. Check stop/TP exits and close positions if hit
  4. (If no position) ask StrategySelector for a signal
  5. Apply news blackout, ML score (if enabled), then risk approval
  6. Submit entry via OrderRouter
  7. Push state updates to Dashboard and Notifier

Errors in one symbol's task never bring down others; each task self-heals via
LiveFeed's own backoff/reconnect, plus a top-level supervisor that restarts a
crashed task.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from alerts.notifier import Notifier
from dashboard.tui import Dashboard
from data.calendar import EconomicCalendar
from data.feed import LiveFeed, make_data_adapter
from data.indicators import add_all_indicators
from execution.broker_base import BrokerBase
from execution.router import OrderRouter
from ml.model import TradeScorer
from risk.manager import RiskManager
from strategy.selector import StrategySelector
from utils.types import Side, Signal, Trade

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.starting_equity = float(cfg["account"]["starting_equity"])

        # Build core components
        self.risk = RiskManager(cfg, starting_equity=self.starting_equity)
        self.selector = StrategySelector(cfg)

        # Calendar (news blackout)
        cal_cfg = cfg.get("calendar", {})
        self.calendar = EconomicCalendar(
            events_file=cal_cfg.get("events_file"),
            blackout_minutes=cal_cfg.get("blackout_minutes", 15),
        )

        # Brokers + router (built by run_live; orchestrator just holds refs)
        self.brokers: Dict[str, BrokerBase] = {}
        self.router: Optional[OrderRouter] = None

        # Per-symbol state
        self.symbol_cfgs: Dict[str, Dict[str, Any]] = {
            s["symbol"]: s for s in cfg["universe"]
        }
        self.htf_data: Dict[str, pd.DataFrame] = {}  # symbol -> HTF bars
        self.ltf_data: Dict[str, pd.DataFrame] = {}  # symbol -> LTF bars
        self.last_marks: Dict[str, float] = {}

        # ML scorer (optional)
        ml_cfg = cfg.get("ml", {})
        self.ml_enabled = ml_cfg.get("enabled", False)
        self.ml_min_score = ml_cfg.get("min_score", 0.55)
        self.ml_scorer: Optional[TradeScorer] = None
        if self.ml_enabled:
            try:
                self.ml_scorer = TradeScorer.load(ml_cfg["model_path"])
                logger.info("ML scorer loaded from %s", ml_cfg["model_path"])
            except Exception as e:
                logger.warning("ML scorer load failed (%s) — disabling ML scoring", e)
                self.ml_enabled = False

        # Notifier + dashboard
        self.notifier = Notifier(cfg)
        self.trade_log: List[Trade] = []
        self.dashboard: Optional[Dashboard] = None
        if cfg.get("dashboard", {}).get("enabled", True):
            self.dashboard = Dashboard(cfg, self.risk, self.brokers, self.trade_log)

        # Concurrency controls
        self._stop = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._approval_lock = asyncio.Lock()  # serialize entry approvals

    # ------------------------------------------------------------------ setup
    def attach_brokers(self, brokers: Dict[str, BrokerBase]) -> None:
        """Called by run_live after brokers are constructed and connected."""
        self.brokers = brokers
        self.router = OrderRouter(self.cfg, brokers)
        if self.dashboard is not None:
            self.dashboard.brokers = brokers

    # ------------------------------------------------------------------- main
    async def run(self) -> None:
        """Start all per-symbol tasks and supervise them."""
        if self.router is None:
            raise RuntimeError("attach_brokers() must be called before run()")

        await self.notifier.info(
            "Bot started",
            f"Mode: {self.cfg['mode']}  Universe: "
            f"{', '.join(self.symbol_cfgs.keys())}  "
            f"Equity: ${self.starting_equity:,.2f}",
        )

        # Per-symbol trading tasks
        for symbol, sym_cfg in self.symbol_cfgs.items():
            t = asyncio.create_task(
                self._supervise_symbol(symbol, sym_cfg), name=f"sym:{symbol}"
            )
            self._tasks.append(t)

        # Periodic background loop: equity marking, daily summary, halt watch
        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="heartbeat"))

        # Dashboard
        if self.dashboard is not None:
            self._tasks.append(asyncio.create_task(self.dashboard.run(), name="dashboard"))

        try:
            await self._stop.wait()
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        self._stop.set()

    async def _shutdown(self) -> None:
        logger.info("orchestrator: shutdown")
        if self.dashboard:
            self.dashboard.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for b in self.brokers.values():
            try:
                await b.disconnect()
            except Exception:
                pass
        await self.notifier.info("Bot stopped", f"Final equity: ${self.risk.equity:,.2f}")

    # ----------------------------------------------------- per-symbol loop
    async def _supervise_symbol(self, symbol: str, sym_cfg: Dict[str, Any]) -> None:
        """Restart the symbol task on uncaught exceptions, with backoff."""
        backoff = 5
        while not self._stop.is_set():
            try:
                await self._run_symbol(symbol, sym_cfg)
                # graceful end → exit supervisor too
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("symbol %s task crashed: %s — restarting in %ds",
                                 symbol, e, backoff)
                await self.notifier.error(f"{symbol} task crashed: {e} (restarting)")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                    return  # stop signaled
                except asyncio.TimeoutError:
                    backoff = min(backoff * 2, 120)

    async def _run_symbol(self, symbol: str, sym_cfg: Dict[str, Any]) -> None:
        broker_name = sym_cfg["broker"]
        asset_class = sym_cfg["asset_class"]
        tf_high = sym_cfg["timeframe_high"]
        tf_low = sym_cfg["timeframe_low"]

        # Build a data adapter for this broker
        adapter_cfg = self.cfg["brokers"][broker_name]
        adapter = make_data_adapter(broker_name, adapter_cfg)

        warmup_bars = self.cfg["data"].get("warmup_bars", 250)
        htf_feed = LiveFeed(adapter, symbol, tf_high, warmup_bars=warmup_bars)
        ltf_feed = LiveFeed(adapter, symbol, tf_low, warmup_bars=warmup_bars)

        # Warmup
        self.htf_data[symbol] = await htf_feed.warmup()
        self.ltf_data[symbol] = await ltf_feed.warmup()
        if self.htf_data[symbol].empty or self.ltf_data[symbol].empty:
            logger.error("symbol %s: empty warmup data — aborting symbol loop", symbol)
            return

        # Mark price seed
        self.last_marks[symbol] = float(self.ltf_data[symbol]["close"].iloc[-1])

        # Run HTF feed in background to keep htf_data fresh
        htf_task = asyncio.create_task(self._htf_updater(symbol, htf_feed),
                                       name=f"htf:{symbol}")
        try:
            async for bar in ltf_feed.stream():
                if self._stop.is_set():
                    break
                await self._on_ltf_bar(symbol, sym_cfg, bar, broker_name,
                                       asset_class, tf_low)
        finally:
            htf_task.cancel()
            try:
                await htf_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                if hasattr(adapter, "close"):
                    await adapter.close()
            except Exception:
                pass

    async def _htf_updater(self, symbol: str, htf_feed: LiveFeed) -> None:
        """Append HTF bars as they close."""
        async for bar in htf_feed.stream():
            df = self.htf_data[symbol]
            ts = bar.name
            if ts not in df.index:
                df.loc[ts] = bar
                # Cap size
                max_bars = self.cfg["data"].get("max_bars_in_memory", 1000)
                if len(df) > max_bars:
                    self.htf_data[symbol] = df.iloc[-max_bars:]

    # ----------------------------------------------------- per-bar handler
    async def _on_ltf_bar(self, symbol: str, sym_cfg: Dict[str, Any],
                          bar: pd.Series, broker_name: str,
                          asset_class: str, tf_low: str) -> None:
        # Append to LTF history
        df = self.ltf_data[symbol]
        ts = bar.name
        if ts in df.index:
            return  # duplicate; ignore
        df.loc[ts] = bar
        max_bars = self.cfg["data"].get("max_bars_in_memory", 1000)
        if len(df) > max_bars:
            df = df.iloc[-max_bars:]
            self.ltf_data[symbol] = df

        close_px = float(bar["close"])
        high_px = float(bar["high"])
        low_px = float(bar["low"])
        self.last_marks[symbol] = close_px

        now = datetime.now(timezone.utc)

        # 1. Mark equity
        self.risk.update_equity_mark(self.last_marks, now)

        # 2. Update trailing stops on any open position for this symbol
        self.risk.update_trailing(symbol, close_px)

        # 3. Check stop/TP exits
        should_exit, reason, exit_price = self.risk.check_exits(symbol, high_px, low_px, close_px)
        if should_exit:
            await self._close_position(symbol, exit_price, broker_name, reason, now)
            # After exiting, do not enter on the same bar
            self._publish_state(symbol, regime="-", strategy="-", conviction=0,
                                signal_side="-")
            return

        # If we already have a position, just update state — don't pyramid
        if symbol in self.risk.positions:
            self._publish_state(symbol, regime="held", strategy="-", conviction=0,
                                signal_side="-")
            return

        # 4. Generate signal via selector
        # Compute indicators on a copy for strategy evaluation
        try:
            ltf_ind = add_all_indicators(self.ltf_data[symbol], self.cfg)
            htf_ind = add_all_indicators(self.htf_data[symbol], self.cfg)
        except Exception as e:
            logger.warning("%s indicator computation failed: %s", symbol, e)
            return

        regime, signal = self.selector.select(htf_ind, ltf_ind, symbol, asset_class, tf_low)

        self._publish_state(
            symbol,
            regime=regime.value,
            strategy=signal.strategy if signal else "-",
            conviction=signal.conviction if signal else 0,
            signal_side=signal.side.value if signal else "-",
        )

        if signal is None:
            return

        # 5. News blackout
        is_blackout, blk_reason = self.calendar.is_blackout(symbol, now)
        if is_blackout:
            logger.info("%s signal skipped: blackout (%s)", symbol, blk_reason)
            return

        # 6. ML scoring
        if self.ml_enabled and self.ml_scorer is not None:
            try:
                from ml.features import build_features
                feats = build_features(signal, ltf_ind, htf_ind)
                if feats is not None:
                    score = float(self.ml_scorer.score(feats))
                    signal.ml_score = score
                    if score < self.ml_min_score:
                        logger.info("%s signal filtered by ML (score=%.3f < %.3f)",
                                    symbol, score, self.ml_min_score)
                        return
            except Exception as e:
                logger.warning("ML scoring error on %s: %s", symbol, e)

        # 7. Risk approval (serialized to prevent races on max_concurrent)
        async with self._approval_lock:
            ok, reason, plan = self.risk.approve_trade(signal, now)
            if not ok:
                logger.info("%s signal rejected: %s", symbol, reason)
                # Halt notifications
                if reason.startswith("halt:"):
                    await self.notifier.halt(reason, self.risk.equity,
                                             (self.risk.peak_equity - self.risk.equity)
                                             / max(self.risk.peak_equity, 1))
                return

            qty = plan["quantity"]
            risk_amount = plan["risk_amount"]

            # 8. Submit entry
            try:
                order = await self.router.execute_entry(signal, qty, broker_name)
            except Exception as e:
                logger.exception("%s entry submit failed: %s", symbol, e)
                await self.notifier.error(f"{symbol} entry failed: {e}")
                return

            if order is None or not getattr(order, "avg_fill_price", 0):
                logger.warning("%s entry not filled (order=%s)", symbol, order)
                return

            fill_price = float(order.avg_fill_price)
            fill_qty = float(order.filled_qty) if order.filled_qty else qty
            pos = self.risk.open_position(signal, fill_qty, fill_price,
                                          risk_amount, broker_name, now)

        # 9. Notify entry
        await self.notifier.entry(
            symbol=pos.symbol,
            side=pos.side.value.upper(),
            qty=pos.quantity,
            entry=pos.entry_price,
            stop=pos.stop_price,
            target=pos.take_profit,
            strategy=pos.strategy,
            conviction=signal.conviction,
        )

    # ----------------------------------------------------- close helper
    async def _close_position(self, symbol: str, exit_price: float,
                              broker_name: str, reason: str,
                              now: datetime) -> None:
        pos = self.risk.positions.get(symbol)
        if pos is None:
            return
        try:
            assert self.router is not None
            order = await self.router.execute_exit(
                symbol=symbol,
                side_to_close=pos.side,
                qty=pos.quantity,
                broker_name=broker_name,
                strategy=pos.strategy,
                reason=reason,
            )
            actual_exit = exit_price
            if order is not None and getattr(order, "avg_fill_price", 0):
                actual_exit = float(order.avg_fill_price)
        except Exception as e:
            logger.exception("%s exit submit failed: %s", symbol, e)
            await self.notifier.error(f"{symbol} exit failed: {e}")
            actual_exit = exit_price  # close in book regardless

        trade = self.risk.close_position(symbol, actual_exit, now, reason)
        if trade is not None:
            self.trade_log.append(trade)
            await self.notifier.exit(trade)

    # ----------------------------------------------------- background loop
    async def _heartbeat_loop(self) -> None:
        """Periodic equity mark + daily-summary watch + halt notifications."""
        last_halt_state = (False, False)
        while not self._stop.is_set():
            try:
                now = datetime.now(timezone.utc)
                self.risk.update_equity_mark(self.last_marks, now)

                # Halt-state edge detection
                cur_state = (self.risk.daily_halted, self.risk.weekly_halted)
                if cur_state != last_halt_state:
                    if self.risk.daily_halted and not last_halt_state[0]:
                        await self.notifier.halt(
                            f"daily DD halt: {self.risk.halt_reason}",
                            self.risk.equity,
                            (self.risk.peak_equity - self.risk.equity)
                            / max(self.risk.peak_equity, 1),
                        )
                    if self.risk.weekly_halted and not last_halt_state[1]:
                        await self.notifier.halt(
                            f"weekly DD halt: {self.risk.halt_reason}",
                            self.risk.equity,
                            (self.risk.peak_equity - self.risk.equity)
                            / max(self.risk.peak_equity, 1),
                        )
                    last_halt_state = cur_state

                # Daily summary
                stats = self.risk.stats()
                day_start = self.risk._day_start_equity
                day_pnl = self.risk.equity - day_start
                day_pnl_pct = day_pnl / day_start if day_start else 0
                # Trades today
                today = now.date()
                day_trades = [t for t in self.risk.closed_trades
                              if t.exit_time.date() == today]
                day_wins = [t for t in day_trades if t.pnl > 0]
                win_rate = len(day_wins) / len(day_trades) if day_trades else 0.0
                await self.notifier.maybe_send_daily_summary(
                    self.risk.equity, day_pnl, day_pnl_pct,
                    len(day_trades), win_rate, len(self.risk.positions),
                )
            except Exception as e:
                logger.warning("heartbeat error: %s", e)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=30)
                return  # stop signaled
            except asyncio.TimeoutError:
                continue

    # ----------------------------------------------------- dashboard helper
    def _publish_state(self, symbol: str, regime: str, strategy: str,
                       conviction: float, signal_side: str) -> None:
        if self.dashboard is None:
            return
        self.dashboard.update_signal(symbol, {
            "regime": regime, "strategy": strategy,
            "conviction": conviction, "signal_side": signal_side,
        })
