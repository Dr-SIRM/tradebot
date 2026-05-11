"""Alert dispatch: Telegram, Discord, console.

Channels are async-fire-and-forget; failures are logged but never crash the
main trading loop.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

from utils.types import Trade

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    HALT = "HALT"
    SUMMARY = "SUMMARY"
    ERROR = "ERROR"
    INFO = "INFO"


@dataclass
class Alert:
    type: AlertType
    title: str
    body: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def format_text(self) -> str:
        return f"[{self.type.value}] {self.title}\n{self.body}"


class Channel:
    name: str = "channel"

    async def send(self, alert: Alert) -> None:
        raise NotImplementedError


class ConsoleChannel(Channel):
    name = "console"

    async def send(self, alert: Alert) -> None:
        ts = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n=== ALERT {ts} ===\n{alert.format_text()}\n")


class TelegramChannel(Channel):
    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    async def send(self, alert: Alert) -> None:
        payload = {
            "chat_id": self.chat_id,
            "text": alert.format_text(),
            "parse_mode": "Markdown",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload, timeout=10) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        logger.warning("Telegram send failed %s: %s", resp.status, text)
        except Exception as e:
            logger.warning("Telegram send error: %s", e)


class DiscordChannel(Channel):
    name = "discord"

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, alert: Alert) -> None:
        payload = {"content": alert.format_text()}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=10) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        logger.warning("Discord send failed %s: %s", resp.status, text)
        except Exception as e:
            logger.warning("Discord send error: %s", e)


class Notifier:
    """Routes alerts to all enabled channels concurrently."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        a = cfg.get("alerts", {})
        self.channels: List[Channel] = []

        if a.get("console", {}).get("enabled", True):
            self.channels.append(ConsoleChannel())

        tg = a.get("telegram", {})
        if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
            self.channels.append(TelegramChannel(tg["bot_token"], tg["chat_id"]))

        dc = a.get("discord", {})
        if dc.get("enabled") and dc.get("webhook_url"):
            self.channels.append(DiscordChannel(dc["webhook_url"]))

        # Daily summary scheduling
        ds = a.get("daily_summary", {})
        self.daily_summary_enabled = ds.get("enabled", False)
        self.daily_summary_hour_utc = int(ds.get("hour_utc", 23))
        self._last_summary_date: Optional[Any] = None

        # Alert filters
        ev = a.get("events", {})
        self.send_entry = ev.get("entry", True)
        self.send_exit = ev.get("exit", True)
        self.send_halt = ev.get("halt", True)
        self.send_error = ev.get("error", True)

        logger.info(
            "Notifier configured with %d channel(s): %s",
            len(self.channels),
            [c.name for c in self.channels],
        )

    async def send(self, alert: Alert) -> None:
        # Filter by event type
        if alert.type == AlertType.ENTRY and not self.send_entry:
            return
        if alert.type == AlertType.EXIT and not self.send_exit:
            return
        if alert.type == AlertType.HALT and not self.send_halt:
            return
        if alert.type == AlertType.ERROR and not self.send_error:
            return

        await asyncio.gather(
            *(c.send(alert) for c in self.channels), return_exceptions=True
        )

    # Convenience constructors -----------------------------------------------
    async def entry(self, symbol: str, side: str, qty: float, entry: float,
                    stop: float, target: float, strategy: str, conviction: float) -> None:
        body = (
            f"`{symbol}` {side} qty={qty:.4f}\n"
            f"Entry: {entry:.4f}  Stop: {stop:.4f}  Target: {target:.4f}\n"
            f"Strategy: {strategy}  Conviction: {conviction:.2f}\n"
            f"R:R = {abs(target - entry) / max(abs(entry - stop), 1e-9):.2f}"
        )
        await self.send(Alert(AlertType.ENTRY, f"{symbol} {side} entry", body))

    async def exit(self, trade: Trade) -> None:
        body = (
            f"`{trade.symbol}` {trade.side.value.upper()}\n"
            f"Entry: {trade.entry_price:.4f}  Exit: {trade.exit_price:.4f}\n"
            f"P&L: ${trade.pnl:.2f}  R: {trade.r_multiple:.2f}\n"
            f"Strategy: {trade.strategy}  Reason: {trade.exit_reason}"
        )
        await self.send(Alert(AlertType.EXIT, f"{trade.symbol} exit ({trade.exit_reason})", body))

    async def halt(self, reason: str, equity: float, dd: float) -> None:
        body = f"Reason: {reason}\nEquity: ${equity:,.2f}\nDrawdown: {dd*100:.2f}%"
        await self.send(Alert(AlertType.HALT, "Trading halted", body))

    async def error(self, msg: str) -> None:
        await self.send(Alert(AlertType.ERROR, "Error", msg))

    async def info(self, title: str, body: str) -> None:
        await self.send(Alert(AlertType.INFO, title, body))

    async def daily_summary(self, equity: float, day_pnl: float, day_pnl_pct: float,
                            n_trades: int, win_rate: float, open_positions: int) -> None:
        body = (
            f"Equity: ${equity:,.2f}\n"
            f"Day P&L: ${day_pnl:+,.2f} ({day_pnl_pct*100:+.2f}%)\n"
            f"Trades today: {n_trades}  Win rate: {win_rate*100:.1f}%\n"
            f"Open positions: {open_positions}"
        )
        await self.send(Alert(AlertType.SUMMARY, "Daily summary", body))

    async def maybe_send_daily_summary(self, equity: float, day_pnl: float,
                                       day_pnl_pct: float, n_trades: int,
                                       win_rate: float, open_positions: int) -> None:
        """Call periodically; sends once per day at configured UTC hour."""
        if not self.daily_summary_enabled:
            return
        now = datetime.now(timezone.utc)
        today = now.date()
        if self._last_summary_date == today:
            return
        if now.hour < self.daily_summary_hour_utc:
            return
        self._last_summary_date = today
        await self.daily_summary(equity, day_pnl, day_pnl_pct, n_trades, win_rate, open_positions)
