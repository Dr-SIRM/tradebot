"""Real-time terminal UI dashboard using rich.

Shows: account equity, daily P&L, open positions, recent trades, active signals,
risk status (halt flags, drawdown). Read-only — pulls state from RiskManager
and broker.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from execution.broker_base import BrokerBase
from risk.manager import RiskManager
from utils.types import Trade

logger = logging.getLogger(__name__)


class Dashboard:
    def __init__(self, cfg: Dict[str, Any], risk: RiskManager,
                 brokers: Dict[str, BrokerBase], trade_log: List[Trade]):
        self.cfg = cfg
        self.risk = risk
        self.brokers = brokers
        self.trade_log = trade_log  # shared mutable list, appended by orchestrator
        self.refresh_seconds = cfg.get("dashboard", {}).get("refresh_seconds", 2)
        self.console = Console()
        self._live_signals: Dict[str, Dict[str, Any]] = {}  # symbol -> latest signal info
        self._stop = False

    def update_signal(self, symbol: str, info: Dict[str, Any]) -> None:
        """Called by orchestrator to publish the latest signal/regime per symbol."""
        info["updated"] = datetime.now(timezone.utc)
        self._live_signals[symbol] = info

    def stop(self) -> None:
        self._stop = True

    # Render helpers ---------------------------------------------------------
    def _render_header(self) -> Panel:
        stats = self.risk.stats()
        eq = stats["equity"]
        peak = stats["peak_equity"]
        # Pull period anchors directly from RiskManager (not exposed via stats())
        day_start = self.risk._day_start_equity
        week_start = self.risk._week_start_equity

        day_pnl = eq - day_start
        day_pnl_pct = day_pnl / day_start if day_start else 0
        week_pnl = eq - week_start
        week_pnl_pct = week_pnl / week_start if week_start else 0
        peak_dd = (peak - eq) / peak if peak else 0

        mode = self.cfg.get("mode", "paper").upper()
        halted = stats["daily_halted"] or stats["weekly_halted"]

        text = Text()
        text.append(f"  Mode: {mode}", style="bold cyan")
        text.append("    ")
        text.append(f"Equity: ${eq:,.2f}", style="bold white")
        text.append("    ")
        day_style = "green" if day_pnl >= 0 else "red"
        text.append(f"Day: ${day_pnl:+,.2f} ({day_pnl_pct*100:+.2f}%)", style=day_style)
        text.append("    ")
        wk_style = "green" if week_pnl >= 0 else "red"
        text.append(f"Week: ${week_pnl:+,.2f} ({week_pnl_pct*100:+.2f}%)", style=wk_style)
        text.append("    ")
        text.append(f"DD from peak: {peak_dd*100:.2f}%",
                    style="red" if peak_dd > 0.02 else "yellow")
        if halted:
            text.append("    ")
            text.append(f"HALTED: {stats.get('halt_reason', '')}", style="bold red on white")
        return Panel(text, title="Account", border_style="cyan")

    def _render_positions(self) -> Panel:
        table = Table(expand=True)
        table.add_column("Symbol")
        table.add_column("Side")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Stop", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Strategy")
        table.add_column("Opened")

        positions = list(self.risk.positions.values())
        for pos in positions:
            opened = pos.open_time.strftime("%H:%M:%S") if pos.open_time else "-"
            side_color = "green" if pos.side.value == "long" else "red"
            table.add_row(
                pos.symbol,
                Text(pos.side.value.upper(), style=side_color),
                f"{pos.quantity:.4f}",
                f"{pos.entry_price:.4f}",
                f"{pos.stop_price:.4f}",
                f"{pos.take_profit:.4f}",
                pos.strategy or "-",
                opened,
            )
        if not positions:
            table.add_row("-", "-", "-", "-", "-", "-", "-", "-")
        return Panel(table, title=f"Open Positions ({len(positions)})", border_style="green")

    def _render_signals(self) -> Panel:
        table = Table(expand=True)
        table.add_column("Symbol")
        table.add_column("Regime")
        table.add_column("Strategy")
        table.add_column("Conviction", justify="right")
        table.add_column("Last Signal")
        table.add_column("Updated")

        for symbol, info in sorted(self._live_signals.items()):
            updated = info.get("updated")
            updated_str = updated.strftime("%H:%M:%S") if updated else "-"
            sig_side = info.get("signal_side", "-")
            sig_style = "green" if sig_side == "long" else ("red" if sig_side == "short" else "dim")
            table.add_row(
                symbol,
                info.get("regime", "-"),
                info.get("strategy", "-"),
                f"{info.get('conviction', 0):.2f}",
                Text(sig_side.upper(), style=sig_style),
                updated_str,
            )
        if not self._live_signals:
            table.add_row("-", "-", "-", "-", "-", "-")
        return Panel(table, title="Live Signals", border_style="magenta")

    def _render_recent_trades(self) -> Panel:
        table = Table(expand=True)
        table.add_column("Time")
        table.add_column("Symbol")
        table.add_column("Side")
        table.add_column("Strategy")
        table.add_column("R", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Reason")

        recent = self.trade_log[-10:][::-1]
        for tr in recent:
            ts = tr.exit_time.strftime("%m-%d %H:%M") if tr.exit_time else "-"
            r_style = "green" if tr.r_multiple > 0 else "red"
            table.add_row(
                ts,
                tr.symbol,
                tr.side.value.upper(),
                tr.strategy,
                Text(f"{tr.r_multiple:+.2f}", style=r_style),
                Text(f"${tr.pnl:+,.2f}", style=r_style),
                tr.exit_reason,
            )
        if not recent:
            table.add_row("-", "-", "-", "-", "-", "-", "-")
        return Panel(table, title="Recent Trades", border_style="yellow")

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="positions"),
            Layout(name="trades"),
        )
        layout["right"].update(self._render_signals())
        layout["header"].update(self._render_header())
        layout["positions"].update(self._render_positions())
        layout["trades"].update(self._render_recent_trades())
        return layout

    async def run(self) -> None:
        """Run the live dashboard until stop() is called."""
        try:
            with Live(self._build_layout(), console=self.console,
                      refresh_per_second=1, screen=False) as live:
                while not self._stop:
                    await asyncio.sleep(self.refresh_seconds)
                    try:
                        live.update(self._build_layout())
                    except Exception as e:
                        logger.warning("Dashboard render error: %s", e)
        except Exception as e:
            logger.exception("Dashboard crashed: %s", e)
