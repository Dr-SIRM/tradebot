"""Economic calendar — blackout windows around high-impact news.

The default `manual` provider reads a static YAML file; production users should
swap in a ForexFactory/Investing.com scraper or a paid API. The interface is the
same: `is_blackout(symbol, ts) -> bool`.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml

from utils.logging import get_logger

log = get_logger(__name__)


class EconomicCalendar:
    """Static manual calendar.

    Events file format:
        events:
          - time: '2024-12-18T19:00:00Z'   # UTC
            currency: USD
            impact: high                    # high | medium | low
            name: FOMC Statement
    """

    # Symbol → list of currencies whose events affect it
    _SYMBOL_CURRENCIES = {
        "BTC/USDT": ["USD"], "ETH/USDT": ["USD"], "SOL/USDT": ["USD"],
        "SPY": ["USD"], "QQQ": ["USD"], "IWM": ["USD"],
        "EUR_USD": ["EUR", "USD"], "GBP_USD": ["GBP", "USD"],
        "USD_JPY": ["USD", "JPY"],
        "XAU_USD": ["USD"], "XAG_USD": ["USD"],
        "WTI_USD": ["USD"],
    }

    def __init__(self, events_file: str | Path | None,
                 blackout_minutes: int = 15):
        self.blackout = timedelta(minutes=blackout_minutes)
        self.events: list[dict] = []
        if events_file:
            p = Path(events_file)
            if p.exists():
                with p.open("r") as f:
                    data = yaml.safe_load(f) or {}
                for e in data.get("events", []):
                    try:
                        e_time = datetime.fromisoformat(e["time"].replace("Z", "+00:00"))
                        if e_time.tzinfo is None:
                            e_time = e_time.replace(tzinfo=timezone.utc)
                        self.events.append({
                            "time": e_time,
                            "currency": e["currency"].upper(),
                            "impact": e.get("impact", "high").lower(),
                            "name": e.get("name", ""),
                        })
                    except (KeyError, ValueError) as ex:
                        log.warning("bad event entry: %s (%s)", e, ex)
                log.info("loaded %d economic events from %s", len(self.events), p)
            else:
                log.warning("events file not found: %s", p)

    def is_blackout(self, symbol: str, ts: datetime) -> tuple[bool, str]:
        """Return (in_blackout, reason). Only high-impact events block trading."""
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        currencies = self._SYMBOL_CURRENCIES.get(symbol, [])
        if not currencies:
            return False, ""
        for e in self.events:
            if e["impact"] != "high":
                continue
            if e["currency"] not in currencies:
                continue
            if abs((ts - e["time"]).total_seconds()) <= self.blackout.total_seconds():
                return True, f"blackout: {e['name']} ({e['currency']}) at {e['time'].isoformat()}"
        return False, ""
