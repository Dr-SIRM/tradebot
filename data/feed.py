"""Async OHLCV data feed.

Supports two modes:

1. **Historical** — load CSV/parquet from disk for backtesting. Schema:
   timestamp, open, high, low, close, volume.

2. **Live** — async generator that yields new bars as they close. Routes to the
   right adapter based on the symbol's `broker` config. Adapters use ccxt for
   crypto and the broker's REST/WS APIs for the others.

The feed deliberately yields *closed* bars only — never partial. Every strategy
runs on bar-close.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
import pandas as pd

from utils.logging import get_logger
from utils.time import timeframe_to_seconds, utc_now

log = get_logger(__name__)


def load_historical(path: str | Path) -> pd.DataFrame:
    """Load OHLCV CSV/parquet. Index becomes a UTC DatetimeIndex named 'timestamp'.

    Accepts CSVs with either a 'timestamp' column or with the timestamp as
    the first column (named or unnamed).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif df.columns[0] in ("", "Unnamed: 0") or df.columns[0].lower() in (
            "time", "date", "datetime", "ts"):
        first = df.columns[0]
        df[first] = pd.to_datetime(df[first], utc=True)
        df = df.set_index(first)
        df.index.name = "timestamp"
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Last resort: try to parse the existing index
        try:
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"
        except Exception:
            pass
    df = df.sort_index()
    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {p}: {missing}")
    return df


class LiveFeed:
    """Async live feed. One instance per (symbol, timeframe). Yields bars on close.

    Uses an injected adapter object that exposes:
        async def fetch_recent_ohlcv(symbol, timeframe, limit) -> list[Bar-like dict]

    The adapter is responsible for broker-specific rate limiting & reconnects.
    """

    def __init__(self, adapter, symbol: str, timeframe: str,
                 reconnect_seconds: int = 5, warmup_bars: int = 200):
        self.adapter = adapter
        self.symbol = symbol
        self.timeframe = timeframe
        self.reconnect_seconds = reconnect_seconds
        self.warmup_bars = warmup_bars
        self._last_ts: datetime | None = None

    async def warmup(self) -> pd.DataFrame:
        """Fetch enough history to seed indicators."""
        rows = await self.adapter.fetch_recent_ohlcv(
            self.symbol, self.timeframe, limit=self.warmup_bars
        )
        df = self._rows_to_df(rows)
        if not df.empty:
            self._last_ts = df.index[-1].to_pydatetime()
        log.info("warmup %s %s: %d bars", self.symbol, self.timeframe, len(df))
        return df

    async def stream(self) -> AsyncIterator[pd.Series]:
        """Yield each newly closed bar as a Series indexed by timestamp."""
        tf_secs = timeframe_to_seconds(self.timeframe)
        # Sleep at most a quarter of the timeframe to keep latency low while not hammering APIs
        poll_interval = max(2, min(tf_secs // 4, 30))

        while True:
            try:
                rows = await self.adapter.fetch_recent_ohlcv(
                    self.symbol, self.timeframe, limit=3
                )
                df = self._rows_to_df(rows)
                if df.empty:
                    await asyncio.sleep(poll_interval)
                    continue
                # Use second-to-last bar (last is still forming)
                if len(df) >= 2:
                    closed = df.iloc[-2]
                    closed_ts = df.index[-2].to_pydatetime()
                    if self._last_ts is None or closed_ts > self._last_ts:
                        self._last_ts = closed_ts
                        s = closed.copy()
                        s.name = closed_ts
                        yield s
                await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # broad: any adapter failure → backoff & retry
                log.warning("feed %s %s error: %s — retry in %ds",
                            self.symbol, self.timeframe, e, self.reconnect_seconds)
                await asyncio.sleep(self.reconnect_seconds)

    @staticmethod
    def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp").sort_index()


# Adapters --------------------------------------------------------------------
# Each adapter wraps a broker SDK and normalizes OHLCV. Adapters are intentionally
# minimal — full broker functionality (orders, balances) lives in execution/.

class CCXTAdapter:
    """Crypto OHLCV via ccxt. Used for binance, bybit, coinbase."""

    def __init__(self, exchange_id: str, api_key: str = "", api_secret: str = "",
                 testnet: bool = True):
        import ccxt.async_support as ccxt  # async variant
        self.ex = getattr(ccxt, exchange_id)({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        if testnet and exchange_id == "binance":
            self.ex.set_sandbox_mode(True)

    async def fetch_recent_ohlcv(self, symbol: str, timeframe: str,
                                  limit: int = 200) -> list[dict]:
        raw = await self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        # ccxt returns [ts_ms, o, h, l, c, v]
        return [{
            "timestamp": datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc),
            "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5],
        } for r in raw]

    async def close(self):
        await self.ex.close()


class AlpacaDataAdapter:
    """Alpaca market-data adapter for equities/ETFs.

    Wraps alpaca-py's StockHistoricalDataClient. The SDK is sync; we run it in
    the default executor to avoid blocking the event loop.
    """

    _TF_MAP = {"1m": "1Min", "5m": "5Min", "15m": "15Min",
               "1h": "1Hour", "1d": "1Day"}

    def __init__(self, api_key: str, api_secret: str):
        from alpaca.data.historical import StockHistoricalDataClient
        self.client = StockHistoricalDataClient(api_key, api_secret)

    async def fetch_recent_ohlcv(self, symbol: str, timeframe: str,
                                  limit: int = 200) -> list[dict]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        tf_str = self._TF_MAP.get(timeframe, timeframe)
        # Convert to alpaca TimeFrame
        unit_map = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour,
                    "Day": TimeFrameUnit.Day}
        for suffix, unit in unit_map.items():
            if tf_str.endswith(suffix):
                amount = int(tf_str[:-len(suffix)])
                tf_obj = TimeFrame(amount, unit)
                break
        else:
            raise ValueError(f"unsupported timeframe: {timeframe}")

        end = utc_now()
        # rough lookback: limit * tf_seconds * 2 for safety (markets close)
        lookback_secs = timeframe_to_seconds(timeframe) * limit * 2
        start = end - pd.Timedelta(seconds=lookback_secs)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf_obj,
                                start=start, end=end, limit=limit)
        loop = asyncio.get_event_loop()
        bars = await loop.run_in_executor(None, self.client.get_stock_bars, req)
        data = bars.data.get(symbol, [])
        return [{
            "timestamp": b.timestamp.astimezone(timezone.utc),
            "open": float(b.open), "high": float(b.high), "low": float(b.low),
            "close": float(b.close), "volume": float(b.volume),
        } for b in data]

    async def close(self):
        # alpaca-py client has no explicit close
        return


class OandaDataAdapter:
    """OANDA forex/CFD adapter via oandapyV20 (sync, dispatched to executor)."""

    _TF_MAP = {"1m": "M1", "5m": "M5", "15m": "M15", "1h": "H1", "4h": "H4", "1d": "D"}

    def __init__(self, api_key: str, account_id: str, practice: bool = True):
        from oandapyV20 import API
        env = "practice" if practice else "live"
        self.api = API(access_token=api_key, environment=env)
        self.account_id = account_id

    async def fetch_recent_ohlcv(self, symbol: str, timeframe: str,
                                  limit: int = 200) -> list[dict]:
        from oandapyV20.endpoints.instruments import InstrumentsCandles
        gran = self._TF_MAP.get(timeframe)
        if not gran:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        params = {"granularity": gran, "count": limit, "price": "M"}
        req = InstrumentsCandles(instrument=symbol, params=params)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, self.api.request, req)
        out = []
        for c in resp.get("candles", []):
            if not c.get("complete"):
                continue
            mid = c["mid"]
            out.append({
                "timestamp": pd.to_datetime(c["time"], utc=True).to_pydatetime(),
                "open": float(mid["o"]), "high": float(mid["h"]),
                "low": float(mid["l"]), "close": float(mid["c"]),
                "volume": float(c.get("volume", 0)),
            })
        return out

    async def close(self):
        return


def make_data_adapter(broker: str, cfg: dict):
    """Factory: returns the right data adapter for a broker."""
    b = broker.lower()
    if b == "binance":
        bcfg = cfg["brokers"]["binance"]
        return CCXTAdapter("binance", bcfg.get("api_key", ""),
                           bcfg.get("api_secret", ""), bcfg.get("testnet", True))
    if b in ("bybit", "coinbase"):
        bcfg = cfg["brokers"].get(b, {})
        return CCXTAdapter(b, bcfg.get("api_key", ""), bcfg.get("api_secret", ""))
    if b == "alpaca":
        bcfg = cfg["brokers"]["alpaca"]
        return AlpacaDataAdapter(bcfg["api_key"], bcfg["api_secret"])
    if b == "oanda":
        bcfg = cfg["brokers"]["oanda"]
        return OandaDataAdapter(bcfg["api_key"], bcfg["account_id"],
                                bcfg.get("practice", True))
    raise ValueError(f"unknown broker: {broker}")
