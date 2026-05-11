"""Time and timeframe utilities."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import re


_TF_RE = re.compile(r"^(\d+)([smhdw])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def timeframe_to_seconds(tf: str) -> int:
    """'5m' -> 300, '1h' -> 3600. Raises on bad input."""
    m = _TF_RE.match(tf.strip().lower())
    if not m:
        raise ValueError(f"bad timeframe: {tf!r}")
    n, unit = int(m.group(1)), m.group(2)
    return n * _UNIT_SECONDS[unit]


def floor_to_timeframe(ts: datetime, tf: str) -> datetime:
    """Floor a timestamp to its bar boundary."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    secs = timeframe_to_seconds(tf)
    epoch = int(ts.timestamp())
    floored = (epoch // secs) * secs
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def next_bar_close(ts: datetime, tf: str) -> datetime:
    return floor_to_timeframe(ts, tf) + timedelta(seconds=timeframe_to_seconds(tf))
