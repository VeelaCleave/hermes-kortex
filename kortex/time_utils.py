"""Shared timestamp helpers for KORTEX.

Internal storage uses epoch floats. Human-facing output can still render
 readable UTC strings when needed.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


def now_epoch() -> float:
    """Current UTC time as an epoch float."""
    return time.time()


def parse_timestamp(value: float | int | str | datetime | None) -> float | None:
    """Normalize multiple timestamp representations to epoch float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()

    raw = str(value).strip()
    if not raw:
        return None

    try:
        return float(raw)
    except ValueError:
        pass

    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def epoch_to_datetime(ts: float) -> datetime:
    """Convert epoch float to UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def epoch_to_iso(ts: float) -> str:
    """Convert epoch float to ISO8601 UTC string."""
    return epoch_to_datetime(ts).isoformat()


def epoch_to_display(ts: float) -> str:
    """Human-readable UTC timestamp for tools and memory rendering."""
    return epoch_to_datetime(ts).strftime("%a %b %d, %H:%M UTC")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value to a range."""
    return max(low, min(high, value))


def _ema(current: float, target: float, alpha: float) -> float:
    """Exponential moving average step."""
    return (1 - alpha) * current + alpha * target
