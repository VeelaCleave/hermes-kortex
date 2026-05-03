"""Shared timestamp helpers for KORTEX.

Internal storage uses epoch floats. Human-facing output can still render
 readable UTC strings when needed.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Optional


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

    raw = value.strip()
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


def query_emotion_score(query: str) -> float:
    """Estimate emotional valence of a query string from -1.0 (negative) to +1.0 (positive)."""
    if not query:
        return 0.0
    q = query.lower()
    positive_words = {
        "awesome", "great", "love", "excited", "happy", "amazing", "fantastic",
        "brilliant", "perfect", "wonderful", "nice", "good", "best", "beautiful",
        "shipped", "breakthrough", "finally", "celebrate", "win", "success",
    }
    negative_words = {
        "frustrat", "annoy", "angr", "hate", "suck", "bug", "error", "fail",
        "sigh", "ugh", "meh", "worried", "anxious", "confused", "tired",
        "disappoint", "stuck", "slow", "glitch", "crash", "regret",
    }
    pos_count = neg_count = 0
    for word in q.split():
        if word in positive_words:
            pos_count += 1
        elif word in negative_words:
            neg_count += 1
        elif any(stem in word for stem in negative_words):
            neg_count += 1
        elif any(stem in word for stem in positive_words):
            pos_count += 1
    total = pos_count + neg_count
    return 0.0 if total == 0 else (pos_count - neg_count) / total


def detect_temporal_window_days(query: str) -> Optional[float]:
    """Detect temporal window from query string (e.g. 'last week' -> 7.0)."""
    if not query:
        return None
    lowered = query.lower()
    direct_map = {"today": 0.0, "yesterday": 1.0, "last week": 7.0, "last month": 30.0}
    for phrase, days in direct_map.items():
        if phrase in lowered:
            return days
    if "in march" in lowered:
        return 30.0
    match = re.search(r"(\d+)\s+(day|days|week|weeks|month|months)\s+ago", lowered)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("day"):
        return value
    if unit.startswith("week"):
        return value * 7.0
    return value * 30.0
