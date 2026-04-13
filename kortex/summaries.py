"""Conversation-level summary generation for KORTEX."""

from __future__ import annotations

from typing import Any, Dict, List

from .time_utils import now_epoch


def build_conversation_summary(
    session_id: str,
    episodes: List[Any],
    messages: List[Dict[str, Any]] | None = None,
    user_id: str = "__default__",
) -> Dict[str, Any] | None:
    """Build a lightweight whole-conversation summary.

    Uses existing episode summaries first, with message fallback when needed.
    """
    messages = messages or []
    if not episodes and not messages:
        return None

    episode_summaries = [
        ep.summary.strip() for ep in episodes if getattr(ep, "summary", "").strip()
    ]
    if not episode_summaries:
        episode_summaries = [
            _clip(str(msg.get("content", "")).strip(), 140)
            for msg in messages
            if str(msg.get("content", "")).strip()
            and msg.get("role") in {"user", "assistant"}
        ]

    episode_summaries = episode_summaries[:3]
    if not episode_summaries:
        return None

    all_entities: list[str] = []
    seen_entities: set[str] = set()
    for ep in episodes:
        raw_entities = getattr(ep, "entities", "") or ""
        for entity in [
            part.strip() for part in raw_entities.split(",") if part.strip()
        ]:
            key = entity.lower()
            if key not in seen_entities:
                seen_entities.add(key)
                all_entities.append(entity)
            if len(all_entities) >= 6:
                break
        if len(all_entities) >= 6:
            break

    summary_text = "Conversation covered: " + " | ".join(
        _clip(text, 180) for text in episode_summaries
    )
    created_at = now_epoch()

    return {
        "user_id": user_id,
        "session_id": session_id,
        "summary_text": summary_text,
        "summary_level": "conversation",
        "episode_range_start": min((ep.timestamp for ep in episodes), default=None),
        "episode_range_end": max((ep.timestamp for ep in episodes), default=None),
        "episode_count": len(episodes),
        "key_entities": ",".join(all_entities),
        "created_at": created_at,
        "updated_at": created_at,
    }


def _clip(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."
