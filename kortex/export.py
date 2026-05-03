"""Export and import helpers for KORTEX backups."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from .db import DEFAULT_USER_ID, SCHEMA_VERSION, KortexDB
from .models import Episode, Fact, OpenLoop, Reflection
from .time_utils import epoch_to_iso, now_epoch, parse_timestamp


def _iso_or_none(value):
    return epoch_to_iso(value) if value else None

_DEFAULT_MEMORY_TYPES = {
    "episodes",
    "facts",
    "open_loops",
    "reflections",
    "conversation_summaries",
    "identity_deltas",
}


def export_to_json(
    db: KortexDB,
    *,
    user_id: str = DEFAULT_USER_ID,
    start: Any = None,
    end: Any = None,
    memory_types: Optional[Iterable[str]] = None,
) -> str:
    type_filter = set(memory_types or _DEFAULT_MEMORY_TYPES)
    start_ts = parse_timestamp(start)
    end_ts = parse_timestamp(end)

    payload: Dict[str, Any] = {
        "metadata": {
            "kortex_schema_version": SCHEMA_VERSION,
            "db_user_version": _db_user_version(db),
            "exported_at": _iso_or_none(now_epoch()),
            "user_id": user_id,
            "filters": {
                "start": _iso_or_none(start_ts) if start_ts is not None else None,
                "end": _iso_or_none(end_ts) if end_ts is not None else None,
                "types": sorted(type_filter),
            },
        }
    }

    if "episodes" in type_filter:
        payload["episodes"] = [
            _episode_to_dict(ep)
            for ep in db.get_recent_episodes(
                limit=10000, include_consolidated=True, user_id=user_id
            )
            if _within_range(ep.timestamp, start_ts, end_ts)
        ]
    if "facts" in type_filter:
        payload["facts"] = [
            _fact_to_dict(fact)
            for fact in db.get_active_facts(limit=10000, user_id=user_id)
            if _within_range(fact.first_seen, start_ts, end_ts)
        ]
    if "open_loops" in type_filter:
        payload["open_loops"] = [
            _loop_to_dict(loop)
            for loop in db.get_open_loops(limit=10000, user_id=user_id)
            if _within_range(loop.created_at, start_ts, end_ts)
        ]
    if "reflections" in type_filter:
        payload["reflections"] = [
            _reflection_to_dict(reflection)
            for reflection in db.get_reflections(limit=10000, user_id=user_id)
            if _within_range(reflection.created_at, start_ts, end_ts)
        ]
    if "conversation_summaries" in type_filter:
        payload["conversation_summaries"] = [
            _summary_to_dict(summary)
            for summary in db.list_conversation_summaries(limit=10000, user_id=user_id)
            if _within_range(summary.get("updated_at"), start_ts, end_ts)
        ]
    if "identity_deltas" in type_filter:
        payload["identity_deltas"] = [
            _identity_delta_to_dict(delta)
            for delta in db.get_identity_deltas(limit=10000)
            if delta.user_id == user_id
            and _within_range(delta.created_at, start_ts, end_ts)
        ]

    return json.dumps(payload)


def import_from_json(
    db: KortexDB, json_payload: str, *, allow_override: bool = False
) -> Dict[str, Any]:
    parsed = json.loads(json_payload)
    metadata = parsed.get("metadata", {})
    payload_version = metadata.get("kortex_schema_version")
    if payload_version != SCHEMA_VERSION and not allow_override:
        return {
            "ok": False,
            "error": (
                f"Schema version mismatch: payload={payload_version}, expected={SCHEMA_VERSION}"
            ),
        }

    imported = {
        "episodes": 0,
        "facts": 0,
        "open_loops": 0,
        "reflections": 0,
        "conversation_summaries": 0,
        "identity_deltas": 0,
    }

    for item in parsed.get("episodes", []):
        episode = Episode(
            user_id=item.get("user_id", DEFAULT_USER_ID),
            session_id=item.get("session_id", ""),
            turn_index=item.get("turn_index", 0),
            timestamp=parse_timestamp(item.get("timestamp")) or now_epoch(),
            user_text=item.get("user_text", ""),
            assistant_text=item.get("assistant_text", ""),
            summary=item.get("summary", ""),
            salience=item.get("salience", 0.0),
            valence=item.get("valence", 0),
            arousal=item.get("arousal", 0.0),
            topics=item.get("topics", ""),
            entities=item.get("entities", ""),
            is_consolidated=bool(item.get("is_consolidated", False)),
            last_accessed_at=parse_timestamp(item.get("last_accessed_at")),
            retrieval_count=item.get("retrieval_count", 0),
            consolidated_into=item.get("consolidated_into"),
            raw_preserved=bool(item.get("raw_preserved", True)),
        )
        db.insert_episode(episode)
        imported["episodes"] += 1

    for item in parsed.get("facts", []):
        fact = Fact(
            user_id=item.get("user_id", DEFAULT_USER_ID),
            subject_type=item.get("subject_type", "user"),
            subject_id=item.get("subject_id", ""),
            predicate=item.get("predicate", ""),
            object_text=item.get("object_text", ""),
            confidence=item.get("confidence", 0.5),
            source_episode_id=item.get("source_episode_id"),
            first_seen=parse_timestamp(item.get("first_seen")) or now_epoch(),
            last_seen=parse_timestamp(item.get("last_seen")) or now_epoch(),
            status=item.get("status", "active"),
            superseded_by=item.get("superseded_by"),
            last_accessed_at=parse_timestamp(item.get("last_accessed_at")),
            retrieval_count=item.get("retrieval_count", 0),
            valid_from=parse_timestamp(item.get("valid_from")),
            valid_to=parse_timestamp(item.get("valid_to")),
            contradiction_status=item.get("contradiction_status", "active"),
        )
        db.insert_fact(fact)
        imported["facts"] += 1

    for item in parsed.get("open_loops", []):
        loop = OpenLoop(
            user_id=item.get("user_id", DEFAULT_USER_ID),
            kind=item.get("kind", "commitment"),
            text=item.get("text", ""),
            due_hint=item.get("due_hint", ""),
            status=item.get("status", "open"),
            source_episode_id=item.get("source_episode_id"),
            created_at=parse_timestamp(item.get("created_at")) or now_epoch(),
            resolved_at=parse_timestamp(item.get("resolved_at")),
            last_accessed_at=parse_timestamp(item.get("last_accessed_at")),
            resolution=item.get("resolution", ""),
            resolved_by_episode_id=item.get("resolved_by_episode_id"),
        )
        db.insert_open_loop(loop)
        if loop.status == "resolved":
            db.resolve_loop(
                loop.id,
                resolution=loop.resolution,
                resolved_by_episode_id=loop.resolved_by_episode_id,
            )
        imported["open_loops"] += 1

    for item in parsed.get("reflections", []):
        reflection = Reflection(
            user_id=item.get("user_id", DEFAULT_USER_ID),
            kind=item.get("kind", "pattern"),
            text=item.get("text", ""),
            confidence=item.get("confidence", 0.3),
            source_episode_id=item.get("source_episode_id"),
            created_at=parse_timestamp(item.get("created_at")) or now_epoch(),
            last_reinforced=parse_timestamp(item.get("last_reinforced")) or now_epoch(),
            reinforcement_count=item.get("reinforcement_count", 1),
            last_accessed_at=parse_timestamp(item.get("last_accessed_at")),
            retrieval_count=item.get("retrieval_count", 0),
            promotion_status=item.get("promotion_status", "active"),
            promoted_at=parse_timestamp(item.get("promoted_at")),
        )
        db.insert_reflection(reflection)
        imported["reflections"] += 1

    for item in parsed.get("conversation_summaries", []):
        db.insert_conversation_summary(
            {
                "user_id": item.get("user_id", DEFAULT_USER_ID),
                "session_id": item.get("session_id", ""),
                "summary_text": item.get("summary_text", ""),
                "summary_level": item.get("summary_level", "conversation"),
                "episode_range_start": parse_timestamp(item.get("episode_range_start")),
                "episode_range_end": parse_timestamp(item.get("episode_range_end")),
                "episode_count": item.get("episode_count", 0),
                "key_entities": item.get("key_entities", ""),
                "created_at": parse_timestamp(item.get("created_at")) or now_epoch(),
                "updated_at": parse_timestamp(item.get("updated_at")) or now_epoch(),
            }
        )
        imported["conversation_summaries"] += 1

    for item in parsed.get("identity_deltas", []):
        from .models import IdentityDelta

        delta = IdentityDelta(
            user_id=item.get("user_id", DEFAULT_USER_ID),
            text=item.get("text", ""),
            confidence=item.get("confidence", 0.3),
            source_episode_id=item.get("source_episode_id"),
            created_at=parse_timestamp(item.get("created_at")) or now_epoch(),
            applied=bool(item.get("applied", False)),
        )
        db.insert_identity_delta(delta)
        if delta.applied:
            db.mark_identity_delta_applied(delta.id)
        imported["identity_deltas"] += 1

    return {"ok": True, "imported": imported}


def _db_user_version(db: KortexDB) -> int:
    return int(db._get_conn().execute("PRAGMA user_version").fetchone()[0])


def _within_range(value: Any, start: Optional[float], end: Optional[float]) -> bool:
    ts = parse_timestamp(value)
    if ts is None:
        return False
    if start is not None and ts < start:
        return False
    if end is not None and ts > end:
        return False
    return True


def _episode_to_dict(ep: Episode) -> Dict[str, Any]:
    return {
        "id": ep.id,
        "user_id": ep.user_id,
        "session_id": ep.session_id,
        "turn_index": ep.turn_index,
        "timestamp": ep.timestamp_iso,
        "user_text": ep.user_text,
        "assistant_text": ep.assistant_text,
        "summary": ep.summary,
        "salience": ep.salience,
        "valence": ep.valence,
        "arousal": ep.arousal,
        "topics": ep.topics,
        "entities": ep.entities,
        "is_consolidated": ep.is_consolidated,
        "last_accessed_at": _iso_or_none(ep.last_accessed_at)
        if ep.last_accessed_at
        else None,
        "retrieval_count": ep.retrieval_count,
        "consolidated_into": ep.consolidated_into,
        "raw_preserved": ep.raw_preserved,
    }


def _fact_to_dict(fact: Fact) -> Dict[str, Any]:
    return {
        "id": fact.id,
        "user_id": fact.user_id,
        "subject_type": fact.subject_type,
        "subject_id": fact.subject_id,
        "predicate": fact.predicate,
        "object_text": fact.object_text,
        "confidence": fact.confidence,
        "source_episode_id": fact.source_episode_id,
        "first_seen": _iso_or_none(fact.first_seen),
        "last_seen": _iso_or_none(fact.last_seen),
        "status": fact.status,
        "superseded_by": fact.superseded_by,
        "last_accessed_at": _iso_or_none(fact.last_accessed_at)
        if fact.last_accessed_at
        else None,
        "retrieval_count": fact.retrieval_count,
        "valid_from": _iso_or_none(fact.valid_from) if fact.valid_from else None,
        "valid_to": _iso_or_none(fact.valid_to) if fact.valid_to else None,
        "contradiction_status": fact.contradiction_status,
    }


def _loop_to_dict(loop: OpenLoop) -> Dict[str, Any]:
    return {
        "id": loop.id,
        "user_id": loop.user_id,
        "kind": loop.kind,
        "text": loop.text,
        "due_hint": loop.due_hint,
        "status": loop.status,
        "source_episode_id": loop.source_episode_id,
        "created_at": _iso_or_none(loop.created_at),
        "resolved_at": _iso_or_none(loop.resolved_at) if loop.resolved_at else None,
        "last_accessed_at": _iso_or_none(loop.last_accessed_at)
        if loop.last_accessed_at
        else None,
        "resolution": loop.resolution,
        "resolved_by_episode_id": loop.resolved_by_episode_id,
    }


def _reflection_to_dict(reflection: Reflection) -> Dict[str, Any]:
    return {
        "id": reflection.id,
        "user_id": reflection.user_id,
        "kind": reflection.kind,
        "text": reflection.text,
        "confidence": reflection.confidence,
        "source_episode_id": reflection.source_episode_id,
        "created_at": _iso_or_none(reflection.created_at),
        "last_reinforced": _iso_or_none(reflection.last_reinforced),
        "reinforcement_count": reflection.reinforcement_count,
        "last_accessed_at": _iso_or_none(reflection.last_accessed_at)
        if reflection.last_accessed_at
        else None,
        "retrieval_count": reflection.retrieval_count,
        "promotion_status": reflection.promotion_status,
        "promoted_at": _iso_or_none(reflection.promoted_at)
        if reflection.promoted_at
        else None,
    }


def _summary_to_dict(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": summary.get("id"),
        "user_id": summary.get("user_id", DEFAULT_USER_ID),
        "session_id": summary.get("session_id", ""),
        "summary_text": summary.get("summary_text", ""),
        "summary_level": summary.get("summary_level", "conversation"),
        "episode_range_start": _iso_or_none(summary["episode_range_start"])
        if summary.get("episode_range_start")
        else None,
        "episode_range_end": _iso_or_none(summary["episode_range_end"])
        if summary.get("episode_range_end")
        else None,
        "episode_count": summary.get("episode_count", 0),
        "key_entities": summary.get("key_entities", ""),
        "created_at": _iso_or_none(summary["created_at"])
        if summary.get("created_at")
        else None,
        "updated_at": _iso_or_none(summary["updated_at"])
        if summary.get("updated_at")
        else None,
    }


def _identity_delta_to_dict(delta: Any) -> Dict[str, Any]:
    return {
        "id": delta.id,
        "user_id": delta.user_id,
        "text": delta.text,
        "confidence": delta.confidence,
        "source_episode_id": delta.source_episode_id,
        "created_at": _iso_or_none(delta.created_at),
        "applied": delta.applied,
    }
