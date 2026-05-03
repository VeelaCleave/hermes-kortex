"""Data models for KORTEX memory system.

All internal data structures as plain dataclasses. No ORM, no magic —
just typed containers that map 1:1 to SQLite rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .time_utils import epoch_to_display, epoch_to_iso, now_epoch, parse_timestamp


class Valence(Enum):
    """Emotional valence of an episode."""

    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class Episode:
    """A single conversational turn with extracted metadata."""

    id: Optional[int] = None
    user_id: str = "__default__"
    session_id: str = ""
    turn_index: int = 0
    timestamp: float = field(default_factory=now_epoch)
    user_text: str = ""
    assistant_text: str = ""
    summary: str = ""
    salience: float = 0.0  # 0.0 = mundane, 1.0 = unforgettable
    valence: int = 0  # -2 to +2 (Valence enum value)
    arousal: float = 0.0  # 0.0 = calm, 1.0 = intense
    topics: str = ""  # comma-separated topic tags
    entities: str = ""  # comma-separated entity names
    is_consolidated: bool = False
    last_accessed_at: Optional[float] = None
    retrieval_count: int = 0
    consolidated_into: Optional[int] = None
    raw_preserved: bool = True

    @property
    def emotional_weight(self) -> float:
        """Combined emotional significance for ranking."""
        return abs(self.valence) * 0.5 + self.arousal * 0.5

    @property
    def timestamp_iso(self) -> str:
        return epoch_to_iso(self.timestamp)

    def to_recall_text(self, now: Optional[float] = None) -> str:
        """Format this episode for injection into context."""
        now = now or now_epoch()
        delta_seconds = max(now - self.timestamp, 0.0)
        days = int(delta_seconds // 86400)

        if days == 0:
            time_anchor = "earlier today"
        elif days == 1:
            time_anchor = "yesterday"
        elif days < 7:
            time_anchor = f"{days} days ago"
        elif days < 30:
            weeks = days // 7
            time_anchor = f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            # Include the actual date for older memories
            time_anchor = f"{days // 7} weeks ago ({epoch_to_display(self.timestamp)})"

        valence_label = ""
        if self.valence <= -2:
            valence_label = " [tense/hostile]"
        elif self.valence == -1:
            valence_label = " [frustrated]"
        elif self.valence == 1:
            valence_label = " [warm]"
        elif self.valence >= 2:
            valence_label = " [very positive]"

        text = f"[{time_anchor}{valence_label}] {self.summary}"
        return text

    @classmethod
    def from_db_row(cls, row: dict) -> "Episode":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            timestamp=parse_timestamp(row["timestamp"]) or 0.0,
            user_text=row["user_text"],
            assistant_text=row["assistant_text"],
            summary=row["summary"],
            salience=row["salience"],
            valence=row["valence"],
            arousal=row["arousal"],
            topics=row["topics"],
            entities=row["entities"],
            is_consolidated=bool(row["is_consolidated"]),
            last_accessed_at=parse_timestamp(row["last_accessed_at"]),
            retrieval_count=row["retrieval_count"],
            consolidated_into=row["consolidated_into"],
            raw_preserved=bool(row["raw_preserved"]),
        )


@dataclass
class Fact:
    """A durable fact about the user, agent, or a project."""

    id: Optional[int] = None
    user_id: str = "__default__"
    subject_type: str = "user"  # user, agent, project, general
    subject_id: str = ""
    predicate: str = ""  # e.g. "prefers", "works_on", "dislikes"
    object_text: str = ""  # the actual fact content
    confidence: float = 0.5  # 0.0 = uncertain, 1.0 = rock solid
    source_episode_id: Optional[int] = None
    first_seen: float = field(default_factory=now_epoch)
    last_seen: float = field(default_factory=now_epoch)
    status: str = "active"  # active, superseded, retracted
    superseded_by: Optional[int] = None
    last_accessed_at: Optional[float] = None
    retrieval_count: int = 0
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    contradiction_status: str = "active"

    @classmethod
    def from_db_row(cls, row: dict) -> "Fact":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            subject_type=row["subject_type"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_text=row["object_text"],
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
            first_seen=parse_timestamp(row["first_seen"]) or 0.0,
            last_seen=parse_timestamp(row["last_seen"]) or 0.0,
            status=row["status"],
            superseded_by=row["superseded_by"],
            last_accessed_at=parse_timestamp(row["last_accessed_at"]),
            retrieval_count=row["retrieval_count"],
            valid_from=parse_timestamp(row["valid_from"]),
            valid_to=parse_timestamp(row["valid_to"]),
            contradiction_status=row["contradiction_status"],
        )


@dataclass
class OpenLoop:
    """A commitment, unresolved question, or pending task."""

    id: Optional[int] = None
    user_id: str = "__default__"
    kind: str = "commitment"  # commitment, task, question, tension
    text: str = ""
    due_hint: str = ""  # optional date hint
    status: str = "open"  # open, resolved, expired, cancelled
    source_episode_id: Optional[int] = None
    created_at: float = field(default_factory=now_epoch)
    resolved_at: Optional[float] = None
    last_accessed_at: Optional[float] = None
    resolution: str = ""
    resolved_by_episode_id: Optional[int] = None

    @classmethod
    def from_db_row(cls, row: dict) -> "OpenLoop":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            kind=row["kind"],
            text=row["text"],
            due_hint=row["due_hint"],
            status=row["status"],
            source_episode_id=row["source_episode_id"],
            created_at=parse_timestamp(row["created_at"]) or 0.0,
            resolved_at=parse_timestamp(row["resolved_at"]),
            last_accessed_at=parse_timestamp(row["last_accessed_at"]),
            resolution=row["resolution"],
            resolved_by_episode_id=row["resolved_by_episode_id"],
        )


@dataclass
class Reflection:
    """A learned pattern, mistake, or style preference."""

    id: Optional[int] = None
    user_id: str = "__default__"
    kind: str = "pattern"  # mistake, pattern, preference, style
    text: str = ""
    confidence: float = 0.3
    source_episode_id: Optional[int] = None
    created_at: float = field(default_factory=now_epoch)
    last_reinforced: float = field(default_factory=now_epoch)
    reinforcement_count: int = 1
    last_accessed_at: Optional[float] = None
    retrieval_count: int = 0
    promotion_status: str = "active"
    promoted_at: Optional[float] = None

    @classmethod
    def from_db_row(cls, row: dict) -> "Reflection":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            kind=row["kind"],
            text=row["text"],
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
            created_at=parse_timestamp(row["created_at"]) or 0.0,
            last_reinforced=parse_timestamp(row["last_reinforced"]) or 0.0,
            reinforcement_count=row["reinforcement_count"],
            last_accessed_at=parse_timestamp(row["last_accessed_at"]),
            retrieval_count=row["retrieval_count"],
            promotion_status=row["promotion_status"],
            promoted_at=parse_timestamp(row["promoted_at"]),
        )


@dataclass
class RelationshipState:
    """Current relationship dynamics with a user."""

    id: Optional[int] = None
    user_id: str = "__default__"
    warmth: float = 0.5  # 0=cold, 1=warm
    trust: float = 0.5  # 0=distrustful, 1=high trust
    tension: float = 0.0  # 0=none, 1=high tension
    familiarity: float = 0.0  # 0=stranger, 1=deeply familiar
    humor: float = 0.0  # 0=formal, 1=very playful
    formality: float = 0.5  # 0=casual, 1=formal
    volatility: float = 0.0  # 0=stable, 1=unpredictable
    last_updated: float = field(default_factory=now_epoch)
    total_turns: int = 0

    def to_compact_text(self) -> str:
        """Format relationship state for context injection."""
        descriptors = []
        if self.warmth > 0.7:
            descriptors.append("warm rapport")
        elif self.warmth < 0.3:
            descriptors.append("cool/distant")
        if self.trust > 0.7:
            descriptors.append("high trust")
        elif self.trust < 0.3:
            descriptors.append("guarded trust")
        if self.tension > 0.5:
            descriptors.append("ongoing tension")
        if self.humor > 0.6:
            descriptors.append("playful dynamic")
        if self.familiarity > 0.7:
            descriptors.append("deeply familiar")
        elif self.familiarity < 0.2:
            descriptors.append("still getting to know each other")

        if not descriptors:
            descriptors.append("neutral")

        return (
            f"Relationship: {', '.join(descriptors)} ({self.total_turns} interactions)"
        )

    @classmethod
    def from_db_row(cls, row: dict) -> "RelationshipState":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            warmth=row["warmth"],
            trust=row["trust"],
            tension=row["tension"],
            familiarity=row["familiarity"],
            humor=row["humor"],
            formality=row["formality"],
            volatility=row["volatility"],
            last_updated=parse_timestamp(row["last_updated"]) or 0.0,
            total_turns=row["total_turns"],
        )


@dataclass
class AffectSignal:
    """Per-turn emotional affect detection result."""

    frustration: float = 0.0
    warmth: float = 0.0
    humor: float = 0.0
    hostility: float = 0.0
    gratitude: float = 0.0
    anxiety: float = 0.0
    excitement: float = 0.0
    trust_signal: float = 0.0
    valence: float = 0.0  # -1.0 (negative) to +1.0 (positive)
    arousal: float = 0.0  # 0.0 (calm) to 1.0 (intense)
    dominant_emotion: str = "neutral"
    is_sarcastic: bool = False

    @property
    def is_significant(self) -> bool:
        """Whether this signal is worth storing (any dimension > threshold)."""
        return self.arousal > 0.2 or abs(self.valence) > 0.2

    def to_compact_text(self) -> str:
        """Brief human-readable summary for context injection."""
        if not self.is_significant:
            return ""
        parts = []
        if self.dominant_emotion != "neutral":
            parts.append(self.dominant_emotion)
        if self.is_sarcastic:
            parts.append("sarcastic")
        intensity = "mild" if self.arousal < 0.5 else "strong"
        return f"[{intensity} {', '.join(parts)}]" if parts else ""

    def to_db_row(self) -> dict:
        return {
            "frustration": self.frustration,
            "warmth": self.warmth,
            "humor": self.humor,
            "hostility": self.hostility,
            "gratitude": self.gratitude,
            "anxiety": self.anxiety,
            "excitement": self.excitement,
            "trust_signal": self.trust_signal,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominant_emotion": self.dominant_emotion,
            "is_sarcastic": int(self.is_sarcastic),
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "AffectSignal":
        return cls(
            frustration=row["frustration"],
            warmth=row["warmth"],
            humor=row["humor"],
            hostility=row["hostility"],
            gratitude=row["gratitude"],
            anxiety=row["anxiety"],
            excitement=row["excitement"],
            trust_signal=row["trust_signal"],
            valence=row["valence"],
            arousal=row["arousal"],
            dominant_emotion=row["dominant_emotion"],
            is_sarcastic=bool(row["is_sarcastic"]),
        )


@dataclass
class IdentityDelta:
    """A proposed change to the agent's self-model / SOUL.md."""

    id: Optional[int] = None
    user_id: str = "__default__"
    text: str = ""
    confidence: float = 0.3
    source_episode_id: Optional[int] = None
    created_at: float = field(default_factory=now_epoch)
    applied: bool = False

    @classmethod
    def from_db_row(cls, row: dict) -> "IdentityDelta":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            text=row["text"],
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
            created_at=parse_timestamp(row["created_at"]) or 0.0,
            applied=bool(row["applied"]),
        )


@dataclass
class EntityLink:
    """A typed edge between any two objects in the memory graph."""

    id: Optional[int] = None
    user_id: str = "__default__"
    src_type: str = ""  # episode, fact, entity, reflection
    src_id: int = 0
    dst_type: str = ""
    dst_id: int = 0
    relation: str = ""  # mentions, contradicts, follows_up, caused_by
    weight: float = 1.0
