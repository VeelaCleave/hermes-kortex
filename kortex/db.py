"""SQLite storage layer for KORTEX.

Single-file database with FTS5 for full-text search. All tables created
on first initialize(). Schema versioned via user_version pragma.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import AffectSignal, Episode, Fact, OpenLoop, Reflection, RelationshipState

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    turn_index      INTEGER NOT NULL DEFAULT 0,
    timestamp       TEXT NOT NULL,
    user_text       TEXT NOT NULL DEFAULT '',
    assistant_text  TEXT NOT NULL DEFAULT '',
    summary         TEXT NOT NULL DEFAULT '',
    salience        REAL NOT NULL DEFAULT 0.0,
    valence         INTEGER NOT NULL DEFAULT 0,
    arousal         REAL NOT NULL DEFAULT 0.0,
    topics          TEXT NOT NULL DEFAULT '',
    entities        TEXT NOT NULL DEFAULT '',
    is_consolidated INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_salience ON episodes(salience DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    summary, user_text, assistant_text, topics, entities,
    content=episodes,
    content_rowid=id,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, summary, user_text, assistant_text, topics, entities)
    VALUES (new.id, new.summary, new.user_text, new.assistant_text, new.topics, new.entities);
END;

CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, summary, user_text, assistant_text, topics, entities)
    VALUES ('delete', old.id, old.summary, old.user_text, old.assistant_text, old.topics, old.entities);
END;

CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, summary, user_text, assistant_text, topics, entities)
    VALUES ('delete', old.id, old.summary, old.user_text, old.assistant_text, old.topics, old.entities);
    INSERT INTO episodes_fts(rowid, summary, user_text, assistant_text, topics, entities)
    VALUES (new.id, new.summary, new.user_text, new.assistant_text, new.topics, new.entities);
END;

CREATE TABLE IF NOT EXISTS facts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_type        TEXT NOT NULL DEFAULT 'user',
    subject_id          TEXT NOT NULL DEFAULT '',
    predicate           TEXT NOT NULL DEFAULT '',
    object_text         TEXT NOT NULL DEFAULT '',
    confidence          REAL NOT NULL DEFAULT 0.5,
    source_episode_id   INTEGER REFERENCES episodes(id),
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
    superseded_by       INTEGER REFERENCES facts(id)
);

CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_type, subject_id);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    object_text, predicate,
    content=facts,
    content_rowid=id,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, object_text, predicate)
    VALUES (new.id, new.object_text, new.predicate);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, object_text, predicate)
    VALUES ('delete', old.id, old.object_text, old.predicate);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, object_text, predicate)
    VALUES ('delete', old.id, old.object_text, old.predicate);
    INSERT INTO facts_fts(rowid, object_text, predicate)
    VALUES (new.id, new.object_text, new.predicate);
END;

CREATE TABLE IF NOT EXISTS open_loops (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    kind                TEXT NOT NULL DEFAULT 'commitment',
    text                TEXT NOT NULL DEFAULT '',
    due_hint            TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'open',
    source_episode_id   INTEGER REFERENCES episodes(id),
    created_at          TEXT NOT NULL,
    resolved_at         TEXT
);

CREATE INDEX IF NOT EXISTS idx_loops_status ON open_loops(status);

CREATE TABLE IF NOT EXISTS reflections (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    kind                    TEXT NOT NULL DEFAULT 'pattern',
    text                    TEXT NOT NULL DEFAULT '',
    confidence              REAL NOT NULL DEFAULT 0.3,
    source_episode_id       INTEGER REFERENCES episodes(id),
    created_at              TEXT NOT NULL,
    last_reinforced         TEXT NOT NULL,
    reinforcement_count     INTEGER NOT NULL DEFAULT 1
);

CREATE VIRTUAL TABLE IF NOT EXISTS reflections_fts USING fts5(
    text,
    content=reflections,
    content_rowid=id,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS reflections_ai AFTER INSERT ON reflections BEGIN
    INSERT INTO reflections_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS reflections_ad AFTER DELETE ON reflections BEGIN
    INSERT INTO reflections_fts(reflections_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS reflections_au AFTER UPDATE ON reflections BEGIN
    INSERT INTO reflections_fts(reflections_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO reflections_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS relationship_state (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL DEFAULT 'default' UNIQUE,
    warmth          REAL NOT NULL DEFAULT 0.5,
    trust           REAL NOT NULL DEFAULT 0.5,
    tension         REAL NOT NULL DEFAULT 0.0,
    familiarity     REAL NOT NULL DEFAULT 0.0,
    humor           REAL NOT NULL DEFAULT 0.0,
    formality       REAL NOT NULL DEFAULT 0.5,
    volatility      REAL NOT NULL DEFAULT 0.0,
    last_updated    TEXT NOT NULL,
    total_turns     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS identity_deltas (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    text                TEXT NOT NULL DEFAULT '',
    confidence          REAL NOT NULL DEFAULT 0.3,
    source_episode_id   INTEGER REFERENCES episodes(id),
    created_at          TEXT NOT NULL,
    applied             INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entity_links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src_type    TEXT NOT NULL,
    src_id      INTEGER NOT NULL,
    dst_type    TEXT NOT NULL,
    dst_id      INTEGER NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_links_src ON entity_links(src_type, src_id);
CREATE INDEX IF NOT EXISTS idx_links_dst ON entity_links(dst_type, dst_id);

CREATE TABLE IF NOT EXISTS emotion_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id          INTEGER REFERENCES episodes(id),
    session_id          TEXT NOT NULL DEFAULT '',
    timestamp           TEXT NOT NULL,
    frustration         REAL NOT NULL DEFAULT 0.0,
    warmth              REAL NOT NULL DEFAULT 0.0,
    humor               REAL NOT NULL DEFAULT 0.0,
    hostility           REAL NOT NULL DEFAULT 0.0,
    gratitude           REAL NOT NULL DEFAULT 0.0,
    anxiety             REAL NOT NULL DEFAULT 0.0,
    excitement          REAL NOT NULL DEFAULT 0.0,
    trust_signal        REAL NOT NULL DEFAULT 0.0,
    valence             REAL NOT NULL DEFAULT 0.0,
    arousal             REAL NOT NULL DEFAULT 0.0,
    dominant_emotion    TEXT NOT NULL DEFAULT 'neutral',
    is_sarcastic        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_emotion_log_episode ON emotion_log(episode_id);
CREATE INDEX IF NOT EXISTS idx_emotion_log_ts ON emotion_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_emotion_log_session ON emotion_log(session_id);
"""


class KortexDB:
    """Thread-safe SQLite storage for KORTEX."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _tx(self):
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        conn = self._get_conn()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            conn.executescript(_SCHEMA_SQL)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            logger.info(
                "KORTEX DB initialized (v%d) at %s", SCHEMA_VERSION, self._db_path
            )
        elif version < SCHEMA_VERSION:
            self._migrate(conn, version)

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        if from_version < 2:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS emotion_log (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id          INTEGER REFERENCES episodes(id),
                    session_id          TEXT NOT NULL DEFAULT '',
                    timestamp           TEXT NOT NULL,
                    frustration         REAL NOT NULL DEFAULT 0.0,
                    warmth              REAL NOT NULL DEFAULT 0.0,
                    humor               REAL NOT NULL DEFAULT 0.0,
                    hostility           REAL NOT NULL DEFAULT 0.0,
                    gratitude           REAL NOT NULL DEFAULT 0.0,
                    anxiety             REAL NOT NULL DEFAULT 0.0,
                    excitement          REAL NOT NULL DEFAULT 0.0,
                    trust_signal        REAL NOT NULL DEFAULT 0.0,
                    valence             REAL NOT NULL DEFAULT 0.0,
                    arousal             REAL NOT NULL DEFAULT 0.0,
                    dominant_emotion    TEXT NOT NULL DEFAULT 'neutral',
                    is_sarcastic        INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_emotion_log_episode ON emotion_log(episode_id);
                CREATE INDEX IF NOT EXISTS idx_emotion_log_ts ON emotion_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_emotion_log_session ON emotion_log(session_id);
            """)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        logger.info("KORTEX DB migrated to v%d at %s", SCHEMA_VERSION, self._db_path)

    # -- Episodes ------------------------------------------------------------

    def insert_episode(self, ep: Episode) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO episodes
                   (session_id, turn_index, timestamp, user_text, assistant_text,
                    summary, salience, valence, arousal, topics, entities, is_consolidated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ep.session_id,
                    ep.turn_index,
                    ep.timestamp.isoformat(),
                    ep.user_text,
                    ep.assistant_text,
                    ep.summary,
                    ep.salience,
                    ep.valence,
                    ep.arousal,
                    ep.topics,
                    ep.entities,
                    int(ep.is_consolidated),
                ),
            )
            ep.id = cur.lastrowid
            return ep.id

    def update_episode(self, ep: Episode) -> None:
        with self._tx() as conn:
            conn.execute(
                """UPDATE episodes SET summary=?, salience=?, valence=?, arousal=?,
                   topics=?, entities=?, is_consolidated=? WHERE id=?""",
                (
                    ep.summary,
                    ep.salience,
                    ep.valence,
                    ep.arousal,
                    ep.topics,
                    ep.entities,
                    int(ep.is_consolidated),
                    ep.id,
                ),
            )

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM episodes WHERE id=?", (episode_id,))
            .fetchone()
        )
        return self._row_to_episode(row) if row else None

    def get_recent_episodes(
        self, limit: int = 10, session_id: Optional[str] = None
    ) -> List[Episode]:
        if session_id:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM episodes WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
                )
                .fetchall()
            )
        return [self._row_to_episode(r) for r in rows]

    def search_episodes(self, query: str, limit: int = 10) -> List[Episode]:
        rows = (
            self._get_conn()
            .execute(
                """SELECT e.* FROM episodes e
               JOIN episodes_fts f ON e.id = f.rowid
               WHERE episodes_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
                (query, limit),
            )
            .fetchall()
        )
        return [self._row_to_episode(r) for r in rows]

    def get_salient_episodes(
        self, min_salience: float = 0.5, limit: int = 10
    ) -> List[Episode]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM episodes WHERE salience >= ? ORDER BY salience DESC LIMIT ?",
                (min_salience, limit),
            )
            .fetchall()
        )
        return [self._row_to_episode(r) for r in rows]

    def count_episodes(self) -> int:
        return self._get_conn().execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def get_session_turn_count(self, session_id: str) -> int:
        return (
            self._get_conn()
            .execute("SELECT COUNT(*) FROM episodes WHERE session_id=?", (session_id,))
            .fetchone()[0]
        )

    # -- Facts ---------------------------------------------------------------

    def insert_fact(self, fact: Fact) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO facts
                   (subject_type, subject_id, predicate, object_text, confidence,
                    source_episode_id, first_seen, last_seen, status, superseded_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact.subject_type,
                    fact.subject_id,
                    fact.predicate,
                    fact.object_text,
                    fact.confidence,
                    fact.source_episode_id,
                    fact.first_seen.isoformat(),
                    fact.last_seen.isoformat(),
                    fact.status,
                    fact.superseded_by,
                ),
            )
            fact.id = cur.lastrowid
            return fact.id

    def get_active_facts(
        self, subject_type: Optional[str] = None, limit: int = 20
    ) -> List[Fact]:
        if subject_type:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM facts WHERE status='active' AND subject_type=? ORDER BY confidence DESC LIMIT ?",
                    (subject_type, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM facts WHERE status='active' ORDER BY confidence DESC LIMIT ?",
                    (limit,),
                )
                .fetchall()
            )
        return [self._row_to_fact(r) for r in rows]

    def get_fact(self, fact_id: int) -> Optional[Fact]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM facts WHERE id=?", (fact_id,))
            .fetchone()
        )
        return self._row_to_fact(row) if row else None

    def get_facts_superseded_by(self, new_fact_id: int, limit: int = 20) -> List[Fact]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM facts WHERE superseded_by=? ORDER BY id DESC LIMIT ?",
                (new_fact_id, limit),
            )
            .fetchall()
        )
        return [self._row_to_fact(r) for r in rows]

    def search_facts(self, query: str, limit: int = 10) -> List[Fact]:
        rows = (
            self._get_conn()
            .execute(
                """SELECT f.* FROM facts f
               JOIN facts_fts fts ON f.id = fts.rowid
               WHERE facts_fts MATCH ? AND f.status='active'
               ORDER BY rank
               LIMIT ?""",
                (query, limit),
            )
            .fetchall()
        )
        return [self._row_to_fact(r) for r in rows]

    def update_fact_confidence(self, fact_id: int, confidence: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                "UPDATE facts SET confidence=?, last_seen=? WHERE id=?",
                (confidence, now, fact_id),
            )

    def bump_fact_last_seen(self, fact_id: int) -> None:
        """Update last_seen without changing confidence."""
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                "UPDATE facts SET last_seen=? WHERE id=?",
                (now, fact_id),
            )

    def supersede_fact(self, old_id: int, new_id: int) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE facts SET status='superseded', superseded_by=? WHERE id=?",
                (new_id, old_id),
            )

    def get_facts_by_predicate(
        self, predicate: str, status: str = "active", limit: int = 20
    ) -> List[Fact]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM facts WHERE predicate=? AND status=? ORDER BY confidence DESC LIMIT ?",
                (predicate, status, limit),
            )
            .fetchall()
        )
        return [self._row_to_fact(r) for r in rows]

    def find_similar_facts(
        self, text: str, predicate: Optional[str] = None, limit: int = 5
    ) -> List[Fact]:
        """Find active facts with similar object_text via FTS."""
        if not text.strip():
            return []
        try:
            if predicate:
                rows = (
                    self._get_conn()
                    .execute(
                        """SELECT f.* FROM facts f
                        JOIN facts_fts fts ON f.id = fts.rowid
                        WHERE facts_fts MATCH ? AND f.status='active' AND f.predicate=?
                        ORDER BY rank LIMIT ?""",
                        (text, predicate, limit),
                    )
                    .fetchall()
                )
            else:
                rows = (
                    self._get_conn()
                    .execute(
                        """SELECT f.* FROM facts f
                        JOIN facts_fts fts ON f.id = fts.rowid
                        WHERE facts_fts MATCH ? AND f.status='active'
                        ORDER BY rank LIMIT ?""",
                        (text, limit),
                    )
                    .fetchall()
                )
            return [self._row_to_fact(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS query syntax error — fall back to empty
            return []

    def count_facts(self, status: str = "active") -> int:
        return (
            self._get_conn()
            .execute("SELECT COUNT(*) FROM facts WHERE status=?", (status,))
            .fetchone()[0]
        )

    def decay_stale_facts(
        self, days_threshold: float = 60.0, decay_rate: float = 0.05
    ) -> int:
        """Reduce confidence of facts not seen recently. Returns count of decayed facts."""
        cutoff = datetime.now(timezone.utc).timestamp() - (days_threshold * 86400)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        with self._tx() as conn:
            cur = conn.execute(
                """UPDATE facts
                   SET confidence = MAX(0.1, confidence - ?)
                   WHERE status='active' AND last_seen < ? AND confidence > 0.1""",
                (decay_rate, cutoff_iso),
            )
            return cur.rowcount

    # -- Open Loops ----------------------------------------------------------

    def insert_open_loop(self, loop: OpenLoop) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO open_loops (kind, text, due_hint, status, source_episode_id, created_at, resolved_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    loop.kind,
                    loop.text,
                    loop.due_hint,
                    loop.status,
                    loop.source_episode_id,
                    loop.created_at.isoformat(),
                    loop.resolved_at.isoformat() if loop.resolved_at else None,
                ),
            )
            loop.id = cur.lastrowid
            return loop.id

    def get_open_loops(self, limit: int = 10) -> List[OpenLoop]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM open_loops WHERE status='open' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            .fetchall()
        )
        return [self._row_to_open_loop(r) for r in rows]

    def resolve_loop(self, loop_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                "UPDATE open_loops SET status='resolved', resolved_at=? WHERE id=?",
                (now, loop_id),
            )

    def expire_old_loops(self, days_threshold: float = 14.0) -> int:
        """Mark open loops older than threshold as expired. Returns count."""
        cutoff = datetime.now(timezone.utc).timestamp() - (days_threshold * 86400)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE open_loops SET status='expired' WHERE status='open' AND created_at < ?",
                (cutoff_iso,),
            )
            return cur.rowcount

    def count_open_loops(self) -> int:
        return (
            self._get_conn()
            .execute("SELECT COUNT(*) FROM open_loops WHERE status='open'")
            .fetchone()[0]
        )

    def search_open_loops(self, text: str, limit: int = 5) -> List[OpenLoop]:
        """Simple LIKE search on open loop text (no FTS table for loops)."""
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM open_loops WHERE status='open' AND text LIKE ? LIMIT ?",
                (f"%{text}%", limit),
            )
            .fetchall()
        )
        return [self._row_to_open_loop(r) for r in rows]

    # -- Reflections ---------------------------------------------------------

    def insert_reflection(self, ref: Reflection) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO reflections (kind, text, confidence, source_episode_id,
                   created_at, last_reinforced, reinforcement_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ref.kind,
                    ref.text,
                    ref.confidence,
                    ref.source_episode_id,
                    ref.created_at.isoformat(),
                    ref.last_reinforced.isoformat(),
                    ref.reinforcement_count,
                ),
            )
            ref.id = cur.lastrowid
            return ref.id

    def get_reflections(
        self, kind: Optional[str] = None, limit: int = 10
    ) -> List[Reflection]:
        if kind:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM reflections WHERE kind=? ORDER BY confidence DESC LIMIT ?",
                    (kind, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM reflections ORDER BY confidence DESC LIMIT ?",
                    (limit,),
                )
                .fetchall()
            )
        return [self._row_to_reflection(r) for r in rows]

    def get_reflection(self, reflection_id: int) -> Optional[Reflection]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM reflections WHERE id=?", (reflection_id,))
            .fetchone()
        )
        return self._row_to_reflection(row) if row else None

    def search_reflections(self, query: str, limit: int = 10) -> List[Reflection]:
        rows = (
            self._get_conn()
            .execute(
                """SELECT r.* FROM reflections r
               JOIN reflections_fts fts ON r.id = fts.rowid
               WHERE reflections_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
                (query, limit),
            )
            .fetchall()
        )
        return [self._row_to_reflection(r) for r in rows]

    def reinforce_reflection(
        self, reflection_id: int, confidence_boost: float = 0.1
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """UPDATE reflections
                   SET confidence = MIN(1.0, confidence + ?),
                       last_reinforced = ?,
                       reinforcement_count = reinforcement_count + 1
                   WHERE id = ?""",
                (confidence_boost, now, reflection_id),
            )

    def get_high_confidence_reflections(
        self, min_confidence: float = 0.5, limit: int = 10
    ) -> List[Reflection]:
        """Get reflections above a confidence threshold, ordered by confidence."""
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM reflections WHERE confidence >= ? ORDER BY confidence DESC LIMIT ?",
                (min_confidence, limit),
            )
            .fetchall()
        )
        return [self._row_to_reflection(r) for r in rows]

    def decay_stale_reflections(
        self, days_threshold: float = 30.0, decay_rate: float = 0.05
    ) -> int:
        """Reduce confidence of reflections not reinforced recently. Returns count."""
        cutoff = datetime.now(timezone.utc).timestamp() - (days_threshold * 86400)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        with self._tx() as conn:
            cur = conn.execute(
                """UPDATE reflections
                   SET confidence = MAX(0.05, confidence - ?)
                   WHERE last_reinforced < ? AND confidence > 0.05""",
                (decay_rate, cutoff_iso),
            )
            return cur.rowcount

    def get_identity_deltas(
        self, applied: Optional[bool] = None, limit: int = 10
    ) -> List["IdentityDelta"]:
        """Get identity deltas, optionally filtered by applied status."""
        from .models import IdentityDelta

        if applied is not None:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM identity_deltas WHERE applied=? ORDER BY created_at DESC LIMIT ?",
                    (int(applied), limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM identity_deltas ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
                .fetchall()
            )
        return [
            IdentityDelta(
                id=r["id"],
                text=r["text"],
                confidence=r["confidence"],
                source_episode_id=r["source_episode_id"],
                created_at=datetime.fromisoformat(r["created_at"]),
                applied=bool(r["applied"]),
            )
            for r in rows
        ]

    # -- Relationship State --------------------------------------------------

    def get_relationship(self, user_id: str = "default") -> RelationshipState:
        row = (
            self._get_conn()
            .execute("SELECT * FROM relationship_state WHERE user_id=?", (user_id,))
            .fetchone()
        )
        if row:
            return self._row_to_relationship(row)
        return RelationshipState(user_id=user_id)

    def upsert_relationship(self, rs: RelationshipState) -> None:
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO relationship_state
                   (user_id, warmth, trust, tension, familiarity, humor,
                    formality, volatility, last_updated, total_turns)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                    warmth=excluded.warmth, trust=excluded.trust,
                    tension=excluded.tension, familiarity=excluded.familiarity,
                    humor=excluded.humor, formality=excluded.formality,
                    volatility=excluded.volatility, last_updated=excluded.last_updated,
                    total_turns=excluded.total_turns""",
                (
                    rs.user_id,
                    rs.warmth,
                    rs.trust,
                    rs.tension,
                    rs.familiarity,
                    rs.humor,
                    rs.formality,
                    rs.volatility,
                    rs.last_updated.isoformat(),
                    rs.total_turns,
                ),
            )

    # -- Identity Deltas -----------------------------------------------------

    def insert_identity_delta(self, delta: "IdentityDelta") -> int:
        from .models import IdentityDelta

        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO identity_deltas (text, confidence, source_episode_id, created_at, applied)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    delta.text,
                    delta.confidence,
                    delta.source_episode_id,
                    delta.created_at.isoformat(),
                    int(delta.applied),
                ),
            )
            delta.id = cur.lastrowid
            return delta.id

    # -- Entity Links --------------------------------------------------------

    def insert_link(
        self,
        src_type: str,
        src_id: int,
        dst_type: str,
        dst_id: int,
        relation: str,
        weight: float = 1.0,
    ) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO entity_links (src_type, src_id, dst_type, dst_id, relation, weight)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (src_type, src_id, dst_type, dst_id, relation, weight),
            )
            return cur.lastrowid

    def get_links_from(
        self,
        src_type: str,
        src_id: int,
        relation: str = None,
        limit: int = 20,
    ) -> List[dict]:
        query = (
            "SELECT dst_type, dst_id, relation, weight FROM entity_links "
            "WHERE src_type=? AND src_id=?"
        )
        params: List[Any] = [src_type, src_id]
        if relation:
            query += " AND relation=?"
            params.append(relation)
        query += " ORDER BY weight DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self._get_conn().execute(query, tuple(params)).fetchall()
        return [
            {
                "dst_type": row["dst_type"],
                "dst_id": row["dst_id"],
                "relation": row["relation"],
                "weight": row["weight"],
            }
            for row in rows
        ]

    def get_links_to(
        self,
        dst_type: str,
        dst_id: int,
        relation: str = None,
        limit: int = 20,
    ) -> List[dict]:
        query = (
            "SELECT src_type, src_id, relation, weight FROM entity_links "
            "WHERE dst_type=? AND dst_id=?"
        )
        params: List[Any] = [dst_type, dst_id]
        if relation:
            query += " AND relation=?"
            params.append(relation)
        query += " ORDER BY weight DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self._get_conn().execute(query, tuple(params)).fetchall()
        return [
            {
                "src_type": row["src_type"],
                "src_id": row["src_id"],
                "relation": row["relation"],
                "weight": row["weight"],
            }
            for row in rows
        ]

    def delete_links(self, src_type: str, src_id: int) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                "DELETE FROM entity_links WHERE src_type=? AND src_id=?",
                (src_type, src_id),
            )
            return cur.rowcount

    def count_links(self) -> int:
        return (
            self._get_conn().execute("SELECT COUNT(*) FROM entity_links").fetchone()[0]
        )

    def link_exists(
        self,
        src_type: str,
        src_id: int,
        dst_type: str,
        dst_id: int,
        relation: str,
    ) -> bool:
        row = (
            self._get_conn()
            .execute(
                """SELECT 1 FROM entity_links
                   WHERE src_type=? AND src_id=? AND dst_type=? AND dst_id=? AND relation=?
                   LIMIT 1""",
                (src_type, src_id, dst_type, dst_id, relation),
            )
            .fetchone()
        )
        return row is not None

    # -- Emotion Log ---------------------------------------------------------

    def insert_emotion_log(
        self,
        affect: AffectSignal,
        episode_id: int,
        session_id: str = "",
        timestamp: Optional[datetime] = None,
    ) -> int:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO emotion_log
                   (episode_id, session_id, timestamp, frustration, warmth, humor,
                    hostility, gratitude, anxiety, excitement, trust_signal,
                    valence, arousal, dominant_emotion, is_sarcastic)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode_id,
                    session_id,
                    ts,
                    affect.frustration,
                    affect.warmth,
                    affect.humor,
                    affect.hostility,
                    affect.gratitude,
                    affect.anxiety,
                    affect.excitement,
                    affect.trust_signal,
                    affect.valence,
                    affect.arousal,
                    affect.dominant_emotion,
                    int(affect.is_sarcastic),
                ),
            )
            return cur.lastrowid

    def get_recent_emotions(
        self, limit: int = 10, session_id: Optional[str] = None
    ) -> List[AffectSignal]:
        if session_id:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM emotion_log WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM emotion_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
                .fetchall()
            )
        return [self._row_to_affect_signal(r) for r in rows]

    def get_emotion_for_episode(self, episode_id: int) -> Optional[AffectSignal]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM emotion_log WHERE episode_id=?", (episode_id,))
            .fetchone()
        )
        return self._row_to_affect_signal(row) if row else None

    def get_emotional_trajectory(
        self, limit: int = 20, session_id: Optional[str] = None
    ) -> List[dict]:
        """Get recent emotion entries as compact dicts for trajectory analysis."""
        if session_id:
            rows = (
                self._get_conn()
                .execute(
                    """SELECT timestamp, valence, arousal, dominant_emotion
                       FROM emotion_log WHERE session_id=?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (session_id, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    """SELECT timestamp, valence, arousal, dominant_emotion
                       FROM emotion_log ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                )
                .fetchall()
            )
        return [
            {
                "timestamp": r["timestamp"],
                "valence": r["valence"],
                "arousal": r["arousal"],
                "emotion": r["dominant_emotion"],
            }
            for r in rows
        ]

    # -- Row mappers ---------------------------------------------------------

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            user_text=row["user_text"],
            assistant_text=row["assistant_text"],
            summary=row["summary"],
            salience=row["salience"],
            valence=row["valence"],
            arousal=row["arousal"],
            topics=row["topics"],
            entities=row["entities"],
            is_consolidated=bool(row["is_consolidated"]),
        )

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            subject_type=row["subject_type"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_text=row["object_text"],
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            status=row["status"],
            superseded_by=row["superseded_by"],
        )

    @staticmethod
    def _row_to_open_loop(row: sqlite3.Row) -> OpenLoop:
        return OpenLoop(
            id=row["id"],
            kind=row["kind"],
            text=row["text"],
            due_hint=row["due_hint"],
            status=row["status"],
            source_episode_id=row["source_episode_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            resolved_at=datetime.fromisoformat(row["resolved_at"])
            if row["resolved_at"]
            else None,
        )

    @staticmethod
    def _row_to_reflection(row: sqlite3.Row) -> Reflection:
        return Reflection(
            id=row["id"],
            kind=row["kind"],
            text=row["text"],
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_reinforced=datetime.fromisoformat(row["last_reinforced"]),
            reinforcement_count=row["reinforcement_count"],
        )

    @staticmethod
    def _row_to_relationship(row: sqlite3.Row) -> RelationshipState:
        return RelationshipState(
            id=row["id"],
            user_id=row["user_id"],
            warmth=row["warmth"],
            trust=row["trust"],
            tension=row["tension"],
            familiarity=row["familiarity"],
            humor=row["humor"],
            formality=row["formality"],
            volatility=row["volatility"],
            last_updated=datetime.fromisoformat(row["last_updated"]),
            total_turns=row["total_turns"],
        )

    @staticmethod
    def _row_to_affect_signal(row: sqlite3.Row) -> AffectSignal:
        return AffectSignal(
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

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
