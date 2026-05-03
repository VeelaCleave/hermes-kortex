"""SQLite storage layer for KORTEX.

Single-file database with FTS5 for full-text search. All tables created
on first initialize(). Schema versioned via user_version pragma.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    AffectSignal,
    Episode,
    Fact,
    IdentityDelta,
    OpenLoop,
    Reflection,
    RelationshipState,
)
from .calibrate import AffectBaseline
from .time_utils import now_epoch, parse_timestamp

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "__default__"
SCHEMA_VERSION = 5

# OCEAN personality schema (Big Five: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
_OCEAN_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ocean_profiles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL DEFAULT '',
    openness            REAL NOT NULL DEFAULT 0.5,
    conscientiousness   REAL NOT NULL DEFAULT 0.5,
    extraversion        REAL NOT NULL DEFAULT 0.5,
    agreeableness       REAL NOT NULL DEFAULT 0.5,
    neuroticism         REAL NOT NULL DEFAULT 0.5,
    confidence          REAL NOT NULL DEFAULT 0.3,
    turn_count          INTEGER NOT NULL DEFAULT 0,
    updated_at          REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_ocean_user_id ON ocean_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_ocean_updated ON ocean_profiles(updated_at);
"""

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS episodes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
    session_id          TEXT NOT NULL,
    turn_index          INTEGER NOT NULL DEFAULT 0,
    timestamp           REAL NOT NULL,
    user_text           TEXT NOT NULL DEFAULT '',
    assistant_text      TEXT NOT NULL DEFAULT '',
    summary             TEXT NOT NULL DEFAULT '',
    salience            REAL NOT NULL DEFAULT 0.0,
    valence             INTEGER NOT NULL DEFAULT 0,
    arousal             REAL NOT NULL DEFAULT 0.0,
    topics              TEXT NOT NULL DEFAULT '',
    entities            TEXT NOT NULL DEFAULT '',
    is_consolidated     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at    REAL,
    retrieval_count     INTEGER NOT NULL DEFAULT 0,
    consolidated_into   INTEGER REFERENCES episodes(id),
    raw_preserved       INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_salience ON episodes(salience DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);

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
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                 TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
    subject_type            TEXT NOT NULL DEFAULT 'user',
    subject_id              TEXT NOT NULL DEFAULT '',
    predicate               TEXT NOT NULL DEFAULT '',
    object_text             TEXT NOT NULL DEFAULT '',
    confidence              REAL NOT NULL DEFAULT 0.5,
    source_episode_id       INTEGER REFERENCES episodes(id),
    first_seen              REAL NOT NULL,
    last_seen               REAL NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'active',
    superseded_by           INTEGER REFERENCES facts(id),
    last_accessed_at        REAL,
    retrieval_count         INTEGER NOT NULL DEFAULT 0,
    valid_from              REAL,
    valid_to                REAL,
    contradiction_status    TEXT NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_type, subject_id);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_contradiction ON facts(contradiction_status);
CREATE INDEX IF NOT EXISTS idx_facts_accessed ON facts(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_facts_source_episode ON facts(source_episode_id);

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
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                 TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
    kind                    TEXT NOT NULL DEFAULT 'commitment',
    text                    TEXT NOT NULL DEFAULT '',
    due_hint                TEXT NOT NULL DEFAULT '',
    status                  TEXT NOT NULL DEFAULT 'open',
    source_episode_id       INTEGER REFERENCES episodes(id),
    created_at              REAL NOT NULL,
    resolved_at             REAL,
    last_accessed_at        REAL,
    resolution              TEXT NOT NULL DEFAULT '',
    resolved_by_episode_id  INTEGER REFERENCES episodes(id)
);

CREATE INDEX IF NOT EXISTS idx_loops_status ON open_loops(status);
CREATE INDEX IF NOT EXISTS idx_loops_user ON open_loops(user_id);

CREATE TABLE IF NOT EXISTS reflections (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                 TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
    kind                    TEXT NOT NULL DEFAULT 'pattern',
    text                    TEXT NOT NULL DEFAULT '',
    confidence              REAL NOT NULL DEFAULT 0.3,
    source_episode_id       INTEGER REFERENCES episodes(id),
    created_at              REAL NOT NULL,
    last_reinforced         REAL NOT NULL,
    reinforcement_count     INTEGER NOT NULL DEFAULT 1,
    last_accessed_at        REAL,
    retrieval_count         INTEGER NOT NULL DEFAULT 0,
    promotion_status        TEXT NOT NULL DEFAULT 'active',
    promoted_at             REAL
);

CREATE INDEX IF NOT EXISTS idx_reflections_user ON reflections(user_id);
CREATE INDEX IF NOT EXISTS idx_reflections_promotion ON reflections(promotion_status);

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
    user_id         TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}' UNIQUE,
    warmth          REAL NOT NULL DEFAULT 0.5,
    trust           REAL NOT NULL DEFAULT 0.5,
    tension         REAL NOT NULL DEFAULT 0.0,
    familiarity     REAL NOT NULL DEFAULT 0.0,
    humor           REAL NOT NULL DEFAULT 0.0,
    formality       REAL NOT NULL DEFAULT 0.5,
    volatility      REAL NOT NULL DEFAULT 0.0,
    last_updated    REAL NOT NULL,
    total_turns     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS identity_deltas (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
    text                TEXT NOT NULL DEFAULT '',
    confidence          REAL NOT NULL DEFAULT 0.3,
    source_episode_id   INTEGER REFERENCES episodes(id),
    created_at          REAL NOT NULL,
    applied             INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entity_links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
    src_type    TEXT NOT NULL,
    src_id      INTEGER NOT NULL,
    dst_type    TEXT NOT NULL,
    dst_id      INTEGER NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_links_src ON entity_links(src_type, src_id);
CREATE INDEX IF NOT EXISTS idx_links_dst ON entity_links(dst_type, dst_id);
CREATE INDEX IF NOT EXISTS idx_links_user ON entity_links(user_id);

CREATE TABLE IF NOT EXISTS emotion_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
    episode_id          INTEGER REFERENCES episodes(id),
    session_id          TEXT NOT NULL DEFAULT '',
    timestamp           REAL NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_emotion_user ON emotion_log(user_id);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
    session_id          TEXT NOT NULL,
    summary_text        TEXT NOT NULL,
    summary_level       TEXT NOT NULL DEFAULT 'conversation',
    episode_range_start REAL,
    episode_range_end   REAL,
    episode_count       INTEGER NOT NULL DEFAULT 0,
    key_entities        TEXT NOT NULL DEFAULT '',
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_summaries_user ON conversation_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_summaries_session ON conversation_summaries(session_id);

CREATE TABLE IF NOT EXISTS affect_baselines (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                 TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}' UNIQUE,
    baseline_frustration    REAL NOT NULL DEFAULT 0.0,
    baseline_warmth         REAL NOT NULL DEFAULT 0.0,
    baseline_humor          REAL NOT NULL DEFAULT 0.0,
    baseline_hostility      REAL NOT NULL DEFAULT 0.0,
    baseline_gratitude      REAL NOT NULL DEFAULT 0.0,
    baseline_anxiety        REAL NOT NULL DEFAULT 0.0,
    baseline_excitement     REAL NOT NULL DEFAULT 0.0,
    baseline_trust_signal   REAL NOT NULL DEFAULT 0.0,
    sample_count            INTEGER NOT NULL DEFAULT 0,
    ema_alpha               REAL NOT NULL DEFAULT 0.1,
    created_at              REAL NOT NULL,
    updated_at              REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kortex_schema_version (
    version     INTEGER NOT NULL,
    migrated_at REAL NOT NULL
);

-- Semantic embeddings table for vector similarity search
CREATE TABLE IF NOT EXISTS embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id       INTEGER NOT NULL,
    entity_type     TEXT NOT NULL,
    embedding_vector BLOB NOT NULL,
    created_at      REAL NOT NULL,
    user_id         TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'
);
CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_user ON embeddings(user_id);
"""

_LOSSLESS_CONTEXT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS context_conversations (
    conversation_id       TEXT PRIMARY KEY,
    user_id               TEXT NOT NULL DEFAULT '__default__',
    created_at            REAL NOT NULL,
    updated_at            REAL NOT NULL,
    status                TEXT NOT NULL DEFAULT 'active',
    active_checkpoint_id  TEXT
);

CREATE TABLE IF NOT EXISTS context_session_aliases (
    hermes_session_id         TEXT PRIMARY KEY,
    conversation_id           TEXT NOT NULL REFERENCES context_conversations(conversation_id),
    lineage_parent_session_id TEXT,
    created_at                REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS context_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES context_conversations(conversation_id),
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    raw_json        TEXT NOT NULL,
    text_content    TEXT NOT NULL DEFAULT '',
    content_hash    TEXT NOT NULL DEFAULT '',
    token_estimate  INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    UNIQUE (conversation_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_context_messages_conv_seq
ON context_messages (conversation_id, seq);

CREATE VIRTUAL TABLE IF NOT EXISTS context_messages_fts USING fts5(
    text_content,
    conversation_id UNINDEXED,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS context_messages_ai AFTER INSERT ON context_messages BEGIN
    INSERT INTO context_messages_fts(rowid, text_content, conversation_id)
    VALUES (new.id, new.text_content, new.conversation_id);
END;

CREATE TRIGGER IF NOT EXISTS context_messages_ad AFTER DELETE ON context_messages BEGIN
    INSERT INTO context_messages_fts(context_messages_fts, rowid, text_content, conversation_id)
    VALUES ('delete', old.id, old.text_content, old.conversation_id);
END;

CREATE TRIGGER IF NOT EXISTS context_messages_au AFTER UPDATE ON context_messages BEGIN
    INSERT INTO context_messages_fts(context_messages_fts, rowid, text_content, conversation_id)
    VALUES ('delete', old.id, old.text_content, old.conversation_id);
    INSERT INTO context_messages_fts(rowid, text_content, conversation_id)
    VALUES (new.id, new.text_content, new.conversation_id);
END;

CREATE TABLE IF NOT EXISTS context_spans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES context_conversations(conversation_id),
    start_seq       INTEGER NOT NULL,
    end_seq         INTEGER NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'raw',
    checkpoint_id   TEXT,
    archived_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_context_spans_conv_range
ON context_spans (conversation_id, start_seq, end_seq);

CREATE TABLE IF NOT EXISTS context_refs (
    ref_id          TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES context_conversations(conversation_id),
    ref_type        TEXT NOT NULL,
    label           TEXT NOT NULL,
    payload_json    TEXT NOT NULL DEFAULT '{}',
    source_span_id  INTEGER REFERENCES context_spans(id),
    salience        REAL NOT NULL DEFAULT 0.0,
    open_state      TEXT NOT NULL DEFAULT 'open',
    created_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_context_refs_conv_type
ON context_refs (conversation_id, ref_type);

CREATE VIRTUAL TABLE IF NOT EXISTS context_refs_fts USING fts5(
    label,
    conversation_id UNINDEXED,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS context_refs_ai AFTER INSERT ON context_refs BEGIN
    INSERT INTO context_refs_fts(rowid, label, conversation_id)
    VALUES (new.rowid, new.label, new.conversation_id);
END;

CREATE TRIGGER IF NOT EXISTS context_refs_ad AFTER DELETE ON context_refs BEGIN
    INSERT INTO context_refs_fts(context_refs_fts, rowid, label, conversation_id)
    VALUES ('delete', old.rowid, old.label, old.conversation_id);
END;

CREATE TRIGGER IF NOT EXISTS context_refs_au AFTER UPDATE ON context_refs BEGIN
    INSERT INTO context_refs_fts(context_refs_fts, rowid, label, conversation_id)
    VALUES ('delete', old.rowid, old.label, old.conversation_id);
    INSERT INTO context_refs_fts(rowid, label, conversation_id)
    VALUES (new.rowid, new.label, new.conversation_id);
END;

CREATE TABLE IF NOT EXISTS context_checkpoints (
    checkpoint_id         TEXT PRIMARY KEY,
    conversation_id       TEXT NOT NULL REFERENCES context_conversations(conversation_id),
    replaced_start_seq    INTEGER NOT NULL,
    replaced_end_seq      INTEGER NOT NULL,
    export_text           TEXT NOT NULL,
    export_token_estimate INTEGER NOT NULL DEFAULT 0,
    source_span_ids_json  TEXT NOT NULL DEFAULT '[]',
    hot_ref_ids_json      TEXT NOT NULL DEFAULT '[]',
    created_at            REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_context_checkpoints_conv_created
ON context_checkpoints (conversation_id, created_at);
"""


def _as_epoch(value: Any) -> Optional[float]:
    return parse_timestamp(value)


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
            conn.executescript(_OCEAN_SCHEMA_SQL)
            conn.executescript(_LOSSLESS_CONTEXT_SCHEMA_SQL)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._record_schema_version(conn)
            conn.commit()
            logger.info(
                "KORTEX DB initialized (v%d) at %s", SCHEMA_VERSION, self._db_path
            )
        elif version < SCHEMA_VERSION:
            self._migrate(conn, version)
        else:
            conn.executescript(_LOSSLESS_CONTEXT_SCHEMA_SQL)
            conn.commit()

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        if from_version < 2:
            conn.executescript(
                """
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
            )
            from_version = 2

        if from_version < 3:
            self._migrate_v2_to_v3(conn)

        if from_version < 4:
            conn.executescript(_OCEAN_SCHEMA_SQL)
            from_version = 4

        # v5: Evidence trace index + embeddings table for semantic search
        if from_version < 5:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_source_episode ON facts(source_episode_id)"
            )
            # Create embeddings table for semantic search (was missing from migration)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id       INTEGER NOT NULL,
                    entity_type     TEXT NOT NULL,
                    embedding_vector BLOB NOT NULL,
                    created_at      REAL NOT NULL,
                    user_id         TEXT NOT NULL DEFAULT '__default__'
                );
                CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_user ON embeddings(user_id);
            """)
            from_version = 5

        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._record_schema_version(conn)
        conn.commit()
        logger.info("KORTEX DB migrated to v%d at %s", SCHEMA_VERSION, self._db_path)

    @staticmethod
    def _record_schema_version(conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kortex_schema_version (version INTEGER NOT NULL, migrated_at REAL NOT NULL)"
        )
        conn.execute("DELETE FROM kortex_schema_version")
        conn.execute(
            "INSERT INTO kortex_schema_version(version, migrated_at) VALUES (?, ?)",
            (SCHEMA_VERSION, now_epoch()),
        )

    def _migrate_v2_to_v3(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            conn.executescript(
                f"""
                CREATE TABLE episodes_v3 (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id             TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
                    session_id          TEXT NOT NULL,
                    turn_index          INTEGER NOT NULL DEFAULT 0,
                    timestamp           REAL NOT NULL,
                    user_text           TEXT NOT NULL DEFAULT '',
                    assistant_text      TEXT NOT NULL DEFAULT '',
                    summary             TEXT NOT NULL DEFAULT '',
                    salience            REAL NOT NULL DEFAULT 0.0,
                    valence             INTEGER NOT NULL DEFAULT 0,
                    arousal             REAL NOT NULL DEFAULT 0.0,
                    topics              TEXT NOT NULL DEFAULT '',
                    entities            TEXT NOT NULL DEFAULT '',
                    is_consolidated     INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at    REAL,
                    retrieval_count     INTEGER NOT NULL DEFAULT 0,
                    consolidated_into   INTEGER REFERENCES episodes(id),
                    raw_preserved       INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE facts_v3 (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id                 TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
                    subject_type            TEXT NOT NULL DEFAULT 'user',
                    subject_id              TEXT NOT NULL DEFAULT '',
                    predicate               TEXT NOT NULL DEFAULT '',
                    object_text             TEXT NOT NULL DEFAULT '',
                    confidence              REAL NOT NULL DEFAULT 0.5,
                    source_episode_id       INTEGER REFERENCES episodes(id),
                    first_seen              REAL NOT NULL,
                    last_seen               REAL NOT NULL,
                    status                  TEXT NOT NULL DEFAULT 'active',
                    superseded_by           INTEGER REFERENCES facts(id),
                    last_accessed_at        REAL,
                    retrieval_count         INTEGER NOT NULL DEFAULT 0,
                    valid_from              REAL,
                    valid_to                REAL,
                    contradiction_status    TEXT NOT NULL DEFAULT 'active'
                );

                CREATE TABLE open_loops_v3 (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id                 TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
                    kind                    TEXT NOT NULL DEFAULT 'commitment',
                    text                    TEXT NOT NULL DEFAULT '',
                    due_hint                TEXT NOT NULL DEFAULT '',
                    status                  TEXT NOT NULL DEFAULT 'open',
                    source_episode_id       INTEGER REFERENCES episodes(id),
                    created_at              REAL NOT NULL,
                    resolved_at             REAL,
                    last_accessed_at        REAL,
                    resolution              TEXT NOT NULL DEFAULT '',
                    resolved_by_episode_id  INTEGER REFERENCES episodes(id)
                );

                CREATE TABLE reflections_v3 (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id                 TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
                    kind                    TEXT NOT NULL DEFAULT 'pattern',
                    text                    TEXT NOT NULL DEFAULT '',
                    confidence              REAL NOT NULL DEFAULT 0.3,
                    source_episode_id       INTEGER REFERENCES episodes(id),
                    created_at              REAL NOT NULL,
                    last_reinforced         REAL NOT NULL,
                    reinforcement_count     INTEGER NOT NULL DEFAULT 1,
                    last_accessed_at        REAL,
                    retrieval_count         INTEGER NOT NULL DEFAULT 0,
                    promotion_status        TEXT NOT NULL DEFAULT 'active',
                    promoted_at             REAL
                );

                CREATE TABLE relationship_state_v3 (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}' UNIQUE,
                    warmth          REAL NOT NULL DEFAULT 0.5,
                    trust           REAL NOT NULL DEFAULT 0.5,
                    tension         REAL NOT NULL DEFAULT 0.0,
                    familiarity     REAL NOT NULL DEFAULT 0.0,
                    humor           REAL NOT NULL DEFAULT 0.0,
                    formality       REAL NOT NULL DEFAULT 0.5,
                    volatility      REAL NOT NULL DEFAULT 0.0,
                    last_updated    REAL NOT NULL,
                    total_turns     INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE identity_deltas_v3 (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id             TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
                    text                TEXT NOT NULL DEFAULT '',
                    confidence          REAL NOT NULL DEFAULT 0.3,
                    source_episode_id   INTEGER REFERENCES episodes(id),
                    created_at          REAL NOT NULL,
                    applied             INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE entity_links_v3 (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
                    src_type    TEXT NOT NULL,
                    src_id      INTEGER NOT NULL,
                    dst_type    TEXT NOT NULL,
                    dst_id      INTEGER NOT NULL,
                    relation    TEXT NOT NULL,
                    weight      REAL NOT NULL DEFAULT 1.0
                );

                CREATE TABLE emotion_log_v3 (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id             TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}',
                    episode_id          INTEGER REFERENCES episodes(id),
                    session_id          TEXT NOT NULL DEFAULT '',
                    timestamp           REAL NOT NULL,
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
                """
            )

            self._copy_table_rows(conn)

            conn.executescript(
                """
                DROP TRIGGER IF EXISTS episodes_ai;
                DROP TRIGGER IF EXISTS episodes_ad;
                DROP TRIGGER IF EXISTS episodes_au;
                DROP TRIGGER IF EXISTS facts_ai;
                DROP TRIGGER IF EXISTS facts_ad;
                DROP TRIGGER IF EXISTS facts_au;
                DROP TRIGGER IF EXISTS reflections_ai;
                DROP TRIGGER IF EXISTS reflections_ad;
                DROP TRIGGER IF EXISTS reflections_au;
                DROP TABLE IF EXISTS episodes_fts;
                DROP TABLE IF EXISTS facts_fts;
                DROP TABLE IF EXISTS reflections_fts;
                DROP TABLE IF EXISTS episodes;
                DROP TABLE IF EXISTS facts;
                DROP TABLE IF EXISTS open_loops;
                DROP TABLE IF EXISTS reflections;
                DROP TABLE IF EXISTS relationship_state;
                DROP TABLE IF EXISTS identity_deltas;
                DROP TABLE IF EXISTS entity_links;
                DROP TABLE IF EXISTS emotion_log;
                ALTER TABLE episodes_v3 RENAME TO episodes;
                ALTER TABLE facts_v3 RENAME TO facts;
                ALTER TABLE open_loops_v3 RENAME TO open_loops;
                ALTER TABLE reflections_v3 RENAME TO reflections;
                ALTER TABLE relationship_state_v3 RENAME TO relationship_state;
                ALTER TABLE identity_deltas_v3 RENAME TO identity_deltas;
                ALTER TABLE entity_links_v3 RENAME TO entity_links;
                ALTER TABLE emotion_log_v3 RENAME TO emotion_log;
                """
            )

            conn.executescript(_SCHEMA_SQL)
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

    def _copy_table_rows(self, conn: sqlite3.Connection) -> None:
        for row in conn.execute("SELECT * FROM episodes").fetchall():
            conn.execute(
                """INSERT INTO episodes_v3
                   (id, user_id, session_id, turn_index, timestamp, user_text, assistant_text,
                    summary, salience, valence, arousal, topics, entities, is_consolidated,
                    last_accessed_at, retrieval_count, consolidated_into, raw_preserved)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["id"],
                    DEFAULT_USER_ID,
                    row["session_id"],
                    row["turn_index"],
                    parse_timestamp(row["timestamp"]) or 0.0,
                    row["user_text"],
                    row["assistant_text"],
                    row["summary"],
                    row["salience"],
                    row["valence"],
                    row["arousal"],
                    row["topics"],
                    row["entities"],
                    row["is_consolidated"],
                    None,
                    0,
                    None,
                    1,
                ),
            )

        for row in conn.execute("SELECT * FROM facts").fetchall():
            first_seen = parse_timestamp(row["first_seen"]) or 0.0
            last_seen = parse_timestamp(row["last_seen"]) or first_seen
            conn.execute(
                """INSERT INTO facts_v3
                   (id, user_id, subject_type, subject_id, predicate, object_text, confidence,
                    source_episode_id, first_seen, last_seen, status, superseded_by,
                    last_accessed_at, retrieval_count, valid_from, valid_to, contradiction_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["id"],
                    DEFAULT_USER_ID,
                    row["subject_type"],
                    row["subject_id"],
                    row["predicate"],
                    row["object_text"],
                    row["confidence"],
                    row["source_episode_id"],
                    first_seen,
                    last_seen,
                    row["status"],
                    row["superseded_by"],
                    None,
                    0,
                    first_seen,
                    None,
                    "active",
                ),
            )

        for row in conn.execute("SELECT * FROM open_loops").fetchall():
            conn.execute(
                """INSERT INTO open_loops_v3
                   (id, user_id, kind, text, due_hint, status, source_episode_id, created_at,
                    resolved_at, last_accessed_at, resolution, resolved_by_episode_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["id"],
                    DEFAULT_USER_ID,
                    row["kind"],
                    row["text"],
                    row["due_hint"],
                    row["status"],
                    row["source_episode_id"],
                    parse_timestamp(row["created_at"]) or 0.0,
                    parse_timestamp(row["resolved_at"]),
                    None,
                    "",
                    None,
                ),
            )

        for row in conn.execute("SELECT * FROM reflections").fetchall():
            conn.execute(
                """INSERT INTO reflections_v3
                   (id, user_id, kind, text, confidence, source_episode_id, created_at,
                    last_reinforced, reinforcement_count, last_accessed_at, retrieval_count,
                    promotion_status, promoted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["id"],
                    DEFAULT_USER_ID,
                    row["kind"],
                    row["text"],
                    row["confidence"],
                    row["source_episode_id"],
                    parse_timestamp(row["created_at"]) or 0.0,
                    parse_timestamp(row["last_reinforced"]) or 0.0,
                    row["reinforcement_count"],
                    None,
                    0,
                    "active",
                    None,
                ),
            )

        for row in conn.execute("SELECT * FROM relationship_state").fetchall():
            conn.execute(
                """INSERT INTO relationship_state_v3
                   (id, user_id, warmth, trust, tension, familiarity, humor, formality,
                    volatility, last_updated, total_turns)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["id"],
                    row["user_id"] if row["user_id"] != "default" else DEFAULT_USER_ID,
                    row["warmth"],
                    row["trust"],
                    row["tension"],
                    row["familiarity"],
                    row["humor"],
                    row["formality"],
                    row["volatility"],
                    parse_timestamp(row["last_updated"]) or 0.0,
                    row["total_turns"],
                ),
            )

        for row in conn.execute("SELECT * FROM identity_deltas").fetchall():
            conn.execute(
                """INSERT INTO identity_deltas_v3
                   (id, user_id, text, confidence, source_episode_id, created_at, applied)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["id"],
                    DEFAULT_USER_ID,
                    row["text"],
                    row["confidence"],
                    row["source_episode_id"],
                    parse_timestamp(row["created_at"]) or 0.0,
                    row["applied"],
                ),
            )

        for row in conn.execute("SELECT * FROM entity_links").fetchall():
            conn.execute(
                """INSERT INTO entity_links_v3
                   (id, user_id, src_type, src_id, dst_type, dst_id, relation, weight)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["id"],
                    DEFAULT_USER_ID,
                    row["src_type"],
                    row["src_id"],
                    row["dst_type"],
                    row["dst_id"],
                    row["relation"],
                    row["weight"],
                ),
            )

        if self._table_exists(conn, "emotion_log"):
            for row in conn.execute("SELECT * FROM emotion_log").fetchall():
                conn.execute(
                    """INSERT INTO emotion_log_v3
                       (id, user_id, episode_id, session_id, timestamp, frustration, warmth,
                        humor, hostility, gratitude, anxiety, excitement, trust_signal,
                        valence, arousal, dominant_emotion, is_sarcastic)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["id"],
                        DEFAULT_USER_ID,
                        row["episode_id"],
                        row["session_id"],
                        parse_timestamp(row["timestamp"]) or 0.0,
                        row["frustration"],
                        row["warmth"],
                        row["humor"],
                        row["hostility"],
                        row["gratitude"],
                        row["anxiety"],
                        row["excitement"],
                        row["trust_signal"],
                        row["valence"],
                        row["arousal"],
                        row["dominant_emotion"],
                        row["is_sarcastic"],
                    ),
                )

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    # -- Episodes ------------------------------------------------------------

    def insert_episode(self, ep: Episode) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO episodes
                   (user_id, session_id, turn_index, timestamp, user_text, assistant_text,
                    summary, salience, valence, arousal, topics, entities, is_consolidated,
                    last_accessed_at, retrieval_count, consolidated_into, raw_preserved)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ep.user_id,
                    ep.session_id,
                    ep.turn_index,
                    ep.timestamp,
                    ep.user_text,
                    ep.assistant_text,
                    ep.summary,
                    ep.salience,
                    ep.valence,
                    ep.arousal,
                    ep.topics,
                    ep.entities,
                    int(ep.is_consolidated),
                    ep.last_accessed_at,
                    ep.retrieval_count,
                    ep.consolidated_into,
                    int(ep.raw_preserved),
                ),
            )
            ep.id = cur.lastrowid
            return ep.id

    def update_episode(self, ep: Episode) -> None:
        with self._tx() as conn:
            conn.execute(
                """UPDATE episodes SET summary=?, salience=?, valence=?, arousal=?,
                   topics=?, entities=?, is_consolidated=?, last_accessed_at=?,
                   retrieval_count=?, consolidated_into=?, raw_preserved=? WHERE id=?""",
                (
                    ep.summary,
                    ep.salience,
                    ep.valence,
                    ep.arousal,
                    ep.topics,
                    ep.entities,
                    int(ep.is_consolidated),
                    ep.last_accessed_at,
                    ep.retrieval_count,
                    ep.consolidated_into,
                    int(ep.raw_preserved),
                    ep.id,
                ),
            )

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM episodes WHERE id=?", (episode_id,))
            .fetchone()
        )
        if row:
            self.touch_episode(episode_id)
        return Episode.from_db_row(dict(row)) if row else None

    def get_recent_episodes(
        self,
        limit: int = 10,
        session_id: Optional[str] = None,
        include_consolidated: bool = False,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Episode]:
        filters = []
        params: List[Any] = [user_id]
        filters.append("user_id=?")
        if not include_consolidated:
            filters.append("is_consolidated=0")
        if session_id:
            filters.append("session_id=?")
            params.append(session_id)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        params.append(limit)
        rows = (
            self._get_conn()
            .execute(
                f"SELECT * FROM episodes {where} ORDER BY timestamp DESC LIMIT ?",
                tuple(params),
            )
            .fetchall()
        )
        self.touch_episodes([row["id"] for row in rows])
        return [Episode.from_db_row(dict(r)) for r in rows]

    def search_episodes(
        self,
        query: str,
        limit: int = 10,
        include_consolidated: bool = False,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Episode]:
        normalized_query = self._normalize_fts_query(query)
        if not normalized_query:
            return []
        consolidated_clause = (
            "AND e.is_consolidated=0" if not include_consolidated else ""
        )
        # FTS5: query the FTS table first, then JOIN back.
        rows = (
            self._get_conn()
            .execute(
                """SELECT e.* FROM episodes_fts
                JOIN episodes e ON e.id = episodes_fts.rowid
                WHERE episodes_fts MATCH ? AND e.user_id=?
                """
                + consolidated_clause
                + """
                ORDER BY rank
                LIMIT ?""",
                (normalized_query, user_id, limit),
            )
            .fetchall()
        )
        self.touch_episodes([row["id"] for row in rows])
        return [Episode.from_db_row(dict(r)) for r in rows]

    def get_salient_episodes(
        self,
        min_salience: float = 0.5,
        limit: int = 10,
        include_consolidated: bool = False,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Episode]:
        filters = ["user_id=?", "salience >= ?"]
        params: List[Any] = [user_id, min_salience]
        if not include_consolidated:
            filters.append("is_consolidated=0")
        params.append(limit)
        rows = (
            self._get_conn()
            .execute(
                f"SELECT * FROM episodes WHERE {' AND '.join(filters)} ORDER BY salience DESC LIMIT ?",
                tuple(params),
            )
            .fetchall()
        )
        self.touch_episodes([row["id"] for row in rows])
        return [Episode.from_db_row(dict(r)) for r in rows]

    def count_episodes(self, user_id: str = DEFAULT_USER_ID) -> int:
        return (
            self._get_conn()
            .execute("SELECT COUNT(*) FROM episodes WHERE user_id=?", (user_id,))
            .fetchone()[0]
        )

    def get_session_turn_count(
        self, session_id: str, user_id: str = DEFAULT_USER_ID
    ) -> int:
        return (
            self._get_conn()
            .execute(
                "SELECT COUNT(*) FROM episodes WHERE session_id=? AND user_id=?",
                (session_id, user_id),
            )
            .fetchone()[0]
        )

    def get_episodes_for_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Episode]:
        query = "SELECT * FROM episodes WHERE session_id=? AND user_id=? ORDER BY timestamp ASC"
        params: List[Any] = [session_id, user_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._get_conn().execute(query, tuple(params)).fetchall()
        self.touch_episodes([row["id"] for row in rows])
        return [Episode.from_db_row(dict(r)) for r in rows]

    def touch_episode(self, episode_id: int) -> None:
        self.touch_episodes([episode_id])

    def touch_episodes(self, episode_ids: List[int]) -> None:
        valid_ids = [episode_id for episode_id in episode_ids if episode_id]
        if not valid_ids:
            return
        placeholders = ",".join("?" for _ in valid_ids)
        now = now_epoch()
        try:
            with self._tx() as conn:
                conn.execute(
                    f"""UPDATE episodes
                       SET last_accessed_at=?, retrieval_count=retrieval_count + 1
                       WHERE id IN ({placeholders})""",
                    (now, *valid_ids),
                )
        except sqlite3.DatabaseError:
            logger.warning("Skipping episode touch due to database write failure")

    def count_unconsolidated_episodes(self, user_id: str = DEFAULT_USER_ID) -> int:
        return (
            self._get_conn()
            .execute(
                "SELECT COUNT(*) FROM episodes WHERE user_id=? AND is_consolidated=0 AND raw_preserved=1",
                (user_id,),
            )
            .fetchone()[0]
        )

    def get_unconsolidated_episodes(
        self,
        limit: int = 100,
        session_id: Optional[str] = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Episode]:
        query = "SELECT * FROM episodes WHERE user_id=? AND is_consolidated=0 AND raw_preserved=1"
        params: List[Any] = [user_id]
        if session_id:
            query += " AND session_id=?"
            params.append(session_id)
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        rows = self._get_conn().execute(query, tuple(params)).fetchall()
        return [Episode.from_db_row(dict(r)) for r in rows]

    def mark_episodes_consolidated(
        self, episode_ids: List[int], summary_episode_id: int
    ) -> int:
        valid_ids = [episode_id for episode_id in episode_ids if episode_id > 0]
        if not valid_ids or summary_episode_id <= 0:
            return 0
        placeholders = ",".join("?" for _ in valid_ids)
        with self._tx() as conn:
            cur = conn.execute(
                f"""UPDATE episodes
                   SET is_consolidated=1, consolidated_into=?
                   WHERE id IN ({placeholders})""",
                (summary_episode_id, *valid_ids),
            )
            return cur.rowcount

    # -- Conversation summaries ---------------------------------------------

    def insert_conversation_summary(self, summary: dict) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO conversation_summaries
                   (user_id, session_id, summary_text, summary_level, episode_range_start,
                    episode_range_end, episode_count, key_entities, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary.get("user_id", DEFAULT_USER_ID),
                    summary.get("session_id", ""),
                    summary.get("summary_text", ""),
                    summary.get("summary_level", "conversation"),
                    summary.get("episode_range_start"),
                    summary.get("episode_range_end"),
                    summary.get("episode_count", 0),
                    summary.get("key_entities", ""),
                    summary.get("created_at", now_epoch()),
                    summary.get("updated_at", now_epoch()),
                ),
            )
            return cur.lastrowid

    def list_conversation_summaries(
        self,
        limit: int = 10,
        session_id: Optional[str] = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[dict]:
        if session_id:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM conversation_summaries WHERE session_id=? AND user_id=? ORDER BY updated_at DESC LIMIT ?",
                    (session_id, user_id, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM conversation_summaries WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
                    (user_id, limit),
                )
                .fetchall()
            )
        return [dict(r) for r in rows]

    def search_conversation_summaries(
        self, query: str, limit: int = 5, user_id: str = DEFAULT_USER_ID
    ) -> List[dict]:
        like = f"%{query}%"
        rows = (
            self._get_conn()
            .execute(
                """SELECT * FROM conversation_summaries
                WHERE user_id=? AND (summary_text LIKE ? OR key_entities LIKE ? OR session_id LIKE ?)
                ORDER BY updated_at DESC LIMIT ?""",
                (user_id, like, like, like, limit),
            )
            .fetchall()
        )
        return [dict(r) for r in rows]

    # -- Facts ---------------------------------------------------------------

    def insert_fact(self, fact: Fact) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO facts
                   (user_id, subject_type, subject_id, predicate, object_text, confidence,
                    source_episode_id, first_seen, last_seen, status, superseded_by,
                    last_accessed_at, retrieval_count, valid_from, valid_to, contradiction_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact.user_id,
                    fact.subject_type,
                    fact.subject_id,
                    fact.predicate,
                    fact.object_text,
                    fact.confidence,
                    fact.source_episode_id,
                    fact.first_seen,
                    fact.last_seen,
                    fact.status,
                    fact.superseded_by,
                    fact.last_accessed_at,
                    fact.retrieval_count,
                    fact.valid_from if fact.valid_from is not None else fact.first_seen,
                    fact.valid_to,
                    fact.contradiction_status,
                ),
            )
            fact.id = cur.lastrowid
            return fact.id

    def get_active_facts(
        self,
        subject_type: Optional[str] = None,
        limit: int = 20,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Fact]:
        if subject_type:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM facts WHERE status='active' AND user_id=? AND subject_type=? ORDER BY confidence DESC LIMIT ?",
                    (user_id, subject_type, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM facts WHERE status='active' AND user_id=? ORDER BY confidence DESC LIMIT ?",
                    (user_id, limit),
                )
                .fetchall()
            )
        return [Fact.from_db_row(dict(r)) for r in rows]

    def get_fact(self, fact_id: int) -> Optional[Fact]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM facts WHERE id=?", (fact_id,))
            .fetchone()
        )
        return Fact.from_db_row(dict(row)) if row else None

    def get_facts_superseded_by(self, new_fact_id: int, limit: int = 20) -> List[Fact]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM facts WHERE superseded_by=? ORDER BY id DESC LIMIT ?",
                (new_fact_id, limit),
            )
            .fetchall()
        )
        return [Fact.from_db_row(dict(r)) for r in rows]

    def search_facts(
        self, query: str, limit: int = 10, user_id: str = DEFAULT_USER_ID
    ) -> List[Fact]:
        normalized_query = self._normalize_fts_query(query)
        if not normalized_query:
            return []
        # FTS5: query FTS table first to use `rank` pseudo-column correctly
        rows = (
            self._get_conn()
            .execute(
                """SELECT f.* FROM facts_fts
                JOIN facts f ON f.id = facts_fts.rowid
                WHERE facts_fts MATCH ? AND f.status='active' AND f.user_id=?
                ORDER BY rank
                LIMIT ?""",
                (normalized_query, user_id, limit),
            )
            .fetchall()
        )
        return [Fact.from_db_row(dict(r)) for r in rows]

    def update_fact_confidence(self, fact_id: int, confidence: float) -> None:
        now = now_epoch()
        with self._tx() as conn:
            conn.execute(
                "UPDATE facts SET confidence=?, last_seen=? WHERE id=?",
                (confidence, now, fact_id),
            )

    def bump_fact_last_seen(self, fact_id: int) -> None:
        now = now_epoch()
        with self._tx() as conn:
            conn.execute("UPDATE facts SET last_seen=? WHERE id=?", (now, fact_id))

    def supersede_fact(self, old_id: int, new_id: int) -> None:
        now = now_epoch()
        with self._tx() as conn:
            conn.execute(
                """UPDATE facts
                   SET status='superseded', superseded_by=?, valid_to=?, contradiction_status='superseded'
                   WHERE id=?""",
                (new_id, now, old_id),
            )

    def mark_fact_contradiction(self, old_id: int, new_id: int) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE facts SET contradiction_status='contradicted' WHERE id IN (?, ?)",
                (old_id, new_id),
            )

    def get_facts_by_predicate(
        self,
        predicate: str,
        status: str = "active",
        limit: int = 20,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Fact]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM facts WHERE predicate=? AND status=? AND user_id=? ORDER BY confidence DESC LIMIT ?",
                (predicate, status, user_id, limit),
            )
            .fetchall()
        )
        return [Fact.from_db_row(dict(r)) for r in rows]

    def find_similar_facts(
        self,
        text: str,
        predicate: Optional[str] = None,
        limit: int = 5,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Fact]:
        normalized_query = self._normalize_fts_query(text)
        if not normalized_query:
            return []
        # FTS5: query FTS table first to use `rank` pseudo-column correctly
        try:
            if predicate:
                rows = (
                    self._get_conn()
                    .execute(
                        """SELECT f.* FROM facts_fts
                        JOIN facts f ON f.id = facts_fts.rowid
                        WHERE facts_fts MATCH ? AND f.status='active' AND f.user_id=? AND f.predicate=?
                        ORDER BY rank LIMIT ?""",
                        (normalized_query, user_id, predicate, limit),
                    )
                    .fetchall()
                )
            else:
                rows = (
                    self._get_conn()
                    .execute(
                        """SELECT f.* FROM facts_fts
                        JOIN facts f ON f.id = facts_fts.rowid
                        WHERE facts_fts MATCH ? AND f.status='active' AND f.user_id=?
                        ORDER BY rank LIMIT ?""",
                        (normalized_query, user_id, limit),
                    )
                    .fetchall()
                )
            return [Fact.from_db_row(dict(r)) for r in rows]
        except sqlite3.OperationalError:
            return []

    @staticmethod
    def _normalize_fts_query(text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        filtered = [token for token in tokens if len(token) > 1]
        return " OR ".join(filtered)

    def count_facts(
        self, status: str = "active", user_id: str = DEFAULT_USER_ID
    ) -> int:
        return (
            self._get_conn()
            .execute(
                "SELECT COUNT(*) FROM facts WHERE status=? AND user_id=?",
                (status, user_id),
            )
            .fetchone()[0]
        )

    def decay_stale_facts(
        self, days_threshold: float = 60.0, decay_rate: float = 0.05
    ) -> int:
        cutoff = now_epoch() - (days_threshold * 86400)
        with self._tx() as conn:
            cur = conn.execute(
                """UPDATE facts
                   SET confidence = MAX(0.1, confidence - ?)
                   WHERE status='active' AND last_seen < ? AND confidence > 0.1""",
                (decay_rate, cutoff),
            )
            return cur.rowcount

    # -- Evidence Traces ----------------------------------------------------

    def get_facts_by_episode(
        self, episode_id: int, user_id: str = DEFAULT_USER_ID
    ) -> List[Fact]:
        """Get all facts that originated from a specific episode."""
        rows = (
            self._get_conn()
            .execute(
                """SELECT * FROM facts
                   WHERE source_episode_id=? AND user_id=?
                   ORDER BY confidence DESC""",
                (episode_id, user_id),
            )
            .fetchall()
        )
        return [Fact.from_db_row(dict(r)) for r in rows]

    def get_evidence_for_fact(
        self, fact_id: int
    ) -> Optional[dict]:
        """Get the evidence trail for a fact: which episode it came from,
        with episode metadata (summary, valence, salience, topics)."""
        row = (
            self._get_conn()
            .execute(
                """SELECT f.id, f.predicate, f.object_text, f.confidence,
                          f.source_episode_id, f.first_seen, f.last_seen, f.status,
                          e.summary, e.valence, e.arousal, e.topics, e.entities,
                          e.user_text, e.assistant_text, e.timestamp
                   FROM facts f
                   LEFT JOIN episodes e ON e.id = f.source_episode_id
                   WHERE f.id=?""",
                (fact_id,),
            )
            .fetchone()
        )
        if not row:
            return None
        return {
            "fact_id": row[0],
            "predicate": row[1],
            "object_text": row[2],
            "confidence": row[3],
            "source_episode_id": row[4],
            "first_seen": row[5],
            "last_seen": row[6],
            "status": row[7],
            "episode_summary": row[8],
            "episode_valence": row[9],
            "episode_arousal": row[10],
            "episode_topics": row[11],
            "episode_entities": row[12],
            "episode_user_text": row[13],
            "episode_assistant_text": row[14],
            "episode_timestamp": row[15],
        }

    def get_fact_evidence_chain(
        self, fact_id: int
    ) -> Optional[dict]:
        """Get the full evidence chain for a fact: fact → episode → superseding facts.
        Think of this as the 'provenance' of a fact."""
        evidence = self.get_evidence_for_fact(fact_id)
        if not evidence:
            return None
        # Get superseded facts (history leading to this fact)
        superseded = self.get_facts_superseded_by(fact_id)
        evidence["superseded_facts"] = [
            {
                "id": f.id,
                "object_text": f.object_text,
                "confidence": f.confidence,
                "source_episode_id": f.source_episode_id,
            }
            for f in superseded
        ]
        return evidence

    def get_evidence_summary(
        self, user_id: str = DEFAULT_USER_ID, limit: int = 20
    ) -> List[dict]:
        """Get a summary of fact evidence: which facts come from which episodes,
        with aggregate stats. Useful for debugging fact quality."""
        rows = (
            self._get_conn()
            .execute(
                """SELECT f.source_episode_id AS eid,
                         COUNT(*) AS fact_count,
                         AVG(f.confidence) AS avg_confidence,
                         MIN(f.confidence) AS min_confidence,
                         MAX(f.confidence) AS max_confidence,
                         GROUP_CONCAT(DISTINCT f.predicate) AS predicates
                   FROM facts f
                   WHERE f.user_id=? AND f.source_episode_id IS NOT NULL
                   GROUP BY f.source_episode_id
                   ORDER BY fact_count DESC, avg_confidence DESC
                   LIMIT ?""",
                (user_id, limit),
            )
            .fetchall()
        )
        results = []
        for row in rows:
            results.append({
                "episode_id": row[0],
                "fact_count": row[1],
                "avg_confidence": round(row[2], 3) if row[2] else 0,
                "min_confidence": round(row[3], 3) if row[3] else 0,
                "max_confidence": round(row[4], 3) if row[4] else 0,
                "predicates": row[5].split(",") if row[5] else [],
            })
        return results

    def get_orphaned_facts(
        self, user_id: str = DEFAULT_USER_ID
    ) -> List[Fact]:
        """Find facts that have a source_episode_id but the episode no longer exists.
        Useful for cleanup."""
        rows = (
            self._get_conn()
            .execute(
                """SELECT * FROM facts f
                   WHERE f.user_id=?
                     AND f.source_episode_id IS NOT NULL
                     AND f.source_episode_id > 0
                     AND NOT EXISTS (
                         SELECT 1 FROM episodes e WHERE e.id = f.source_episode_id
                     )
                   ORDER BY f.confidence DESC""",
                (user_id,),
            )
            .fetchall()
        )
        return [Fact.from_db_row(dict(r)) for r in rows]

    # -- Open Loops ----------------------------------------------------------

    def insert_open_loop(self, loop: OpenLoop) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO open_loops
                   (user_id, kind, text, due_hint, status, source_episode_id, created_at,
                    resolved_at, last_accessed_at, resolution, resolved_by_episode_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    loop.user_id,
                    loop.kind,
                    loop.text,
                    loop.due_hint,
                    loop.status,
                    loop.source_episode_id,
                    loop.created_at,
                    loop.resolved_at,
                    loop.last_accessed_at,
                    loop.resolution,
                    loop.resolved_by_episode_id,
                ),
            )
            loop.id = cur.lastrowid
            return loop.id

    def get_open_loops(
        self, limit: int = 10, user_id: str = DEFAULT_USER_ID
    ) -> List[OpenLoop]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM open_loops WHERE status='open' AND user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            .fetchall()
        )
        return [OpenLoop.from_db_row(dict(r)) for r in rows]

    def resolve_loop(
        self,
        loop_id: int,
        resolution: str = "",
        resolved_by_episode_id: Optional[int] = None,
    ) -> None:
        now = now_epoch()
        with self._tx() as conn:
            conn.execute(
                """UPDATE open_loops
                   SET status='resolved', resolved_at=?, resolution=?, resolved_by_episode_id=?
                   WHERE id=?""",
                (now, resolution, resolved_by_episode_id, loop_id),
            )

    def expire_old_loops(self, days_threshold: float = 14.0) -> int:
        cutoff = now_epoch() - (days_threshold * 86400)
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE open_loops SET status='expired' WHERE status='open' AND created_at < ?",
                (cutoff,),
            )
            return cur.rowcount

    def count_open_loops(self, user_id: str = DEFAULT_USER_ID) -> int:
        return (
            self._get_conn()
            .execute(
                "SELECT COUNT(*) FROM open_loops WHERE status='open' AND user_id=?",
                (user_id,),
            )
            .fetchone()[0]
        )

    def search_open_loops(self, text: str, limit: int = 5) -> List[OpenLoop]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM open_loops WHERE status='open' AND text LIKE ? LIMIT ?",
                (f"%{text}%", limit),
            )
            .fetchall()
        )
        return [OpenLoop.from_db_row(dict(r)) for r in rows]

    def get_recently_resolved_loops(
        self, days_window: float = 7.0, limit: int = 5, user_id: str = DEFAULT_USER_ID
    ) -> List[OpenLoop]:
        """Get loops that were resolved within the given time window."""
        cutoff = now_epoch() - (days_window * 86400)
        rows = (
            self._get_conn()
            .execute(
                """SELECT * FROM open_loops 
                   WHERE status='resolved' AND resolved_at >= ? AND user_id=?
                   ORDER BY resolved_at DESC LIMIT ?""",
                (cutoff, user_id, limit),
            )
            .fetchall()
        )
        return [OpenLoop.from_db_row(dict(r)) for r in rows]

    def get_stale_open_loops(
        self, days_threshold: float = 14.0, limit: int = 10, user_id: str = DEFAULT_USER_ID
    ) -> List[OpenLoop]:
        """Get open loops that haven't been updated in N days (stale)."""
        cutoff = now_epoch() - (days_threshold * 86400)
        rows = (
            self._get_conn()
            .execute(
                """SELECT * FROM open_loops 
                   WHERE status='open' AND created_at < ? AND user_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (cutoff, user_id, limit),
            )
            .fetchall()
        )
        return [OpenLoop.from_db_row(dict(r)) for r in rows]

    def get_active_open_loops(
        self, days_threshold: float = 14.0, limit: int = 10, user_id: str = DEFAULT_USER_ID
    ) -> List[OpenLoop]:
        """Get open loops that are still fresh (not stale)."""
        cutoff = now_epoch() - (days_threshold * 86400)
        rows = (
            self._get_conn()
            .execute(
                """SELECT * FROM open_loops 
                   WHERE status='open' AND created_at >= ? AND user_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (cutoff, user_id, limit),
            )
            .fetchall()
        )
        return [OpenLoop.from_db_row(dict(r)) for r in rows]

    # -- Reflections ---------------------------------------------------------

    def insert_reflection(self, ref: Reflection) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO reflections
                   (user_id, kind, text, confidence, source_episode_id, created_at,
                    last_reinforced, reinforcement_count, last_accessed_at, retrieval_count,
                    promotion_status, promoted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ref.user_id,
                    ref.kind,
                    ref.text,
                    ref.confidence,
                    ref.source_episode_id,
                    ref.created_at,
                    ref.last_reinforced,
                    ref.reinforcement_count,
                    ref.last_accessed_at,
                    ref.retrieval_count,
                    ref.promotion_status,
                    ref.promoted_at,
                ),
            )
            ref.id = cur.lastrowid
            return ref.id

    def get_reflections(
        self,
        kind: Optional[str] = None,
        limit: int = 10,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Reflection]:
        if kind:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM reflections WHERE kind=? AND user_id=? ORDER BY confidence DESC LIMIT ?",
                    (kind, user_id, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM reflections WHERE user_id=? ORDER BY confidence DESC LIMIT ?",
                    (user_id, limit),
                )
                .fetchall()
            )
        return [Reflection.from_db_row(dict(r)) for r in rows]

    def get_reflection(self, reflection_id: int) -> Optional[Reflection]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM reflections WHERE id=?", (reflection_id,))
            .fetchone()
        )
        return Reflection.from_db_row(dict(row)) if row else None

    def search_reflections(
        self, query: str, limit: int = 10, user_id: str = DEFAULT_USER_ID
    ) -> List[Reflection]:
        normalized_query = self._normalize_fts_query(query)
        if not normalized_query:
            return []
        # Query FTS table directly, then join back to reflections for full data.
        rows = (
            self._get_conn()
            .execute(
                """SELECT r.* FROM reflections_fts
                JOIN reflections r ON r.id = reflections_fts.rowid
                WHERE reflections_fts MATCH ? AND r.user_id=?
                ORDER BY rank
                LIMIT ?""",
                (normalized_query, user_id, limit),
            )
            .fetchall()
        )
        return [Reflection.from_db_row(dict(r)) for r in rows]

    def reinforce_reflection(
        self, reflection_id: int, confidence_boost: float = 0.1
    ) -> None:
        now = now_epoch()
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
        self,
        min_confidence: float = 0.5,
        limit: int = 10,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Reflection]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM reflections WHERE confidence >= ? AND user_id=? ORDER BY confidence DESC LIMIT ?",
                (min_confidence, user_id, limit),
            )
            .fetchall()
        )
        return [Reflection.from_db_row(dict(r)) for r in rows]

    def decay_stale_reflections(
        self, days_threshold: float = 30.0, decay_rate: float = 0.05
    ) -> int:
        cutoff = now_epoch() - (days_threshold * 86400)
        with self._tx() as conn:
            cur = conn.execute(
                """UPDATE reflections
                   SET confidence = MAX(0.05, confidence - ?)
                   WHERE last_reinforced < ? AND confidence > 0.05""",
                (decay_rate, cutoff),
            )
            return cur.rowcount

    def get_identity_deltas(
        self, applied: Optional[bool] = None, limit: int = 10
    ) -> List[IdentityDelta]:
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
        return [IdentityDelta.from_db_row(dict(r)) for r in rows]

    def get_identity_delta_by_id(self, delta_id: int) -> Optional[IdentityDelta]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM identity_deltas WHERE id=?", (delta_id,))
            .fetchone()
        )
        return IdentityDelta.from_db_row(dict(row)) if row else None

    # -- Relationship State --------------------------------------------------

    def get_relationship(self, user_id: str = DEFAULT_USER_ID) -> RelationshipState:
        row = (
            self._get_conn()
            .execute("SELECT * FROM relationship_state WHERE user_id=?", (user_id,))
            .fetchone()
        )
        if row:
            return RelationshipState.from_db_row(dict(row))
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
                    rs.last_updated,
                    rs.total_turns,
                ),
            )

    # -- Identity Deltas -----------------------------------------------------

    def insert_identity_delta(self, delta: IdentityDelta) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO identity_deltas
                   (user_id, text, confidence, source_episode_id, created_at, applied)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    delta.user_id,
                    delta.text,
                    delta.confidence,
                    delta.source_episode_id,
                    delta.created_at,
                    int(delta.applied),
                ),
            )
            delta.id = cur.lastrowid
            return delta.id

    def mark_identity_delta_applied(self, delta_id: int) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE identity_deltas SET applied=1 WHERE id=?", (delta_id,)
            )
            return cur.rowcount > 0

    def delete_identity_delta(self, delta_id: int) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM identity_deltas WHERE id=?", (delta_id,))
            return cur.rowcount > 0

    def reject_identity_delta(self, delta_id: int) -> bool:
        return self.delete_identity_delta(delta_id)

    # -- Entity Links --------------------------------------------------------

    def insert_link(
        self,
        src_type: str,
        src_id: int,
        dst_type: str,
        dst_id: int,
        relation: str,
        weight: float = 1.0,
        user_id: str = DEFAULT_USER_ID,
    ) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO entity_links
                   (user_id, src_type, src_id, dst_type, dst_id, relation, weight)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, src_type, src_id, dst_type, dst_id, relation, weight),
            )
            return cur.lastrowid

    def get_links_from(
        self,
        src_type: str,
        src_id: int,
        relation: str = None,
        limit: Optional[int] = 20,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[dict]:
        query = (
            "SELECT dst_type, dst_id, relation, weight FROM entity_links "
            "WHERE src_type=? AND src_id=? AND user_id=?"
        )
        params: List[Any] = [src_type, src_id, user_id]
        if relation:
            query += " AND relation=?"
            params.append(relation)
        query += " ORDER BY weight DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
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
        limit: Optional[int] = 20,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[dict]:
        query = (
            "SELECT src_type, src_id, relation, weight FROM entity_links "
            "WHERE dst_type=? AND dst_id=? AND user_id=?"
        )
        params: List[Any] = [dst_type, dst_id, user_id]
        if relation:
            query += " AND relation=?"
            params.append(relation)
        query += " ORDER BY weight DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
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
        user_id: str = DEFAULT_USER_ID,
    ) -> bool:
        row = (
            self._get_conn()
            .execute(
                """SELECT 1 FROM entity_links
                   WHERE src_type=? AND src_id=? AND dst_type=? AND dst_id=? AND relation=? AND user_id=?
                   LIMIT 1""",
                (src_type, src_id, dst_type, dst_id, relation, user_id),
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
        timestamp: Optional[float] = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> int:
        ts = timestamp if timestamp is not None else now_epoch()
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO emotion_log
                   (user_id, episode_id, session_id, timestamp, frustration, warmth, humor,
                    hostility, gratitude, anxiety, excitement, trust_signal,
                    valence, arousal, dominant_emotion, is_sarcastic)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
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
        self,
        limit: int = 10,
        session_id: Optional[str] = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[AffectSignal]:
        if session_id:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM emotion_log WHERE session_id=? AND user_id=? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, user_id, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    "SELECT * FROM emotion_log WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit),
                )
                .fetchall()
            )
        return [AffectSignal.from_db_row(dict(r)) for r in rows]

    def get_emotion_for_episode(self, episode_id: int) -> Optional[AffectSignal]:
        row = (
            self._get_conn()
            .execute("SELECT * FROM emotion_log WHERE episode_id=?", (episode_id,))
            .fetchone()
        )
        return AffectSignal.from_db_row(dict(row)) if row else None

    def get_emotional_trajectory(
        self,
        limit: int = 20,
        session_id: Optional[str] = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[dict]:
        if session_id:
            rows = (
                self._get_conn()
                .execute(
                    """SELECT timestamp, valence, arousal, dominant_emotion
                       FROM emotion_log WHERE session_id=? AND user_id=?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (session_id, user_id, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._get_conn()
                .execute(
                    """SELECT timestamp, valence, arousal, dominant_emotion
                       FROM emotion_log WHERE user_id=? ORDER BY timestamp DESC LIMIT ?""",
                    (user_id, limit),
                )
                .fetchall()
            )
        return [
            {
                "timestamp": float(r["timestamp"]),
                "valence": r["valence"],
                "arousal": r["arousal"],
                "emotion": r["dominant_emotion"],
            }
            for r in rows
        ]

    def get_affect_baseline(self, user_id: str = DEFAULT_USER_ID) -> AffectBaseline:
        row = (
            self._get_conn()
            .execute("SELECT * FROM affect_baselines WHERE user_id=?", (user_id,))
            .fetchone()
        )
        if not row:
            return AffectBaseline(user_id=user_id)
        return AffectBaseline(
            user_id=row["user_id"],
            baseline_frustration=row["baseline_frustration"],
            baseline_warmth=row["baseline_warmth"],
            baseline_humor=row["baseline_humor"],
            baseline_hostility=row["baseline_hostility"],
            baseline_gratitude=row["baseline_gratitude"],
            baseline_anxiety=row["baseline_anxiety"],
            baseline_excitement=row["baseline_excitement"],
            baseline_trust_signal=row["baseline_trust_signal"],
            sample_count=row["sample_count"],
            ema_alpha=row["ema_alpha"],
            created_at=_as_epoch(row["created_at"]) or 0.0,
            updated_at=_as_epoch(row["updated_at"]) or 0.0,
        )

    def upsert_affect_baseline(self, baseline: AffectBaseline) -> None:
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO affect_baselines
                   (user_id, baseline_frustration, baseline_warmth, baseline_humor,
                    baseline_hostility, baseline_gratitude, baseline_anxiety,
                    baseline_excitement, baseline_trust_signal, sample_count,
                    ema_alpha, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                    baseline_frustration=excluded.baseline_frustration,
                    baseline_warmth=excluded.baseline_warmth,
                    baseline_humor=excluded.baseline_humor,
                    baseline_hostility=excluded.baseline_hostility,
                    baseline_gratitude=excluded.baseline_gratitude,
                    baseline_anxiety=excluded.baseline_anxiety,
                    baseline_excitement=excluded.baseline_excitement,
                    baseline_trust_signal=excluded.baseline_trust_signal,
                    sample_count=excluded.sample_count,
                    ema_alpha=excluded.ema_alpha,
                    updated_at=excluded.updated_at""",
                (
                    baseline.user_id,
                    baseline.baseline_frustration,
                    baseline.baseline_warmth,
                    baseline.baseline_humor,
                    baseline.baseline_hostility,
                    baseline.baseline_gratitude,
                    baseline.baseline_anxiety,
                    baseline.baseline_excitement,
                    baseline.baseline_trust_signal,
                    baseline.sample_count,
                    baseline.ema_alpha,
                    baseline.created_at,
                    baseline.updated_at,
                ),
            )

    # -- Lossless context engine storage ------------------------------------

    def ensure_context_conversation(
        self, conversation_id: str, *, user_id: str = DEFAULT_USER_ID
    ) -> None:
        now = now_epoch()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO context_conversations
                   (conversation_id, user_id, created_at, updated_at, status)
                   VALUES (?, ?, ?, ?, 'active')
                   ON CONFLICT(conversation_id) DO UPDATE SET
                     user_id=excluded.user_id,
                     updated_at=excluded.updated_at""",
                (conversation_id, user_id, now, now),
            )

    def map_session_alias(
        self,
        hermes_session_id: str,
        conversation_id: str,
        *,
        lineage_parent_session_id: Optional[str] = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO context_session_aliases
                   (hermes_session_id, conversation_id, lineage_parent_session_id, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(hermes_session_id) DO UPDATE SET
                     conversation_id=excluded.conversation_id,
                     lineage_parent_session_id=excluded.lineage_parent_session_id""",
                (
                    hermes_session_id,
                    conversation_id,
                    lineage_parent_session_id,
                    now_epoch(),
                ),
            )

    def get_context_conversation_id(self, hermes_session_id: str) -> Optional[str]:
        row = (
            self._get_conn()
            .execute(
                "SELECT conversation_id FROM context_session_aliases WHERE hermes_session_id=?",
                (hermes_session_id,),
            )
            .fetchone()
        )
        return row["conversation_id"] if row else None

    def get_next_context_seq(self, conversation_id: str) -> int:
        row = (
            self._get_conn()
            .execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM context_messages WHERE conversation_id=?",
                (conversation_id,),
            )
            .fetchone()
        )
        return int(row["max_seq"] or 0) + 1

    def archive_context_messages(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        *,
        start_seq: Optional[int] = None,
    ) -> tuple[int, int]:
        if not messages:
            raise ValueError("Cannot archive empty messages")

        seq = start_seq or self.get_next_context_seq(conversation_id)
        first_seq = seq
        created_at = now_epoch()
        with self._tx() as conn:
            for msg in messages:
                raw_json = json.dumps(msg, ensure_ascii=False, sort_keys=True)
                text_content = self._message_text_content(msg)
                conn.execute(
                    """INSERT OR IGNORE INTO context_messages
                       (conversation_id, seq, role, raw_json, text_content, content_hash, token_estimate, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        conversation_id,
                        seq,
                        msg.get("role", "user"),
                        raw_json,
                        text_content,
                        sha256(raw_json.encode("utf-8")).hexdigest(),
                        max(1, len(raw_json) // 4),
                        created_at,
                    ),
                )
                seq += 1
        return first_seq, seq - 1

    def create_context_span(
        self,
        conversation_id: str,
        *,
        start_seq: int,
        end_seq: int,
        kind: str = "raw",
        checkpoint_id: Optional[str] = None,
    ) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO context_spans
                   (conversation_id, start_seq, end_seq, kind, checkpoint_id, archived_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conversation_id, start_seq, end_seq, kind, checkpoint_id, now_epoch()),
            )
            return int(cur.lastrowid)

    def insert_context_ref(
        self,
        conversation_id: str,
        *,
        ref_id: str,
        ref_type: str,
        label: str,
        payload: Dict[str, Any],
        source_span_id: Optional[int],
        salience: float = 0.0,
        open_state: str = "open",
    ) -> None:
        created_at = now_epoch()
        with self._tx() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO context_refs
                   (ref_id, conversation_id, ref_type, label, payload_json, source_span_id, salience, open_state, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM context_refs WHERE ref_id=?), ?))""",
                (
                    ref_id,
                    conversation_id,
                    ref_type,
                    label,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    source_span_id,
                    salience,
                    open_state,
                    ref_id,
                    created_at,
                ),
            )

    def batch_insert_context_refs(
        self,
        conversation_id: str,
        refs: List[Dict[str, Any]],
    ) -> None:
        """Batch insert context refs in a single transaction (Fix #2: streaming refs).

        Replaces N individual INSERT calls with one bulk EXECUTEMANY,
        reducing WAL contention and FTS5 trigger overhead.
        """
        if not refs:
            return
        created_at = now_epoch()
        rows = []
        for ref in refs:
            rows.append((
                ref["ref_id"],
                conversation_id,
                ref["ref_type"],
                ref["label"],
                json.dumps(ref["payload"], ensure_ascii=False, sort_keys=True),
                ref.get("source_span_id"),
                ref.get("salience", 0.0),
                ref.get("open_state", "open"),
                ref["ref_id"],
                created_at,
            ))

        with self._tx() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO context_refs
                   (ref_id, conversation_id, ref_type, label, payload_json, source_span_id, salience, open_state, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM context_refs WHERE ref_id=?), ?))""",
                rows,
            )

    def insert_context_checkpoint(
        self,
        conversation_id: str,
        *,
        checkpoint_id: str,
        replaced_start_seq: int,
        replaced_end_seq: int,
        export_text: str,
        source_span_ids: List[int],
        hot_ref_ids: List[str],
    ) -> None:
        token_estimate = max(1, len(export_text) // 4)
        now = now_epoch()
        with self._tx() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO context_checkpoints
                   (checkpoint_id, conversation_id, replaced_start_seq, replaced_end_seq,
                    export_text, export_token_estimate, source_span_ids_json,
                    hot_ref_ids_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    checkpoint_id,
                    conversation_id,
                    replaced_start_seq,
                    replaced_end_seq,
                    export_text,
                    token_estimate,
                    json.dumps(source_span_ids),
                    json.dumps(hot_ref_ids),
                    now,
                ),
            )
            conn.execute(
                "UPDATE context_conversations SET active_checkpoint_id=?, updated_at=? WHERE conversation_id=?",
                (checkpoint_id, now, conversation_id),
            )

    def get_active_context_checkpoint(
        self, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        row = (
            self._get_conn()
            .execute(
                """SELECT c.* FROM context_checkpoints c
               JOIN context_conversations conv ON conv.active_checkpoint_id = c.checkpoint_id
               WHERE conv.conversation_id=?""",
                (conversation_id,),
            )
            .fetchone()
        )
        return dict(row) if row else None

    def search_context_refs(
        self, conversation_id: str, query: str, limit: int = 8
    ) -> List[Dict[str, Any]]:
        normalized_query = self._normalize_fts_query(query)
        if not normalized_query:
            return []
        rows = (
            self._get_conn()
            .execute(
                """SELECT r.* FROM context_refs r
               JOIN context_refs_fts fts ON r.rowid = fts.rowid
               WHERE context_refs_fts MATCH ? AND r.conversation_id=?
               ORDER BY r.salience DESC, rank
               LIMIT ?""",
                (normalized_query, conversation_id, limit),
            )
            .fetchall()
        )
        return [dict(row) for row in rows]

    def search_context_messages(
        self, conversation_id: str, query: str, limit: int = 8
    ) -> List[Dict[str, Any]]:
        normalized_query = self._normalize_fts_query(query)
        if not normalized_query:
            return []
        rows = (
            self._get_conn()
            .execute(
                """SELECT m.* FROM context_messages m
               JOIN context_messages_fts fts ON m.id = fts.rowid
               WHERE context_messages_fts MATCH ? AND m.conversation_id=?
               ORDER BY m.seq DESC
               LIMIT ?""",
                (normalized_query, conversation_id, limit),
            )
            .fetchall()
        )
        return [dict(row) for row in rows]

    def get_context_messages_by_seq_range(
        self, conversation_id: str, start_seq: int, end_seq: int
    ) -> List[Dict[str, Any]]:
        rows = (
            self._get_conn()
            .execute(
                """SELECT * FROM context_messages
               WHERE conversation_id=? AND seq BETWEEN ? AND ?
               ORDER BY seq ASC""",
                (conversation_id, start_seq, end_seq),
            )
            .fetchall()
        )
        return [dict(row) for row in rows]

    @staticmethod
    def _message_text_content(message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str) and text:
                        chunks.append(text)
            return "\n".join(chunks)
        return str(content or "")


    # ── OCEAN Personality Traits (Big Five) ──────────────────────────────────

    def upsert_ocean_profile(
        self,
        user_id: str,
        openness: float,
        conscientiousness: float,
        extraversion: float,
        agreeableness: float,
        neuroticism: float,
        confidence: float,
        turn_count: int,
    ) -> None:
        """Upsert an OCEAN personality profile for a user."""
        now = now_epoch()
        with self._tx() as conn:
            cur = conn.execute(
                """SELECT id FROM ocean_profiles WHERE user_id=?""",
                (user_id,),
            )
            row = cur.fetchone()
            if row:
                conn.execute(
                    """UPDATE ocean_profiles
                       SET openness=?, conscientiousness=?, extraversion=?,
                           agreeableness=?, neuroticism=?, confidence=?,
                           turn_count=?, updated_at=?
                       WHERE user_id=?""",
                    (openness, conscientiousness, extraversion,
                     agreeableness, neuroticism, confidence,
                     turn_count, now, user_id),
                )
            else:
                conn.execute(
                    """INSERT INTO ocean_profiles
                       (user_id, openness, conscientiousness, extraversion,
                        agreeableness, neuroticism, confidence, turn_count, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, openness, conscientiousness, extraversion,
                     agreeableness, neuroticism, confidence, turn_count, now),
                )

    def get_ocean_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the current OCEAN personality profile for a user."""
        row = (
            self._get_conn()
            .execute(
                """SELECT * FROM ocean_profiles WHERE user_id=?""",
                (user_id,),
            )
            .fetchone()
        )
        return dict(row) if row else None

    def get_ocean_profiles_summary(self) -> List[Dict[str, Any]]:
        """Get OCEAN profiles for status display."""
        rows = self._get_conn().execute(
            """SELECT user_id, openness, conscientiousness, extraversion,
                      agreeableness, neuroticism, confidence, turn_count, updated_at
               FROM ocean_profiles
               ORDER BY updated_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Semantic Embeddings ───────────────────────────────────────────────────

    def insert_embedding(
        self,
        entity_id: int,
        entity_type: str,
        embedding_vector: bytes,
        user_id: str = DEFAULT_USER_ID,
    ) -> int:
        """Store a vector embedding for an entity (episode, fact, etc.)."""
        now = now_epoch()
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO embeddings
                   (entity_id, entity_type, embedding_vector, created_at, user_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (entity_id, entity_type, embedding_vector, now, user_id),
            )
            return cur.lastrowid

    def get_embedding(
        self,
        entity_type: str,
        entity_id: int,
        user_id: str = DEFAULT_USER_ID,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the latest embedding for a given entity."""
        row = (
            self._get_conn()
            .execute(
                """SELECT entity_id, entity_type, embedding_vector, created_at, user_id
                   FROM embeddings
                   WHERE entity_type=? AND entity_id=? AND user_id=?
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (entity_type, entity_id, user_id),
            )
            .fetchone()
        )
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    def delete_embedding(
        self,
        entity_type: str,
        entity_id: int,
        user_id: str = DEFAULT_USER_ID,
    ) -> int:
        """Delete embeddings for a specific entity."""
        with self._tx() as conn:
            cur = conn.execute(
                """DELETE FROM embeddings WHERE entity_type=? AND entity_id=? AND user_id=?""",
                (entity_type, entity_id, user_id),
            )
            return cur.rowcount

    # ── Database Optimization ──────────────────────────────────────────────

    def optimize_database(self) -> Dict[str, Any]:
        """Run full DB optimization: vacuum, reindex, analyze, and add compound indexes.
        
        Returns a summary dict of what was done.
        """
        result = {"vacuumed": False, "reindexed": False, "analyzed": False, "indexes_added": []}
        conn = self._get_conn()
        
        # Add compound indexes for common query patterns
        compound_indexes = [
            # Facts: most common lookup pattern (user + status + last_accessed)
            "CREATE INDEX IF NOT EXISTS idx_facts_user_status_accessed ON facts(user_id, status, last_accessed_at)",
            # Facts: temporal decay queries (user + status + valid_to)
            "CREATE INDEX IF NOT EXISTS idx_facts_user_status_decay ON facts(user_id, status, valid_to)",
            # Episodes: user + salience + timestamp (for recent salient episodes)
            "CREATE INDEX IF NOT EXISTS idx_episodes_user_salience_ts ON episodes(user_id, salience, timestamp)",
            # Open loops: user + status + updated_at (for active loops)
            "CREATE INDEX IF NOT EXISTS idx_loops_user_status_updated ON open_loops(user_id, status, updated_at)",
            # Reflections: user + promotion_status + updated_at
            "CREATE INDEX IF NOT EXISTS idx_reflections_user_promotion_updated ON reflections(user_id, promotion_status, updated_at)",
        ]
        
        for idx_sql in compound_indexes:
            try:
                conn.execute(idx_sql)
                result["indexes_added"].append(idx_sql.split("(")[0].split()[-1])
            except Exception:
                pass
        
        # VACUUM: recompacts the DB file
        try:
            conn.execute("VACUUM")
            result["vacuumed"] = True
        except Exception:
            pass
        
        # REINDEX: rebuilds all B-tree indexes
        try:
            conn.execute("REINDEX")
            result["reindexed"] = True
        except Exception:
            pass
        
        # ANALYZE: updates table statistics for query planner
        try:
            conn.execute("ANALYZE")
            result["analyzed"] = True
        except Exception:
            conn.commit()
        
        return result

    def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics: row counts, file size, index usage."""
        conn = self._get_conn()
        
        stats = {}
        
        # Table row counts
        for table in ["episodes", "facts", "open_loops", "reflections", "entity_links", 
                       "emotion_log", "conversation_summaries", "ocean_profiles"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f"{table}_count"] = count
            except Exception:
                stats[f"{table}_count"] = 0
        
        # DB file size
        try:
            import os
            db_path = conn.execute("PRAGMA database_list").fetchone()[2]
            if db_path and os.path.exists(db_path):
                stats["db_file_size_bytes"] = os.path.getsize(db_path)
        except Exception:
            pass
        
        # Schema version
        stats["schema_version"] = conn.execute("PRAGMA user_version").fetchone()[0]
        
        # Page count and free pages
        stats["page_count"] = conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
        
        return stats
