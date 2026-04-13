import sqlite3
import time
import os
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test_kortex.db")


@pytest.fixture
def kortex_db(tmp_db_path):
    from kortex.db import KortexDB

    db = KortexDB(tmp_db_path)
    yield db
    db.close()


@pytest.fixture
def kortex_config(tmp_db_path):
    from kortex.config import KortexConfig

    return KortexConfig(db_path=tmp_db_path)


@pytest.fixture
def ingestor(kortex_db):
    from kortex.ingest import Ingestor

    return Ingestor(kortex_db)


@pytest.fixture
def recall(kortex_db, kortex_config):
    from kortex.recall import Recall

    return Recall(kortex_db, kortex_config)


@pytest.fixture
def populated_v2_db_path(tmp_path):
    db_path = tmp_path / "populated_v2.db"
    conn = sqlite3.connect(db_path)

    conn.executescript(
        """
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL,
            user_text TEXT NOT NULL DEFAULT '',
            assistant_text TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            salience REAL NOT NULL DEFAULT 0.0,
            valence INTEGER NOT NULL DEFAULT 0,
            arousal REAL NOT NULL DEFAULT 0.0,
            topics TEXT NOT NULL DEFAULT '',
            entities TEXT NOT NULL DEFAULT '',
            is_consolidated INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_type TEXT NOT NULL DEFAULT 'user',
            subject_id TEXT NOT NULL DEFAULT '',
            predicate TEXT NOT NULL DEFAULT '',
            object_text TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.5,
            source_episode_id INTEGER,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            superseded_by INTEGER
        );

        CREATE TABLE open_loops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL DEFAULT 'commitment',
            text TEXT NOT NULL DEFAULT '',
            due_hint TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'open',
            source_episode_id INTEGER,
            created_at TEXT NOT NULL,
            resolved_at TEXT
        );

        CREATE TABLE reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL DEFAULT 'pattern',
            text TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.3,
            source_episode_id INTEGER,
            created_at TEXT NOT NULL,
            last_reinforced TEXT NOT NULL,
            reinforcement_count INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE relationship_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default' UNIQUE,
            warmth REAL NOT NULL DEFAULT 0.5,
            trust REAL NOT NULL DEFAULT 0.5,
            tension REAL NOT NULL DEFAULT 0.0,
            familiarity REAL NOT NULL DEFAULT 0.0,
            humor REAL NOT NULL DEFAULT 0.0,
            formality REAL NOT NULL DEFAULT 0.5,
            volatility REAL NOT NULL DEFAULT 0.0,
            last_updated TEXT NOT NULL,
            total_turns INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE identity_deltas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.3,
            source_episode_id INTEGER,
            created_at TEXT NOT NULL,
            applied INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE entity_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_type TEXT NOT NULL,
            src_id INTEGER NOT NULL,
            dst_type TEXT NOT NULL,
            dst_id INTEGER NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0
        );

        CREATE TABLE emotion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER,
            session_id TEXT NOT NULL DEFAULT '',
            timestamp TEXT NOT NULL,
            frustration REAL NOT NULL DEFAULT 0.0,
            warmth REAL NOT NULL DEFAULT 0.0,
            humor REAL NOT NULL DEFAULT 0.0,
            hostility REAL NOT NULL DEFAULT 0.0,
            gratitude REAL NOT NULL DEFAULT 0.0,
            anxiety REAL NOT NULL DEFAULT 0.0,
            excitement REAL NOT NULL DEFAULT 0.0,
            trust_signal REAL NOT NULL DEFAULT 0.0,
            valence REAL NOT NULL DEFAULT 0.0,
            arousal REAL NOT NULL DEFAULT 0.0,
            dominant_emotion TEXT NOT NULL DEFAULT 'neutral',
            is_sarcastic INTEGER NOT NULL DEFAULT 0
        );

        PRAGMA user_version = 2;
        """
    )

    base_ts = 1_700_000_000.123
    episode_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(base_ts))
    later_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(base_ts + 3600))

    conn.execute(
        """INSERT INTO episodes
           (id, session_id, turn_index, timestamp, user_text, assistant_text, summary, salience, valence, arousal, topics, entities, is_consolidated)
           VALUES (1, 'legacy-session', 0, ?, 'legacy user', 'legacy assistant', 'legacy summary', 0.8, -1, 0.4, 'infra', 'Alice', 0)""",
        (episode_iso,),
    )
    conn.execute(
        """INSERT INTO facts
           (id, subject_type, subject_id, predicate, object_text, confidence, source_episode_id, first_seen, last_seen, status, superseded_by)
           VALUES (1, 'user', '', 'prefers', 'dark mode', 0.7, 1, ?, ?, 'active', NULL)""",
        (episode_iso, later_iso),
    )
    conn.execute(
        """INSERT INTO open_loops
           (id, kind, text, due_hint, status, source_episode_id, created_at, resolved_at)
           VALUES (1, 'question', 'legacy open question', '', 'open', 1, ?, NULL)""",
        (episode_iso,),
    )
    conn.execute(
        """INSERT INTO reflections
           (id, kind, text, confidence, source_episode_id, created_at, last_reinforced, reinforcement_count)
           VALUES (1, 'pattern', 'legacy reflection', 0.6, 1, ?, ?, 2)""",
        (episode_iso, later_iso),
    )
    conn.execute(
        """INSERT INTO relationship_state
           (id, user_id, warmth, trust, tension, familiarity, humor, formality, volatility, last_updated, total_turns)
           VALUES (1, 'default', 0.6, 0.7, 0.1, 0.2, 0.1, 0.5, 0.0, ?, 4)""",
        (later_iso,),
    )
    conn.execute(
        """INSERT INTO identity_deltas
           (id, text, confidence, source_episode_id, created_at, applied)
           VALUES (1, 'legacy identity delta', 0.55, 1, ?, 0)""",
        (later_iso,),
    )
    conn.execute(
        """INSERT INTO entity_links
           (id, src_type, src_id, dst_type, dst_id, relation, weight)
           VALUES (1, 'episode', 1, 'fact', 1, 'extracted_from', 1.0)"""
    )
    conn.execute(
        """INSERT INTO emotion_log
           (id, episode_id, session_id, timestamp, frustration, warmth, humor, hostility, gratitude, anxiety, excitement, trust_signal, valence, arousal, dominant_emotion, is_sarcastic)
           VALUES (1, 1, 'legacy-session', ?, 0.2, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2, -0.1, 0.3, 'frustrated', 0)""",
        (later_iso,),
    )

    conn.commit()
    conn.close()
    return str(db_path)
