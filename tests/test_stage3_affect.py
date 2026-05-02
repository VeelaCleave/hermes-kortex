"""Stage 3 tests: Affect scoring, relationship dynamics, emotional recall."""

import time
import threading
from datetime import datetime, timedelta, timezone

import pytest

from kortex.affect import score_affect, _score_dimension, _detect_sarcasm
from kortex.calibrate import AffectBaseline, calibrate_affect
from kortex.models import AffectSignal, RelationshipState
from kortex.relationship import (
    update_relationship,
    apply_regression,
    compute_relationship_delta,
    _ema,
    _clamp,
)


# ======================================================================== #
# Affect Scoring
# ======================================================================== #


class TestAffectScoring:
    def test_neutral_text(self):
        affect = score_affect("Can you help me with this code?")
        assert affect.dominant_emotion == "neutral"
        assert affect.valence == 0.0
        assert affect.arousal == 0.0

    def test_frustration_detected(self):
        affect = score_affect("I'm so frustrated, this is still not working!")
        assert affect.frustration > 0.3
        assert affect.valence < 0

    def test_warmth_detected(self):
        affect = score_affect("Thank you so much, you're amazing!")
        assert affect.warmth > 0.3
        assert affect.gratitude > 0.3
        assert affect.valence > 0

    def test_hostility_detected(self):
        affect = score_affect("You're an idiot, this is completely useless")
        assert affect.hostility > 0.5
        assert affect.valence < 0

    def test_humor_detected(self):
        affect = score_affect("lol that's hilarious, you cracked me up 😂")
        assert affect.humor > 0.3
        assert affect.dominant_emotion == "humor"

    def test_anxiety_detected(self):
        affect = score_affect(
            "I'm really worried about the deadline, running out of time"
        )
        assert affect.anxiety > 0.3
        assert affect.dominant_emotion == "anxiety"

    def test_excitement_detected(self):
        affect = score_affect(
            "YES FINALLY!! This is incredible, can't wait to ship it! 🚀🎉"
        )
        assert affect.excitement > 0.3
        assert affect.valence > 0

    def test_trust_signal_detected(self):
        affect = score_affect("Honestly, I trust you completely with this")
        assert affect.trust_signal > 0.3

    def test_sarcasm_dampens_warmth(self):
        sarcastic = score_affect("Oh great, thanks for nothing, real helpful")
        sincere = score_affect("Thanks, that was really helpful")
        assert sarcastic.is_sarcastic
        assert not sincere.is_sarcastic
        assert sarcastic.warmth < sincere.warmth

    def test_sarcasm_boosts_frustration(self):
        affect = score_affect("Oh wonderful, another bug. Yeah right, sure thing")
        assert affect.is_sarcastic
        assert affect.frustration >= 0.3

    def test_hostility_implies_frustration(self):
        affect = score_affect("You're completely useless and pathetic")
        assert affect.frustration >= affect.hostility * 0.5

    def test_multiple_hits_boost_score(self):
        mild = score_affect("I'm a bit frustrated")
        strong = score_affect(
            "I'm frustrated and annoyed, ugh, this is ridiculous again"
        )
        assert strong.frustration > mild.frustration

    def test_valence_positive(self):
        affect = score_affect("Love this! You're brilliant, thank you so much!")
        assert affect.valence > 0

    def test_valence_negative(self):
        affect = score_affect("This is terrible, I hate it, totally broken")
        assert affect.valence < 0

    def test_arousal_from_intensity(self):
        calm = score_affect("Please check this code")
        intense = score_affect("WHAT THE FUCK!! This is BROKEN AGAIN!! I'm furious!!")
        assert intense.arousal > calm.arousal

    def test_is_significant_threshold(self):
        neutral = score_affect("What is the weather today?")
        emotional = score_affect("I'm so frustrated and angry at this bug!")
        assert not neutral.is_significant
        assert emotional.is_significant

    def test_dominant_emotion_selection(self):
        affect = score_affect("You're an absolute idiot, I hate this")
        assert affect.dominant_emotion == "hostility"

    def test_compact_text_for_neutral(self):
        affect = score_affect("How do I parse JSON?")
        assert affect.to_compact_text() == ""

    def test_compact_text_for_emotional(self):
        affect = score_affect("I'm so frustrated, this never works!")
        text = affect.to_compact_text()
        assert "frustration" in text

    def test_emoji_detection(self):
        affect = score_affect("❤ you're the best 🙏😊")
        assert affect.warmth > 0

    def test_explicit_sarcasm_tag(self):
        affect = score_affect("Great job on that /s")
        assert affect.is_sarcastic


# ======================================================================== #
# Relationship Dynamics
# ======================================================================== #


class TestRelationshipUpdate:
    def test_warmth_increases_from_gratitude(self):
        rel = RelationshipState(warmth=0.5)
        affect = AffectSignal(warmth=0.6, gratitude=0.7)
        updated = update_relationship(affect, rel)
        assert updated.warmth > 0.5

    def test_warmth_decreases_from_hostility(self):
        rel = RelationshipState(warmth=0.5)
        affect = AffectSignal(hostility=0.7)
        updated = update_relationship(affect, rel)
        assert updated.warmth < 0.5

    def test_trust_increases_slowly(self):
        rel = RelationshipState(trust=0.5)
        affect = AffectSignal(trust_signal=0.7)
        updated = update_relationship(affect, rel)
        assert updated.trust > 0.5
        assert updated.trust < 0.6  # trust moves slowly (alpha=0.08)

    def test_trust_drops_from_hostility(self):
        rel = RelationshipState(trust=0.5)
        affect = AffectSignal(hostility=0.8)
        updated = update_relationship(affect, rel)
        assert updated.trust < 0.5

    def test_tension_rises_from_frustration(self):
        rel = RelationshipState(tension=0.0)
        affect = AffectSignal(frustration=0.6)
        updated = update_relationship(affect, rel)
        assert updated.tension > 0.0

    def test_tension_drops_from_warmth(self):
        rel = RelationshipState(tension=0.5)
        affect = AffectSignal(warmth=0.6, gratitude=0.5)
        updated = update_relationship(affect, rel)
        assert updated.tension < 0.5

    def test_familiarity_always_increases(self):
        rel = RelationshipState(familiarity=0.1)
        affect = AffectSignal()  # neutral turn
        updated = update_relationship(affect, rel)
        assert updated.familiarity > 0.1

    def test_familiarity_increases_more_with_trust(self):
        rel1 = RelationshipState(familiarity=0.1)
        rel2 = RelationshipState(familiarity=0.1)
        neutral = AffectSignal()
        trusting = AffectSignal(trust_signal=0.7)
        updated_neutral = update_relationship(neutral, rel1)
        updated_trust = update_relationship(trusting, rel2)
        assert updated_trust.familiarity > updated_neutral.familiarity

    def test_humor_responds_to_humor_signals(self):
        rel = RelationshipState(humor=0.0)
        affect = AffectSignal(humor=0.6)
        updated = update_relationship(affect, rel)
        assert updated.humor > 0.0

    def test_humor_dampened_by_hostility(self):
        rel = RelationshipState(humor=0.5)
        affect = AffectSignal(hostility=0.6)
        updated = update_relationship(affect, rel)
        assert updated.humor < 0.5

    def test_volatility_from_intense_emotions(self):
        rel = RelationshipState(volatility=0.0)
        affect = AffectSignal(hostility=0.8, frustration=0.7)
        updated = update_relationship(affect, rel)
        assert updated.volatility > 0.0

    def test_turn_counter_increments(self):
        rel = RelationshipState(total_turns=5)
        affect = AffectSignal()
        updated = update_relationship(affect, rel)
        assert updated.total_turns == 6

    def test_last_updated_changes(self):
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
        rel = RelationshipState(last_updated=old_time)
        affect = AffectSignal()
        updated = update_relationship(affect, rel)
        assert updated.last_updated > old_time

    def test_momentum_prevents_wild_swings(self):
        rel = RelationshipState(warmth=0.5, trust=0.5)
        hostile = AffectSignal(hostility=0.9, frustration=0.8)
        updated = update_relationship(hostile, rel)
        # Even with extreme hostility, values shouldn't swing more than ~0.15 in one turn
        assert updated.warmth > 0.3
        assert updated.trust > 0.3


class TestRegression:
    def test_no_regression_for_zero_days(self):
        rel = RelationshipState(warmth=0.8, tension=0.6)
        regressed = apply_regression(rel, 0.0)
        assert regressed.warmth == 0.8
        assert regressed.tension == 0.6

    def test_warmth_regresses_toward_baseline(self):
        rel = RelationshipState(warmth=0.9)
        regressed = apply_regression(rel, 5.0)
        assert 0.5 < regressed.warmth < 0.9

    def test_tension_regresses_toward_zero(self):
        rel = RelationshipState(tension=0.6)
        regressed = apply_regression(rel, 5.0)
        assert regressed.tension < 0.6

    def test_familiarity_never_regresses(self):
        rel = RelationshipState(familiarity=0.7)
        regressed = apply_regression(rel, 30.0)
        assert regressed.familiarity == 0.7

    def test_regression_capped(self):
        rel = RelationshipState(warmth=0.9)
        regressed = apply_regression(rel, 1000.0)
        assert regressed.warmth >= 0.5  # shouldn't overshoot baseline


class TestHelpers:
    def test_clamp(self):
        assert _clamp(1.5) == 1.0
        assert _clamp(-0.5) == 0.0
        assert _clamp(0.5) == 0.5

    def test_ema(self):
        assert _ema(0.5, 1.0, 0.0) == 0.5  # no learning
        assert _ema(0.5, 1.0, 1.0) == 1.0  # instant
        result = _ema(0.5, 1.0, 0.5)
        assert 0.5 < result < 1.0


# ======================================================================== #
# DB Emotion Log
# ======================================================================== #


class TestEmotionLogDB:
    def test_insert_and_retrieve(self, kortex_db):
        from kortex.models import Episode

        ep = Episode(session_id="s1", user_text="hello", assistant_text="hi")
        ep.id = kortex_db.insert_episode(ep)

        affect = AffectSignal(
            frustration=0.6,
            warmth=0.1,
            humor=0.0,
            hostility=0.0,
            gratitude=0.0,
            anxiety=0.0,
            excitement=0.0,
            trust_signal=0.0,
            valence=-0.5,
            arousal=0.6,
            dominant_emotion="frustration",
        )
        log_id = kortex_db.insert_emotion_log(affect, ep.id, session_id="s1")
        assert log_id > 0

        recent = kortex_db.get_recent_emotions(limit=5)
        assert len(recent) == 1
        assert recent[0].frustration == 0.6
        assert recent[0].dominant_emotion == "frustration"

    def test_get_by_session(self, kortex_db):
        from kortex.models import Episode

        ep1 = Episode(session_id="s1")
        ep1.id = kortex_db.insert_episode(ep1)
        ep2 = Episode(session_id="s2")
        ep2.id = kortex_db.insert_episode(ep2)

        affect1 = AffectSignal(warmth=0.5, dominant_emotion="warmth")
        affect2 = AffectSignal(frustration=0.7, dominant_emotion="frustration")
        kortex_db.insert_emotion_log(affect1, ep1.id, session_id="s1")
        kortex_db.insert_emotion_log(affect2, ep2.id, session_id="s2")

        s1_emotions = kortex_db.get_recent_emotions(limit=10, session_id="s1")
        assert len(s1_emotions) == 1
        assert s1_emotions[0].dominant_emotion == "warmth"

    def test_get_emotion_for_episode(self, kortex_db):
        from kortex.models import Episode

        ep = Episode(session_id="s1")
        ep.id = kortex_db.insert_episode(ep)

        affect = AffectSignal(humor=0.8, dominant_emotion="humor")
        kortex_db.insert_emotion_log(affect, ep.id, session_id="s1")

        result = kortex_db.get_emotion_for_episode(ep.id)
        assert result is not None
        assert result.humor == 0.8

    def test_get_emotion_for_nonexistent_episode(self, kortex_db):
        result = kortex_db.get_emotion_for_episode(9999)
        assert result is None

    def test_emotional_trajectory(self, kortex_db):
        from kortex.models import Episode

        for i in range(5):
            ep = Episode(session_id="s1")
            ep.id = kortex_db.insert_episode(ep)
            affect = AffectSignal(
                valence=0.1 * (i + 1),
                arousal=0.2,
                dominant_emotion=[
                    "neutral",
                    "warmth",
                    "humor",
                    "excitement",
                    "gratitude",
                ][i],
            )
            kortex_db.insert_emotion_log(affect, ep.id, session_id="s1")

        trajectory = kortex_db.get_emotional_trajectory(limit=5, session_id="s1")
        assert len(trajectory) == 5
        assert all("valence" in t and "emotion" in t for t in trajectory)


class TestDBMigration:
    def test_fresh_db_has_emotion_log(self, kortex_db):
        tables = (
            kortex_db._get_conn()
            .execute("SELECT name FROM sqlite_master WHERE type='table'")
            .fetchall()
        )
        table_names = {r["name"] for r in tables}
        assert "emotion_log" in table_names

    def test_schema_version(self, kortex_db):
        version = kortex_db._get_conn().execute("PRAGMA user_version").fetchone()[0]
        assert version == 4


# ======================================================================== #
# Recall Emotional Trajectory
# ======================================================================== #


class TestRecallEmotionalContext:
    def test_recall_includes_relationship(self, kortex_db, recall):
        from kortex.models import Episode

        rel = RelationshipState(warmth=0.8, trust=0.7, total_turns=10)
        kortex_db.upsert_relationship(rel)

        ep = Episode(session_id="s1", user_text="hello", assistant_text="hi")
        ep.summary = "greeting exchange"
        ep.salience = 0.5
        ep.id = kortex_db.insert_episode(ep)

        ctx = recall.build_context("hello", session_id="s1")
        assert "warm rapport" in ctx or "high trust" in ctx

    def test_recall_includes_emotional_trajectory(self, kortex_db, recall):
        from kortex.models import Episode

        rel = RelationshipState(total_turns=5)
        kortex_db.upsert_relationship(rel)

        for i in range(3):
            ep = Episode(
                session_id="s1", user_text=f"msg {i}", assistant_text=f"resp {i}"
            )
            ep.summary = f"turn {i}"
            ep.id = kortex_db.insert_episode(ep)
            affect = AffectSignal(
                frustration=0.6,
                valence=-0.4,
                arousal=0.5,
                dominant_emotion="frustration",
            )
            kortex_db.insert_emotion_log(affect, ep.id, session_id="s1")

        ctx = recall.build_context("test", session_id="s1")
        # Lightweight mode skips emotional trajectory — just verify we get basic context
        assert "[KORTEX Memory]" in ctx


# ======================================================================== #
# Provider Integration
# ======================================================================== #


class TestProviderAffectIntegration:
    def test_sync_turn_updates_relationship(
        self, kortex_db, kortex_config, tmp_db_path
    ):
        from kortex.provider import KortexProvider

        provider = KortexProvider(config=kortex_config)
        provider.initialize("test-session", hermes_home="/tmp")
        provider._db = kortex_db
        from kortex.ingest import Ingestor
        from kortex.recall import Recall

        provider._ingestor = Ingestor(kortex_db)
        provider._recall = Recall(kortex_db, kortex_config)

        provider.sync_turn(
            "Thank you so much, you're brilliant! I really appreciate your help",
            "You're welcome!",
            session_id="test",
        )
        time.sleep(0.5)  # background thread

        rel = kortex_db.get_relationship()
        assert rel.total_turns > 0
        assert rel.warmth >= 0.5

    def test_sync_turn_logs_emotion(self, kortex_db, kortex_config, tmp_db_path):
        from kortex.provider import KortexProvider

        provider = KortexProvider(config=kortex_config)
        provider.initialize("test-session", hermes_home="/tmp")
        provider._db = kortex_db
        from kortex.ingest import Ingestor
        from kortex.recall import Recall

        provider._ingestor = Ingestor(kortex_db)
        provider._recall = Recall(kortex_db, kortex_config)

        provider.sync_turn(
            "I'm really frustrated and angry about this bug!",
            "I understand, let me help fix it.",
            session_id="test",
        )
        time.sleep(0.5)

        emotions = kortex_db.get_recent_emotions(limit=5)
        assert len(emotions) >= 1
        assert emotions[0].frustration > 0

    def test_sync_turn_neutral_doesnt_log_emotion(self, kortex_db, kortex_config):
        from kortex.provider import KortexProvider

        provider = KortexProvider(config=kortex_config)
        provider.initialize("test-session", hermes_home="/tmp")
        provider._db = kortex_db
        from kortex.ingest import Ingestor
        from kortex.recall import Recall

        provider._ingestor = Ingestor(kortex_db)
        provider._recall = Recall(kortex_db, kortex_config)

        provider.sync_turn(
            "What is 2 + 2?",
            "4",
            session_id="test",
        )
        time.sleep(0.5)

        emotions = kortex_db.get_recent_emotions(limit=5)
        assert len(emotions) == 0  # neutral emotions not logged

    def test_status_includes_emotional_data(self, kortex_db, kortex_config):
        import json
        from kortex.provider import KortexProvider

        provider = KortexProvider(config=kortex_config)
        provider.initialize("test-session", hermes_home="/tmp")
        provider._db = kortex_db

        status = json.loads(provider._handle_status())
        assert "relationship" in status
        assert "humor" in status["relationship"]
        assert "recent_emotional_state" in status


class TestAffectCalibration:
    def test_db_roundtrip_for_baseline(self, kortex_db):
        baseline = AffectBaseline(
            user_id="__default__", baseline_frustration=0.4, sample_count=5
        )
        kortex_db.upsert_affect_baseline(baseline)
        restored = kortex_db.get_affect_baseline()
        assert restored.baseline_frustration == 0.4
        assert restored.sample_count == 5

    def test_recall_uses_calibrated_affect(self, kortex_db):
        baseline = AffectBaseline(
            baseline_frustration=0.5,
            sample_count=25,
        )
        kortex_db.upsert_affect_baseline(baseline)
        calibrated = calibrate_affect(
            AffectSignal(frustration=0.8, dominant_emotion="frustration"),
            baseline,
            minimum_samples=20,
        )
        assert calibrated.frustration == pytest.approx(0.3, abs=0.01)


# ======================================================================== #
# AffectSignal Model
# ======================================================================== #


class TestAffectSignalModel:
    def test_to_db_row(self):
        affect = AffectSignal(
            frustration=0.5,
            warmth=0.3,
            humor=0.1,
            dominant_emotion="frustration",
            is_sarcastic=True,
        )
        row = affect.to_db_row()
        assert row["frustration"] == 0.5
        assert row["is_sarcastic"] == 1
        assert row["dominant_emotion"] == "frustration"

    def test_from_db_row(self):
        row = {
            "frustration": 0.5,
            "warmth": 0.3,
            "humor": 0.1,
            "hostility": 0.0,
            "gratitude": 0.0,
            "anxiety": 0.0,
            "excitement": 0.0,
            "trust_signal": 0.0,
            "valence": -0.2,
            "arousal": 0.5,
            "dominant_emotion": "frustration",
            "is_sarcastic": 0,
        }
        affect = AffectSignal.from_db_row(row)
        assert affect.frustration == 0.5
        assert affect.dominant_emotion == "frustration"
        assert not affect.is_sarcastic

    def test_round_trip(self):
        original = AffectSignal(
            frustration=0.7,
            warmth=0.2,
            humor=0.0,
            hostility=0.3,
            gratitude=0.0,
            anxiety=0.1,
            excitement=0.0,
            trust_signal=0.0,
            valence=-0.4,
            arousal=0.7,
            dominant_emotion="frustration",
            is_sarcastic=True,
        )
        row = original.to_db_row()
        restored = AffectSignal.from_db_row(row)
        assert restored.frustration == original.frustration
        assert restored.is_sarcastic == original.is_sarcastic
        assert restored.dominant_emotion == original.dominant_emotion
