"""Stage 4 tests — Reflection extraction, storage, reinforcement, decay,
identity directives, provider wiring, and recall injection."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from kortex.affect import score_affect
from kortex.config import KortexConfig
from kortex.db import KortexDB
from kortex.models import AffectSignal, Episode, IdentityDelta, Reflection
from kortex.recall import Recall
from kortex.reflect import (
    _CORRECTION_PATTERNS,
    _IDENTITY_DIRECTIVE_PATTERNS,
    _PRAISE_PATTERNS,
    _REPETITION_PATTERNS,
    _STYLE_PREFERENCE_PATTERNS,
    _WRONG_OUTPUT_PATTERNS,
    _extract_context,
    _reflections_similar,
    extract_identity_directives,
    extract_mistakes,
    extract_style_preferences,
    extract_successes,
    process_reflections,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def affect_neutral():
    return AffectSignal()


@pytest.fixture
def affect_frustrated():
    return AffectSignal(frustration=0.6, valence=-0.5, arousal=0.4)


@pytest.fixture
def affect_warm():
    return AffectSignal(warmth=0.7, valence=0.6, arousal=0.3)


@pytest.fixture
def episode_in_db(kortex_db):
    ep = Episode(
        session_id="test-session",
        turn_index=1,
        user_text="some user text",
        assistant_text="some assistant text",
        summary="test episode",
        salience=0.5,
    )
    ep.id = kortex_db.insert_episode(ep)
    return ep


# ===========================================================================
# extract_mistakes
# ===========================================================================


class TestExtractMistakes:
    def test_correction_no_i_said(self, affect_neutral):
        result = extract_mistakes(
            "No, I said I wanted the blue version",
            "Here is the red version",
            affect_neutral,
        )
        assert len(result) >= 1
        assert any("Correction" in m for m in result)

    def test_correction_thats_wrong(self, affect_neutral):
        result = extract_mistakes(
            "That's not right, the function should return a list",
            "I made it return a dict",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_correction_got_it_wrong(self, affect_neutral):
        result = extract_mistakes(
            "You got it wrong, we need Python 3.10",
            "Using Python 3.9",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_correction_already_told(self, affect_neutral):
        result = extract_mistakes(
            "I already told you the port is 8080",
            "Using port 3000",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_correction_wrong_punctuated(self, affect_neutral):
        result = extract_mistakes(
            "Wrong! That's not the endpoint I specified.",
            "Calling /api/v1",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_correction_try_again(self, affect_neutral):
        result = extract_mistakes(
            "Try again, that didn't work at all.",
            "Failed attempt",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_correction_misunderstood(self, affect_neutral):
        result = extract_mistakes(
            "That's not how it works, you misunderstood",
            "Some wrong approach",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_correction_didnt_ask(self, affect_neutral):
        result = extract_mistakes(
            "I didn't ask for that, I wanted a summary",
            "Here's the full code",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_repetition_like_i_said(self, affect_neutral):
        result = extract_mistakes(
            "Like I said, use the staging environment",
            "Using production",
            affect_neutral,
        )
        assert len(result) >= 1
        assert any("repeat" in m.lower() or "Had to repeat" in m for m in result)

    def test_repetition_for_the_second_time(self, affect_neutral):
        result = extract_mistakes(
            "For the second time, the config is in /etc/app/",
            "Looking in /var/",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_repetition_pay_attention(self, affect_neutral):
        result = extract_mistakes(
            "Pay attention, I need the CSV output format",
            "Generating JSON",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_wrong_output_doesnt_work(self, affect_neutral):
        result = extract_mistakes(
            "That doesn't work, I'm getting a TypeError",
            "def foo(): ...",
            affect_neutral,
        )
        assert len(result) >= 1
        assert any("error" in m.lower() or "Produced error" in m for m in result)

    def test_wrong_output_still_broken(self, affect_neutral):
        result = extract_mistakes(
            "Still broken, the tests are failing again",
            "Fixed version",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_wrong_output_you_broke(self, affect_neutral):
        result = extract_mistakes(
            "You broke the login flow with that change",
            "Updated auth module",
            affect_neutral,
        )
        assert len(result) >= 1

    def test_no_false_positive_on_neutral(self, affect_neutral):
        result = extract_mistakes(
            "Can you help me with this function?",
            "Sure, here it is",
            affect_neutral,
        )
        assert len(result) == 0

    def test_no_false_positive_on_question(self, affect_neutral):
        result = extract_mistakes(
            "How does the caching layer work?",
            "It uses Redis with TTL",
            affect_neutral,
        )
        assert len(result) == 0

    def test_one_per_category(self, affect_neutral):
        """Each category only fires once (break after first match)."""
        result = extract_mistakes(
            "No, I said the blue one. That's not right at all. Wrong!",
            "Here's the red one",
            affect_neutral,
        )
        correction_count = sum(1 for m in result if "Correction" in m)
        assert correction_count == 1


# ===========================================================================
# extract_successes
# ===========================================================================


class TestExtractSuccesses:
    def test_perfect(self, affect_neutral):
        result = extract_successes(
            "Perfect, that's exactly what I needed", "", affect_neutral
        )
        assert len(result) >= 1
        assert any("Approach worked well" in s for s in result)

    def test_great_job(self, affect_neutral):
        result = extract_successes("Great job on the refactor!", "", affect_neutral)
        assert len(result) >= 1

    def test_exactly_what_i_wanted(self, affect_neutral):
        result = extract_successes(
            "This is exactly what I wanted for the dashboard", "", affect_neutral
        )
        assert len(result) >= 1

    def test_youre_the_best(self, affect_neutral):
        result = extract_successes("You're the best, thanks!", "", affect_neutral)
        assert len(result) >= 1

    def test_much_better(self, affect_neutral):
        result = extract_successes(
            "Much better than the first version", "", affect_neutral
        )
        assert len(result) >= 1

    def test_no_false_positive(self, affect_neutral):
        result = extract_successes("OK, next step please", "", affect_neutral)
        assert len(result) == 0


# ===========================================================================
# extract_style_preferences
# ===========================================================================


class TestExtractStylePreferences:
    def test_keep_it_short(self):
        result = extract_style_preferences(
            "Keep it short, I don't need the long version"
        )
        assert any("concise" in p.lower() for p in result)

    def test_more_detail(self):
        result = extract_style_preferences("Can you elaborate more on this?")
        assert any("detailed" in p.lower() for p in result)

    def test_bullet_points(self):
        result = extract_style_preferences("Use bullet points for the summary")
        assert any("structured" in p.lower() or "listed" in p.lower() for p in result)

    def test_just_the_code(self):
        result = extract_style_preferences("Just the code please, no explanation")
        assert any("code-only" in p.lower() or "code only" in p.lower() for p in result)

    def test_show_reasoning(self):
        result = extract_style_preferences("Show me your reasoning step by step")
        assert any("reasoning" in p.lower() or "thinking" in p.lower() for p in result)

    def test_be_casual(self):
        result = extract_style_preferences("Loosen up a bit, be more casual")
        assert any("casual" in p.lower() for p in result)

    def test_be_professional(self):
        result = extract_style_preferences("Be more professional in your responses")
        assert any("professional" in p.lower() for p in result)

    def test_dont_be_robotic(self):
        result = extract_style_preferences("Don't be so robotic with me")
        assert any("robotic" in p.lower() or "clinical" in p.lower() for p in result)

    def test_just_do_it(self):
        result = extract_style_preferences("Don't ask, just do it")
        assert any(
            "autonomous" in p.lower() or "confirmation" in p.lower() for p in result
        )

    def test_ask_first(self):
        result = extract_style_preferences("Ask me first before making changes")
        assert any("consulted" in p.lower() for p in result)

    def test_one_step(self):
        result = extract_style_preferences("One step at a time please")
        assert any("incremental" in p.lower() or "step" in p.lower() for p in result)

    def test_no_explanation(self):
        result = extract_style_preferences("I don't need the explanation, just output")
        assert any("direct" in p.lower() or "without" in p.lower() for p in result)

    def test_no_false_positive(self):
        result = extract_style_preferences("Help me debug this function")
        assert len(result) == 0

    def test_multiple_prefs(self):
        result = extract_style_preferences(
            "Keep it short and just the code, no explanation needed"
        )
        assert len(result) >= 2


# ===========================================================================
# extract_identity_directives
# ===========================================================================


class TestExtractIdentityDirectives:
    def test_from_now_on(self):
        result = extract_identity_directives(
            "From now on, always include type annotations in code"
        )
        assert len(result) >= 1
        kind, text = result[0]
        assert kind == "directive"
        assert "type annotations" in text.lower()

    def test_you_should_always(self):
        result = extract_identity_directives(
            "You should always explain your changes before making them"
        )
        assert len(result) >= 1

    def test_never_do(self):
        result = extract_identity_directives(
            "Never use console.log for debugging in production code"
        )
        assert len(result) >= 1

    def test_your_name_is(self):
        result = extract_identity_directives("Your name is Athena, not Hermes")
        assert len(result) >= 1
        kind, _ = result[0]
        assert kind == "identity"

    def test_act_like(self):
        result = extract_identity_directives(
            "Act like a senior engineer reviewing my code"
        )
        assert len(result) >= 1
        kind, _ = result[0]
        assert kind == "identity"

    def test_dont_ever(self):
        result = extract_identity_directives(
            "Don't ever apologize for being wrong, just fix it"
        )
        assert len(result) >= 1
        kind, _ = result[0]
        assert kind == "constraint"

    def test_stop_being(self):
        result = extract_identity_directives("Stop being so verbose with explanations")
        assert len(result) >= 1
        kind, _ = result[0]
        assert kind == "constraint"

    def test_you_should_be_more(self):
        result = extract_identity_directives("You should be more concise")
        assert len(result) >= 1
        kind, _ = result[0]
        assert kind == "identity"

    def test_short_text_filtered(self):
        result = extract_identity_directives("Your name is Bob")
        for _, text in result:
            assert len(text) >= 5

    def test_no_false_positive(self):
        result = extract_identity_directives("Can you check the database schema?")
        assert len(result) == 0


# ===========================================================================
# _reflections_similar
# ===========================================================================


class TestReflectionsSimilar:
    def test_identical(self):
        assert _reflections_similar(
            "User prefers concise responses", "User prefers concise responses"
        )

    def test_similar(self):
        assert _reflections_similar(
            "User prefers concise short responses",
            "User prefers brief concise responses",
        )

    def test_different(self):
        assert not _reflections_similar(
            "User prefers concise responses",
            "Agent broke the login flow with auth changes",
        )

    def test_empty(self):
        assert not _reflections_similar("", "something")
        assert not _reflections_similar("something", "")

    def test_only_stopwords(self):
        assert not _reflections_similar("the a an", "is was are")


# ===========================================================================
# _extract_context
# ===========================================================================


class TestExtractContext:
    def test_basic(self):
        import re

        pat = re.compile(r"\bwrong\b", re.I)
        result = _extract_context(
            "That's wrong, fix it please.", pat, prefix="Correction"
        )
        assert "Correction" in result
        assert "wrong" in result.lower()

    def test_no_match(self):
        import re

        pat = re.compile(r"\bxyzzy\b", re.I)
        result = _extract_context("Nothing matches here.", pat)
        assert result == ""

    def test_truncation(self):
        import re

        pat = re.compile(r"\bstart\b", re.I)
        long_text = "start " + "word " * 200
        result = _extract_context(long_text, pat, prefix="Test")
        assert len(result) <= 300


# ===========================================================================
# process_reflections (integration)
# ===========================================================================


class TestProcessReflections:
    def test_stores_mistake(self, kortex_db, episode_in_db, affect_frustrated):
        results = process_reflections(
            kortex_db,
            "No, I said use Python 3.10 not 3.9, that's wrong!",
            "Using Python 3.9",
            affect_frustrated,
            episode_in_db.id,
        )
        assert len(results) >= 1
        stored = kortex_db.get_reflections(kind="mistake")
        assert len(stored) >= 1

    def test_stores_success(self, kortex_db, episode_in_db, affect_warm):
        results = process_reflections(
            kortex_db,
            "Perfect, that's exactly what I needed!",
            "Here's the solution",
            affect_warm,
            episode_in_db.id,
        )
        assert len(results) >= 1
        stored = kortex_db.get_reflections(kind="pattern")
        assert len(stored) >= 1

    def test_stores_preference(self, kortex_db, episode_in_db, affect_neutral):
        results = process_reflections(
            kortex_db,
            "Keep it short and just give me the code",
            "Here's a brief explanation...",
            affect_neutral,
            episode_in_db.id,
        )
        prefs = kortex_db.get_reflections(kind="preference")
        assert len(prefs) >= 1

    def test_stores_identity_delta(self, kortex_db, episode_in_db, affect_neutral):
        process_reflections(
            kortex_db,
            "From now on, always include error handling in your code examples",
            "Sure thing",
            affect_neutral,
            episode_in_db.id,
        )
        deltas = kortex_db.get_identity_deltas()
        assert len(deltas) >= 1
        assert "error handling" in deltas[0].text.lower()

    def test_reinforces_existing(self, kortex_db, episode_in_db, affect_neutral):
        ref = Reflection(
            kind="preference",
            text="User prefers concise responses",
            confidence=0.5,
            source_episode_id=episode_in_db.id,
        )
        ref.id = kortex_db.insert_reflection(ref)

        process_reflections(
            kortex_db,
            "Keep it short, I don't need long explanations",
            "Noted",
            affect_neutral,
            episode_in_db.id,
        )

        updated = kortex_db.get_reflections(kind="preference")
        high_conf = [r for r in updated if r.confidence > 0.5]
        assert len(high_conf) >= 1

    def test_frustrated_correction_higher_confidence(
        self, kortex_db, episode_in_db, affect_frustrated
    ):
        results = process_reflections(
            kortex_db,
            "No, I said use port 8080! That's wrong.",
            "Using port 3000",
            affect_frustrated,
            episode_in_db.id,
        )
        mistakes = kortex_db.get_reflections(kind="mistake")
        assert len(mistakes) >= 1
        assert mistakes[0].confidence >= 0.4

    def test_no_reflections_on_neutral(self, kortex_db, episode_in_db, affect_neutral):
        results = process_reflections(
            kortex_db,
            "What time is it?",
            "It's 3pm UTC",
            affect_neutral,
            episode_in_db.id,
        )
        assert len(results) == 0


# ===========================================================================
# DB: get_high_confidence_reflections
# ===========================================================================


class TestDBHighConfidenceReflections:
    def test_filters_by_threshold(self, kortex_db, episode_in_db):
        low = Reflection(
            kind="mistake",
            text="Low confidence mistake note",
            confidence=0.2,
            source_episode_id=episode_in_db.id,
        )
        high = Reflection(
            kind="mistake",
            text="High confidence mistake note",
            confidence=0.8,
            source_episode_id=episode_in_db.id,
        )
        kortex_db.insert_reflection(low)
        kortex_db.insert_reflection(high)

        results = kortex_db.get_high_confidence_reflections(min_confidence=0.5)
        assert len(results) == 1
        assert results[0].text == "High confidence mistake note"

    def test_orders_by_confidence_desc(self, kortex_db, episode_in_db):
        for conf in [0.6, 0.9, 0.7]:
            r = Reflection(
                kind="pattern",
                text=f"Pattern at {conf}",
                confidence=conf,
                source_episode_id=episode_in_db.id,
            )
            kortex_db.insert_reflection(r)

        results = kortex_db.get_high_confidence_reflections(min_confidence=0.5)
        assert results[0].confidence == 0.9
        assert results[1].confidence == 0.7
        assert results[2].confidence == 0.6

    def test_respects_limit(self, kortex_db, episode_in_db):
        for i in range(10):
            r = Reflection(
                kind="pattern",
                text=f"Pattern {i} with high confidence",
                confidence=0.8,
                source_episode_id=episode_in_db.id,
            )
            kortex_db.insert_reflection(r)

        results = kortex_db.get_high_confidence_reflections(min_confidence=0.5, limit=3)
        assert len(results) == 3

    def test_empty_when_none_qualify(self, kortex_db, episode_in_db):
        r = Reflection(
            kind="mistake",
            text="Very low confidence reflection",
            confidence=0.1,
            source_episode_id=episode_in_db.id,
        )
        kortex_db.insert_reflection(r)

        results = kortex_db.get_high_confidence_reflections(min_confidence=0.5)
        assert len(results) == 0


# ===========================================================================
# DB: decay_stale_reflections
# ===========================================================================


class TestDBDecayReflections:
    def test_decays_old_reflections(self, kortex_db, episode_in_db):
        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).timestamp()
        r = Reflection(
            kind="pattern",
            text="Old pattern that should decay over time",
            confidence=0.7,
            source_episode_id=episode_in_db.id,
        )
        r.id = kortex_db.insert_reflection(r)
        kortex_db._get_conn().execute(
            "UPDATE reflections SET last_reinforced=? WHERE id=?",
            (old_time, r.id),
        )
        kortex_db._get_conn().commit()

        count = kortex_db.decay_stale_reflections(days_threshold=30.0, decay_rate=0.1)
        assert count == 1

        updated = kortex_db.get_reflections(kind="pattern")
        assert updated[0].confidence == pytest.approx(0.6, abs=0.01)

    def test_does_not_decay_recent(self, kortex_db, episode_in_db):
        r = Reflection(
            kind="pattern",
            text="Recent pattern should not decay at all",
            confidence=0.7,
            source_episode_id=episode_in_db.id,
        )
        kortex_db.insert_reflection(r)

        count = kortex_db.decay_stale_reflections(days_threshold=30.0, decay_rate=0.1)
        assert count == 0

    def test_floor_at_005(self, kortex_db, episode_in_db):
        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).timestamp()
        r = Reflection(
            kind="mistake",
            text="Near-floor confidence reflection to test floor",
            confidence=0.06,
            source_episode_id=episode_in_db.id,
        )
        r.id = kortex_db.insert_reflection(r)
        kortex_db._get_conn().execute(
            "UPDATE reflections SET last_reinforced=? WHERE id=?",
            (old_time, r.id),
        )
        kortex_db._get_conn().commit()

        kortex_db.decay_stale_reflections(days_threshold=30.0, decay_rate=0.5)
        updated = kortex_db.get_reflections(kind="mistake")
        assert updated[0].confidence >= 0.05


# ===========================================================================
# DB: get_identity_deltas
# ===========================================================================


class TestDBIdentityDeltas:
    def test_insert_and_retrieve(self, kortex_db, episode_in_db):
        delta = IdentityDelta(
            text="[directive] Always include type annotations",
            confidence=0.5,
            source_episode_id=episode_in_db.id,
        )
        kortex_db.insert_identity_delta(delta)

        deltas = kortex_db.get_identity_deltas()
        assert len(deltas) == 1
        assert "type annotations" in deltas[0].text

    def test_filter_by_applied(self, kortex_db, episode_in_db):
        d1 = IdentityDelta(
            text="[directive] Be more concise in responses",
            confidence=0.5,
            source_episode_id=episode_in_db.id,
            applied=False,
        )
        d2 = IdentityDelta(
            text="[identity] Name is Athena now permanently",
            confidence=0.8,
            source_episode_id=episode_in_db.id,
            applied=True,
        )
        kortex_db.insert_identity_delta(d1)
        kortex_db.insert_identity_delta(d2)

        unapplied = kortex_db.get_identity_deltas(applied=False)
        assert len(unapplied) == 1
        assert "concise" in unapplied[0].text

        applied = kortex_db.get_identity_deltas(applied=True)
        assert len(applied) == 1
        assert "Athena" in applied[0].text

    def test_respects_limit(self, kortex_db, episode_in_db):
        for i in range(10):
            d = IdentityDelta(
                text=f"[directive] Directive number {i} for testing",
                confidence=0.3,
                source_episode_id=episode_in_db.id,
            )
            kortex_db.insert_identity_delta(d)

        result = kortex_db.get_identity_deltas(limit=3)
        assert len(result) == 3

    def test_ordered_by_created_at_desc(self, kortex_db, episode_in_db):
        d1 = IdentityDelta(
            text="[directive] First directive created earlier",
            confidence=0.3,
            source_episode_id=episode_in_db.id,
        )
        kortex_db.insert_identity_delta(d1)
        time.sleep(0.01)
        d2 = IdentityDelta(
            text="[directive] Second directive created later",
            confidence=0.3,
            source_episode_id=episode_in_db.id,
        )
        kortex_db.insert_identity_delta(d2)

        result = kortex_db.get_identity_deltas()
        assert "Second" in result[0].text
        assert "First" in result[1].text


# ===========================================================================
# Recall: _build_reflections_section
# ===========================================================================


class TestRecallReflections:
    def test_builds_section(self, kortex_db, kortex_config):
        ep = Episode(session_id="s1", summary="test", salience=0.3)
        ep.id = kortex_db.insert_episode(ep)

        for kind, text, conf in [
            ("mistake", "Agent used wrong port number causing connection failure", 0.7),
            ("pattern", "Step by step approach worked well for debugging", 0.6),
            ("preference", "User prefers concise responses without preamble", 0.8),
        ]:
            r = Reflection(
                kind=kind, text=text, confidence=conf, source_episode_id=ep.id
            )
            kortex_db.insert_reflection(r)

        recall = Recall(kortex_db, kortex_config)
        section = recall._build_reflections_section(budget=500)

        assert "Learned behaviors:" in section
        assert "Avoid" in section
        assert "Works well" in section
        assert "User prefers" in section

    def test_empty_when_no_reflections(self, kortex_db, kortex_config):
        recall = Recall(kortex_db, kortex_config)
        section = recall._build_reflections_section(budget=500)
        assert section == ""

    def test_empty_when_below_threshold(self, kortex_db, kortex_config):
        ep = Episode(session_id="s1", summary="test", salience=0.3)
        ep.id = kortex_db.insert_episode(ep)

        r = Reflection(
            kind="mistake",
            text="Very low confidence reflection text",
            confidence=0.1,
            source_episode_id=ep.id,
        )
        kortex_db.insert_reflection(r)

        recall = Recall(kortex_db, kortex_config)
        section = recall._build_reflections_section(budget=500)
        assert section == ""

    def test_reinforcement_count_shown(self, kortex_db, kortex_config):
        ep = Episode(session_id="s1", summary="test", salience=0.3)
        ep.id = kortex_db.insert_episode(ep)

        r = Reflection(
            kind="preference",
            text="User prefers code-only responses always",
            confidence=0.8,
            source_episode_id=ep.id,
            reinforcement_count=3,
        )
        kortex_db.insert_reflection(r)

        recall = Recall(kortex_db, kortex_config)
        section = recall._build_reflections_section(budget=500)
        assert "(x3)" in section

    def test_included_in_build_context(self, kortex_db, kortex_config):
        ep = Episode(session_id="s1", summary="debugging session", salience=0.5)
        ep.id = kortex_db.insert_episode(ep)

        r = Reflection(
            kind="pattern",
            text="Incremental debugging approach worked well for user",
            confidence=0.8,
            source_episode_id=ep.id,
        )
        kortex_db.insert_reflection(r)

        recall = Recall(kortex_db, kortex_config)
        context = recall.build_context("debugging", session_id="s1")

        assert "Learned behaviors:" in context
        assert "debugging approach" in context.lower() or "debugging" in context.lower()

    def test_budget_trimming(self, kortex_db, kortex_config):
        ep = Episode(session_id="s1", summary="test", salience=0.3)
        ep.id = kortex_db.insert_episode(ep)

        for i in range(20):
            r = Reflection(
                kind="pattern",
                text=f"Very long reflection text number {i} with lots of extra words to fill budget space",
                confidence=0.9,
                source_episode_id=ep.id,
            )
            kortex_db.insert_reflection(r)

        recall = Recall(kortex_db, kortex_config)
        section = recall._build_reflections_section(budget=50)
        assert len(section) <= 50 * 4 + 10


# ===========================================================================
# Config: new fields
# ===========================================================================


class TestConfigReflectionFields:
    def test_defaults(self):
        config = KortexConfig()
        assert config.max_reflections_per_recall == 3
        assert config.reflection_confidence_threshold == 0.4
        assert "reflections" in config.budget
        assert config.budget["reflections"] == 200

    def test_from_dict(self):
        config = KortexConfig.from_dict(
            {
                "max_reflections_per_recall": 5,
                "reflection_confidence_threshold": 0.6,
            }
        )
        assert config.max_reflections_per_recall == 5
        assert config.reflection_confidence_threshold == 0.6

    def test_budget_sum_under_total(self):
        config = KortexConfig()
        active_budget = sum(v for k, v in config.budget.items() if k != "reserve")
        assert active_budget + config.budget["reserve"] <= config.total_budget


# ===========================================================================
# Provider: sync_turn wiring
# ===========================================================================


class TestProviderReflectionWiring:
    def test_sync_turn_creates_reflections(self, tmp_db_path):
        config = KortexConfig(db_path=tmp_db_path)
        from kortex.provider import KortexProvider

        provider = KortexProvider(config)
        provider.initialize("test-session")

        provider.sync_turn(
            "No, I said Python 3.10 not 3.9! That's wrong, you got it wrong.",
            "Using Python 3.9",
            session_id="test-session",
        )

        import time

        time.sleep(0.5)

        reflections = provider._db.get_reflections(kind="mistake")
        assert len(reflections) >= 1

    def test_sync_turn_creates_identity_delta(self, tmp_db_path):
        config = KortexConfig(db_path=tmp_db_path)
        from kortex.provider import KortexProvider

        provider = KortexProvider(config)
        provider.initialize("test-session")

        provider.sync_turn(
            "From now on, always use type annotations in your code examples",
            "Understood, I'll include types",
            session_id="test-session",
        )

        import time

        time.sleep(0.5)

        deltas = provider._db.get_identity_deltas()
        assert len(deltas) >= 1

    def test_sync_turn_no_reflections_on_neutral(self, tmp_db_path):
        config = KortexConfig(db_path=tmp_db_path)
        from kortex.provider import KortexProvider

        provider = KortexProvider(config)
        provider.initialize("test-session")

        provider.sync_turn(
            "What is the weather today?",
            "I don't have weather data",
            session_id="test-session",
        )

        import time

        time.sleep(0.5)

        reflections = provider._db.get_reflections()
        assert len(reflections) == 0
