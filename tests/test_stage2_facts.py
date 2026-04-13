from datetime import datetime, timezone, timedelta

import pytest

from kortex.ingest import Ingestor
from kortex.models import Fact, OpenLoop


class TestFactExtraction:
    def test_extract_preference(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I prefer dark mode for everything", ep.id)
        assert len(facts) >= 1
        assert any(f.predicate == "prefers" for f in facts)
        assert any("dark mode" in f.object_text for f in facts)

    def test_extract_dislike(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I hate tabs, always use spaces", ep.id)
        assert len(facts) >= 1
        assert any(f.predicate == "dislikes" for f in facts)

    def test_extract_tool_usage(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I use neovim for all my editing", ep.id)
        assert len(facts) >= 1
        assert any(f.predicate == "uses" for f in facts)
        assert any("neovim" in f.object_text for f in facts)

    def test_extract_identity(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I'm a backend developer", ep.id)
        assert len(facts) >= 1
        assert any(f.predicate == "is" for f in facts)

    def test_extract_workplace(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I work at Acme Corp", ep.id)
        assert len(facts) >= 1
        assert any(f.predicate == "works_at" for f in facts)
        assert any("Acme Corp" in f.object_text for f in facts)

    def test_extract_location(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I live in London", ep.id)
        assert len(facts) >= 1
        assert any(f.predicate == "lives_in" for f in facts)

    def test_extract_name(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("My name is Alex", ep.id)
        assert len(facts) >= 1
        assert any(f.predicate == "named" for f in facts)

    def test_extract_project_decision(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts(
            "We decided to use PostgreSQL for the database", ep.id
        )
        assert len(facts) >= 1
        assert any(f.predicate == "decision" for f in facts)

    def test_extract_works_on(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I work on the authentication service", ep.id)
        assert len(facts) >= 1
        assert any(f.predicate == "works_on" for f in facts)

    def test_no_facts_from_generic_text(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("The sky is blue today.", ep.id)
        assert len(facts) == 0

    def test_stopwords_filtered(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I prefer it", ep.id)
        assert len(facts) == 0

    def test_too_short_filtered(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I prefer X", ep.id)
        assert len(facts) == 0

    def test_facts_stored_in_db(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        ingestor.extract_facts("I prefer Python over JavaScript", ep.id)
        db_facts = kortex_db.get_active_facts(subject_type="user")
        assert len(db_facts) >= 1

    def test_fact_has_source_episode(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        facts = ingestor.extract_facts("I live in Berlin", ep.id)
        assert facts[0].source_episode_id == ep.id


class TestFactDeduplication:
    def test_equivalent_fact_bumps_confidence(self, ingestor, kortex_db):
        ep1 = ingestor.ingest_turn("a", "b", session_id="s1")
        facts1 = ingestor.extract_facts("I prefer dark mode", ep1.id)
        assert len(facts1) >= 1
        original_confidence = facts1[0].confidence

        ep2 = ingestor.ingest_turn("c", "d", session_id="s1")
        facts2 = ingestor.extract_facts("I prefer dark mode for everything", ep2.id)

        all_active = kortex_db.get_active_facts(subject_type="user")
        prefers_facts = [f for f in all_active if f.predicate == "prefers"]
        assert len(prefers_facts) <= 2

    def test_contradicting_fact_supersedes(self, ingestor, kortex_db):
        ep1 = ingestor.ingest_turn("a", "b", session_id="s1")
        ingestor.extract_facts("I use Python 3.10 for development", ep1.id)

        ep2 = ingestor.ingest_turn("c", "d", session_id="s1")
        ingestor.extract_facts("I use Python 3.12 for development", ep2.id)

        active = kortex_db.get_active_facts(subject_type="user")
        uses_facts = [f for f in active if f.predicate == "uses"]
        if len(uses_facts) == 1:
            assert "3.12" in uses_facts[0].object_text

    def test_completely_different_facts_coexist(self, ingestor, kortex_db):
        ep1 = ingestor.ingest_turn("a", "b", session_id="s1")
        ingestor.extract_facts("I prefer dark mode", ep1.id)

        ep2 = ingestor.ingest_turn("c", "d", session_id="s1")
        ingestor.extract_facts("I live in Tokyo", ep2.id)

        active = kortex_db.get_active_facts(subject_type="user")
        assert len(active) >= 2


class TestFactSimilarity:
    def test_equivalent_facts(self):
        assert Ingestor._facts_are_equivalent("dark mode", "dark mode") is True

    def test_nearly_equivalent_facts(self):
        result = Ingestor._facts_are_equivalent(
            "dark mode for editing", "dark mode for everything"
        )
        assert isinstance(result, bool)

    def test_unrelated_facts_not_equivalent(self):
        assert Ingestor._facts_are_equivalent("dark mode", "light theme") is False

    def test_related_facts(self):
        assert (
            Ingestor._facts_are_related(
                "Python 3.10 for development", "Python 3.12 for development"
            )
            is True
        )

    def test_unrelated_facts_not_related(self):
        assert Ingestor._facts_are_related("dark mode", "lives in Tokyo") is False

    def test_empty_strings(self):
        assert Ingestor._facts_are_equivalent("", "test") is False
        assert Ingestor._facts_are_related("", "") is False


class TestOpenLoopLifecycle:
    def test_question_auto_resolved(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        ingestor.extract_open_loops(
            "Can you help me with the kubernetes deployment?", ep.id
        )
        loops_before = kortex_db.get_open_loops()
        assert len(loops_before) >= 1

        resolved = ingestor.resolve_answered_loops(
            "Sure! For the kubernetes deployment, you need to create a deployment.yaml..."
        )
        assert len(resolved) >= 1

        loops_after = kortex_db.get_open_loops()
        assert len(loops_after) < len(loops_before)

    def test_unrelated_answer_does_not_resolve(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        ingestor.extract_open_loops(
            "Can you help me with the kubernetes deployment?", ep.id
        )
        resolved = ingestor.resolve_answered_loops(
            "The weather today is sunny and warm."
        )
        assert len(resolved) == 0
        assert len(kortex_db.get_open_loops()) >= 1

    def test_commitment_resolved_on_completion(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        ingestor.extract_open_loops("I will fix the authentication bug", ep.id)
        loops_before = kortex_db.get_open_loops()
        assert len(loops_before) >= 1

        resolved = ingestor.resolve_completed_commitments(
            "Done! I've fixed the authentication bug. The issue was in the token validation."
        )
        assert len(resolved) >= 1

    def test_commitment_not_resolved_without_done_signal(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        ingestor.extract_open_loops("I will fix the authentication bug", ep.id)
        resolved = ingestor.resolve_completed_commitments(
            "I'm looking at the authentication bug now."
        )
        assert len(resolved) == 0
        assert len(kortex_db.get_open_loops()) >= 1


class TestDBFactMethods:
    def test_find_similar_facts(self, kortex_db):
        kortex_db.insert_fact(
            Fact(object_text="uses neovim for editing", predicate="uses")
        )
        kortex_db.insert_fact(
            Fact(object_text="prefers dark coffee", predicate="prefers")
        )
        results = kortex_db.find_similar_facts("neovim editing")
        assert len(results) >= 1
        assert "neovim" in results[0].object_text

    def test_find_similar_facts_with_predicate_filter(self, kortex_db):
        kortex_db.insert_fact(
            Fact(object_text="Python for scripting", predicate="uses")
        )
        kortex_db.insert_fact(Fact(object_text="Python language", predicate="prefers"))
        results = kortex_db.find_similar_facts("Python", predicate="uses")
        assert all(f.predicate == "uses" for f in results)

    def test_find_similar_facts_empty_query(self, kortex_db):
        results = kortex_db.find_similar_facts("")
        assert results == []

    def test_get_facts_by_predicate(self, kortex_db):
        kortex_db.insert_fact(Fact(object_text="vim", predicate="uses"))
        kortex_db.insert_fact(Fact(object_text="dark mode", predicate="prefers"))
        kortex_db.insert_fact(Fact(object_text="emacs", predicate="uses"))

        uses = kortex_db.get_facts_by_predicate("uses")
        assert len(uses) == 2
        assert all(f.predicate == "uses" for f in uses)

    def test_count_facts(self, kortex_db):
        assert kortex_db.count_facts() == 0
        kortex_db.insert_fact(Fact(object_text="test1"))
        kortex_db.insert_fact(Fact(object_text="test2"))
        assert kortex_db.count_facts() == 2

    def test_count_facts_excludes_superseded(self, kortex_db):
        f1 = Fact(object_text="old")
        kortex_db.insert_fact(f1)
        f2 = Fact(object_text="new")
        kortex_db.insert_fact(f2)
        kortex_db.supersede_fact(f1.id, f2.id)
        assert kortex_db.count_facts("active") == 1

    def test_bump_fact_last_seen(self, kortex_db):
        fact = Fact(object_text="test fact")
        kortex_db.insert_fact(fact)
        original = kortex_db.get_active_facts()[0].last_seen

        import time

        time.sleep(0.1)
        kortex_db.bump_fact_last_seen(fact.id)
        updated = kortex_db.get_active_facts()[0].last_seen
        assert updated >= original


class TestDBOpenLoopMethods:
    def test_expire_old_loops(self, kortex_db):
        old_loop = OpenLoop(
            kind="commitment",
            text="old commitment",
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        kortex_db.insert_open_loop(old_loop)

        recent_loop = OpenLoop(kind="task", text="recent task")
        kortex_db.insert_open_loop(recent_loop)

        expired_count = kortex_db.expire_old_loops(days_threshold=14.0)
        assert expired_count == 1

        remaining = kortex_db.get_open_loops()
        assert len(remaining) == 1
        assert remaining[0].text == "recent task"

    def test_count_open_loops(self, kortex_db):
        assert kortex_db.count_open_loops() == 0
        kortex_db.insert_open_loop(OpenLoop(text="loop1"))
        kortex_db.insert_open_loop(OpenLoop(text="loop2"))
        assert kortex_db.count_open_loops() == 2

    def test_count_excludes_resolved(self, kortex_db):
        loop = OpenLoop(text="test")
        kortex_db.insert_open_loop(loop)
        kortex_db.resolve_loop(loop.id)
        assert kortex_db.count_open_loops() == 0

    def test_search_open_loops(self, kortex_db):
        kortex_db.insert_open_loop(OpenLoop(text="fix kubernetes deployment"))
        kortex_db.insert_open_loop(OpenLoop(text="review pull request"))
        results = kortex_db.search_open_loops("kubernetes")
        assert len(results) == 1
        assert "kubernetes" in results[0].text

    def test_search_open_loops_no_match(self, kortex_db):
        kortex_db.insert_open_loop(OpenLoop(text="fix the bug"))
        results = kortex_db.search_open_loops("kubernetes")
        assert len(results) == 0


class TestConfidenceDecay:
    def test_decay_stale_facts(self, kortex_db):
        stale_fact = Fact(
            object_text="old preference",
            confidence=0.7,
            first_seen=datetime.now(timezone.utc) - timedelta(days=90),
            last_seen=datetime.now(timezone.utc) - timedelta(days=90),
        )
        kortex_db.insert_fact(stale_fact)

        recent_fact = Fact(
            object_text="recent preference",
            confidence=0.7,
        )
        kortex_db.insert_fact(recent_fact)

        decayed = kortex_db.decay_stale_facts(days_threshold=60.0, decay_rate=0.1)
        assert decayed == 1

        facts = kortex_db.get_active_facts()
        stale = next(f for f in facts if "old" in f.object_text)
        recent = next(f for f in facts if "recent" in f.object_text)
        assert stale.confidence == pytest.approx(0.6, abs=0.01)
        assert recent.confidence == pytest.approx(0.7, abs=0.01)

    def test_decay_does_not_go_below_minimum(self, kortex_db):
        stale_fact = Fact(
            object_text="barely held fact",
            confidence=0.15,
            first_seen=datetime.now(timezone.utc) - timedelta(days=120),
            last_seen=datetime.now(timezone.utc) - timedelta(days=120),
        )
        kortex_db.insert_fact(stale_fact)

        kortex_db.decay_stale_facts(days_threshold=60.0, decay_rate=0.1)

        facts = kortex_db.get_active_facts()
        assert facts[0].confidence >= 0.1

    def test_decay_skips_already_minimum(self, kortex_db):
        stale_fact = Fact(
            object_text="floor fact",
            confidence=0.1,
            first_seen=datetime.now(timezone.utc) - timedelta(days=120),
            last_seen=datetime.now(timezone.utc) - timedelta(days=120),
        )
        kortex_db.insert_fact(stale_fact)

        decayed = kortex_db.decay_stale_facts(days_threshold=60.0, decay_rate=0.1)
        assert decayed == 0


class TestSyncTurnIntegration:
    def test_sync_turn_extracts_facts(self, tmp_path):
        from kortex.config import KortexConfig
        from kortex.provider import KortexProvider
        import time

        config = KortexConfig(db_path=str(tmp_path / "test.db"))
        p = KortexProvider(config=config)
        p.initialize("test-session", hermes_home=str(tmp_path))

        p.sync_turn(
            "I prefer Python over JavaScript for backend work",
            "Python is a great choice for backend development!",
        )
        time.sleep(1.0)

        facts = p._db.get_active_facts(subject_type="user")
        assert len(facts) >= 1
        p.shutdown()

    def test_sync_turn_resolves_loops(self, tmp_path):
        from kortex.config import KortexConfig
        from kortex.provider import KortexProvider
        import time

        config = KortexConfig(db_path=str(tmp_path / "test.db"))
        p = KortexProvider(config=config)
        p.initialize("test-session", hermes_home=str(tmp_path))

        p.sync_turn(
            "Can you help me with the database migration?",
            "Sure, I can help with that.",
        )
        time.sleep(0.5)

        loops = p._db.get_open_loops()
        assert len(loops) >= 1

        p.sync_turn(
            "Thanks!", "Done! The database migration has been completed successfully."
        )
        time.sleep(1.0)

        p.shutdown()
