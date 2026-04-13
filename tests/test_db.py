import time

import pytest

from kortex.db import KortexDB
from kortex.models import Episode, Fact, OpenLoop, Reflection, RelationshipState


class TestEpisodeCRUD:
    def test_insert_and_retrieve(self, kortex_db):
        ep = Episode(
            session_id="test-session",
            turn_index=0,
            user_text="Hello there",
            assistant_text="Hi! How can I help?",
            summary="Greeting exchange",
            salience=0.1,
            valence=1,
        )
        ep_id = kortex_db.insert_episode(ep)
        assert ep_id > 0
        assert ep.id == ep_id

        retrieved = kortex_db.get_episode(ep_id)
        assert retrieved is not None
        assert retrieved.session_id == "test-session"
        assert retrieved.user_text == "Hello there"
        assert retrieved.valence == 1

    def test_get_recent_episodes(self, kortex_db):
        for i in range(5):
            kortex_db.insert_episode(
                Episode(
                    session_id="s1",
                    turn_index=i,
                    user_text=f"msg {i}",
                    summary=f"turn {i}",
                )
            )
        recent = kortex_db.get_recent_episodes(limit=3)
        assert len(recent) == 3
        assert recent[0].turn_index == 4

    def test_get_recent_by_session(self, kortex_db):
        kortex_db.insert_episode(Episode(session_id="s1", user_text="a"))
        kortex_db.insert_episode(Episode(session_id="s2", user_text="b"))
        result = kortex_db.get_recent_episodes(limit=10, session_id="s1")
        assert len(result) == 1
        assert result[0].session_id == "s1"

    def test_search_episodes_fts(self, kortex_db):
        kortex_db.insert_episode(
            Episode(
                session_id="s1",
                summary="discussed kubernetes deployment",
                user_text="how do I deploy to k8s",
            )
        )
        kortex_db.insert_episode(
            Episode(
                session_id="s1",
                summary="talked about python testing",
                user_text="write pytest tests",
            )
        )
        results = kortex_db.search_episodes("kubernetes")
        assert len(results) >= 1
        assert "kubernetes" in results[0].summary

    def test_get_salient_episodes(self, kortex_db):
        kortex_db.insert_episode(
            Episode(session_id="s1", salience=0.1, summary="boring")
        )
        kortex_db.insert_episode(
            Episode(session_id="s1", salience=0.8, summary="important")
        )
        kortex_db.insert_episode(
            Episode(session_id="s1", salience=0.9, summary="critical")
        )

        salient = kortex_db.get_salient_episodes(min_salience=0.5)
        assert len(salient) == 2
        assert salient[0].salience >= salient[1].salience

    def test_update_episode(self, kortex_db):
        ep = Episode(session_id="s1", summary="original")
        kortex_db.insert_episode(ep)
        ep.summary = "updated"
        ep.salience = 0.9
        kortex_db.update_episode(ep)
        retrieved = kortex_db.get_episode(ep.id)
        assert retrieved.summary == "updated"
        assert retrieved.salience == 0.9

    def test_count_episodes(self, kortex_db):
        assert kortex_db.count_episodes() == 0
        kortex_db.insert_episode(Episode(session_id="s1"))
        kortex_db.insert_episode(Episode(session_id="s1"))
        assert kortex_db.count_episodes() == 2

    def test_session_turn_count(self, kortex_db):
        kortex_db.insert_episode(Episode(session_id="s1"))
        kortex_db.insert_episode(Episode(session_id="s1"))
        kortex_db.insert_episode(Episode(session_id="s2"))
        assert kortex_db.get_session_turn_count("s1") == 2
        assert kortex_db.get_session_turn_count("s2") == 1

    def test_get_episodes_for_session(self, kortex_db):
        kortex_db.insert_episode(
            Episode(session_id="s1", turn_index=0, summary="first")
        )
        kortex_db.insert_episode(
            Episode(session_id="s1", turn_index=1, summary="second")
        )
        kortex_db.insert_episode(
            Episode(session_id="s2", turn_index=0, summary="other")
        )
        episodes = kortex_db.get_episodes_for_session("s1")
        assert [ep.summary for ep in episodes] == ["first", "second"]


class TestConversationSummaries:
    def test_insert_and_list_conversation_summaries(self, kortex_db):
        summary_id = kortex_db.insert_conversation_summary(
            {
                "session_id": "s1",
                "summary_text": "Conversation covered: architecture and deployment",
                "episode_count": 2,
                "key_entities": "Alice,PostgreSQL",
            }
        )
        assert summary_id > 0

        summaries = kortex_db.list_conversation_summaries(limit=5)
        assert len(summaries) == 1
        assert summaries[0]["session_id"] == "s1"
        assert "architecture" in summaries[0]["summary_text"]

    def test_search_conversation_summaries(self, kortex_db):
        kortex_db.insert_conversation_summary(
            {
                "session_id": "s1",
                "summary_text": "Conversation covered: kubernetes deployment and rollback",
                "episode_count": 3,
                "key_entities": "Kubernetes",
            }
        )
        results = kortex_db.search_conversation_summaries("rollback")
        assert len(results) == 1
        assert "kubernetes" in results[0]["summary_text"].lower()


class TestFactCRUD:
    def test_insert_and_retrieve(self, kortex_db):
        fact = Fact(
            subject_type="user",
            predicate="prefers",
            object_text="dark mode",
            confidence=0.8,
        )
        fact_id = kortex_db.insert_fact(fact)
        assert fact_id > 0

        facts = kortex_db.get_active_facts(subject_type="user")
        assert len(facts) == 1
        assert facts[0].object_text == "dark mode"

    def test_search_facts(self, kortex_db):
        kortex_db.insert_fact(
            Fact(object_text="uses vim for editing", predicate="uses")
        )
        kortex_db.insert_fact(
            Fact(object_text="prefers dark roast coffee", predicate="prefers")
        )

        results = kortex_db.search_facts("vim")
        assert len(results) >= 1
        assert "vim" in results[0].object_text

    def test_supersede_fact(self, kortex_db):
        old = Fact(object_text="uses Python 3.10", predicate="uses")
        kortex_db.insert_fact(old)
        new = Fact(object_text="uses Python 3.12", predicate="uses")
        kortex_db.insert_fact(new)

        kortex_db.supersede_fact(old.id, new.id)
        active = kortex_db.get_active_facts()
        assert len(active) == 1
        assert active[0].object_text == "uses Python 3.12"
        old_row = kortex_db.get_fact(old.id)
        assert old_row.valid_to is not None
        assert old_row.contradiction_status == "superseded"

    def test_update_confidence(self, kortex_db):
        fact = Fact(object_text="test", confidence=0.5)
        kortex_db.insert_fact(fact)
        kortex_db.update_fact_confidence(fact.id, 0.9)
        facts = kortex_db.get_active_facts()
        assert facts[0].confidence == 0.9


class TestOpenLoopCRUD:
    def test_insert_and_retrieve(self, kortex_db):
        loop = OpenLoop(kind="commitment", text="will fix the bug tomorrow")
        kortex_db.insert_open_loop(loop)
        loops = kortex_db.get_open_loops()
        assert len(loops) == 1
        assert loops[0].text == "will fix the bug tomorrow"

    def test_resolve_loop(self, kortex_db):
        loop = OpenLoop(kind="task", text="deploy to staging")
        kortex_db.insert_open_loop(loop)
        kortex_db.resolve_loop(loop.id)
        open_loops = kortex_db.get_open_loops()
        assert len(open_loops) == 0


class TestReflectionCRUD:
    def test_insert_and_retrieve(self, kortex_db):
        ref = Reflection(kind="mistake", text="gave wrong port number")
        kortex_db.insert_reflection(ref)
        refs = kortex_db.get_reflections(kind="mistake")
        assert len(refs) == 1

    def test_search_reflections(self, kortex_db):
        kortex_db.insert_reflection(Reflection(text="user dislikes hedged answers"))
        kortex_db.insert_reflection(
            Reflection(text="code examples work better than explanations")
        )
        results = kortex_db.search_reflections("hedged answers")
        assert len(results) >= 1

    def test_reinforce(self, kortex_db):
        ref = Reflection(text="test", confidence=0.3)
        kortex_db.insert_reflection(ref)
        kortex_db.reinforce_reflection(ref.id, confidence_boost=0.2)
        refs = kortex_db.get_reflections()
        assert refs[0].confidence == pytest.approx(0.5, abs=0.01)
        assert refs[0].reinforcement_count == 2


class TestRelationshipState:
    def test_default_state(self, kortex_db):
        rel = kortex_db.get_relationship()
        assert rel.user_id == "__default__"
        assert rel.warmth == 0.5
        assert rel.total_turns == 0

    def test_upsert(self, kortex_db):
        rel = RelationshipState(warmth=0.8, trust=0.7, total_turns=42)
        kortex_db.upsert_relationship(rel)
        retrieved = kortex_db.get_relationship()
        assert retrieved.warmth == 0.8
        assert retrieved.total_turns == 42

    def test_upsert_update(self, kortex_db):
        kortex_db.upsert_relationship(RelationshipState(warmth=0.3))
        kortex_db.upsert_relationship(RelationshipState(warmth=0.9, total_turns=10))
        rel = kortex_db.get_relationship()
        assert rel.warmth == 0.9
        assert rel.total_turns == 10

    def test_compact_text(self):
        rel = RelationshipState(
            warmth=0.9, trust=0.8, humor=0.7, familiarity=0.8, total_turns=100
        )
        text = rel.to_compact_text()
        assert "warm rapport" in text
        assert "high trust" in text
        assert "100 interactions" in text
