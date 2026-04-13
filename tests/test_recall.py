import time

from kortex.models import Episode
from kortex.recall import Recall


class TestBuildContext:
    def test_empty_db_returns_empty(self, recall):
        ctx = recall.build_context("hello")
        assert ctx == ""

    def test_returns_context_with_episodes(self, kortex_db, recall):
        kortex_db.insert_episode(
            Episode(
                session_id="s1",
                summary="discussed project architecture",
                user_text="let's talk about the architecture",
                salience=0.5,
                valence=1,
            )
        )
        ctx = recall.build_context("architecture")
        assert "KORTEX Memory" in ctx
        assert "architecture" in ctx.lower()

    def test_context_includes_facts(self, kortex_db, recall):
        from kortex.models import Fact

        kortex_db.insert_fact(
            Fact(
                subject_type="user",
                predicate="prefers",
                object_text="dark mode in all editors",
                confidence=0.8,
            )
        )
        ctx = recall.build_context("editor preferences")
        assert "dark mode" in ctx

    def test_context_includes_open_loops(self, kortex_db, recall):
        from kortex.models import OpenLoop

        kortex_db.insert_open_loop(
            OpenLoop(
                kind="commitment",
                text="fix auth bug by Friday",
            )
        )
        ctx = recall.build_context("anything")
        assert "auth bug" in ctx

    def test_budget_trimming(self, kortex_db, kortex_config, recall):
        for i in range(50):
            kortex_db.insert_episode(
                Episode(
                    session_id="s1",
                    summary=f"very important episode number {i} with lots of details "
                    * 5,
                    salience=0.8,
                    user_text="x" * 500,
                )
            )
        kortex_config.total_budget = 200
        recall_small = Recall(kortex_db, kortex_config)
        ctx = recall_small.build_context("episode")
        token_estimate = len(ctx) // 4
        assert token_estimate <= 250

    def test_context_includes_conversation_summaries(self, kortex_db, recall):
        kortex_db.insert_conversation_summary(
            {
                "session_id": "s1",
                "summary_text": "Conversation covered: project architecture and testing strategy",
                "episode_count": 2,
                "key_entities": "Architecture,Pytest",
            }
        )
        ctx = recall.build_context("architecture")
        assert "Conversation summaries:" in ctx
        assert "testing strategy" in ctx


class TestEpisodeRanking:
    def test_salient_episodes_rank_higher(self, kortex_db, recall):
        kortex_db.insert_episode(
            Episode(
                session_id="s1",
                summary="boring chat",
                salience=0.1,
            )
        )
        kortex_db.insert_episode(
            Episode(
                session_id="s1",
                summary="critical incident",
                salience=0.9,
                valence=-2,
                arousal=0.8,
            )
        )
        ctx = recall.build_context("incident")
        lines = ctx.split("\n")
        memory_lines = [l for l in lines if l.startswith("- ")]
        assert len(memory_lines) >= 1
        assert "critical" in memory_lines[0].lower()

    def test_emotional_episodes_recalled(self, kortex_db, recall):
        kortex_db.insert_episode(
            Episode(
                session_id="s1",
                summary="user called me an idiot at 23:19",
                salience=0.8,
                valence=-2,
                arousal=0.9,
            )
        )
        ctx = recall.build_context("frustration")
        assert "idiot" in ctx


class TestRecallText:
    def test_episode_recall_format(self):
        ep = Episode(
            summary="argued about deployment strategy",
            valence=-1,
            timestamp=time.time() - (3 * 86400),
        )
        text = ep.to_recall_text()
        assert "3 days ago" in text
        assert "frustrated" in text

    def test_old_episode_includes_date(self):
        ep = Episode(
            summary="first meeting",
            valence=1,
            timestamp=time.time() - (63 * 86400),
        )
        text = ep.to_recall_text()
        assert "9 weeks ago" in text
        assert "warm" in text
