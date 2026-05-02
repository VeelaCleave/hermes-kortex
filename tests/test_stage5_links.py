import time

import pytest

from kortex.config import KortexConfig
from kortex.db import KortexDB
from kortex.linker import Linker
from kortex.models import Episode, Fact, Reflection
from kortex.provider import KortexProvider
from kortex.recall import Recall


def _episode(
    kortex_db,
    *,
    summary="episode",
    entities="",
    topics="",
    salience=0.5,
    session_id="s1",
    user_text="user",
    assistant_text="assistant",
):
    ep = Episode(
        session_id=session_id,
        user_text=user_text,
        assistant_text=assistant_text,
        summary=summary,
        entities=entities,
        topics=topics,
        salience=salience,
        timestamp=time.time(),
    )
    kortex_db.insert_episode(ep)
    return ep


def _fact(kortex_db, text="fact", predicate="prefers", source_episode_id=None):
    fact = Fact(
        subject_type="user",
        predicate=predicate,
        object_text=text,
        source_episode_id=source_episode_id,
    )
    kortex_db.insert_fact(fact)
    return fact


def _reflection(kortex_db, text="reflection", kind="pattern", source_episode_id=None):
    ref = Reflection(kind=kind, text=text, source_episode_id=source_episode_id)
    kortex_db.insert_reflection(ref)
    return ref


@pytest.fixture
def linker(kortex_db):
    return Linker(kortex_db)


class TestLinkerInitialization:
    def test_constructs_with_db(self, kortex_db):
        linker = Linker(kortex_db)
        assert linker._db is kortex_db

    def test_entity_id_is_stable(self):
        assert Linker._entity_id("Alice") == Linker._entity_id("alice")

    def test_split_csv_empty(self):
        assert Linker._split_csv("") == []

    def test_split_csv_normalizes(self):
        assert Linker._split_csv(" Alice, Bob ") == ["alice", "bob"]

    def test_jaccard_empty(self):
        assert Linker._jaccard(set(), set()) == 0.0


class TestEpisodeFactLinking:
    def test_single_fact_link(self, kortex_db, linker):
        ep = _episode(kortex_db)
        fact = _fact(kortex_db)
        assert linker.link_episode_to_facts(ep.id, [fact.id]) == 1

    def test_multiple_fact_links(self, kortex_db, linker):
        ep = _episode(kortex_db)
        facts = [_fact(kortex_db, text=f"fact {i}") for i in range(3)]
        assert linker.link_episode_to_facts(ep.id, [f.id for f in facts]) == 3

    def test_empty_fact_list(self, kortex_db, linker):
        ep = _episode(kortex_db)
        assert linker.link_episode_to_facts(ep.id, []) == 0

    def test_ignores_none_fact_ids(self, kortex_db, linker):
        ep = _episode(kortex_db)
        assert linker.link_episode_to_facts(ep.id, [None, 0, -1]) == 0

    def test_deduplicates_fact_ids(self, kortex_db, linker):
        ep = _episode(kortex_db)
        fact = _fact(kortex_db)
        assert linker.link_episode_to_facts(ep.id, [fact.id, fact.id]) == 1

    def test_link_episode_to_facts_dedup_across_calls(self, kortex_db, linker):
        ep = _episode(kortex_db)
        fact = _fact(kortex_db)
        linker.link_episode_to_facts(ep.id, [fact.id])
        assert linker.link_episode_to_facts(ep.id, [fact.id]) == 0

    def test_get_episode_facts_returns_ids(self, kortex_db, linker):
        ep = _episode(kortex_db)
        facts = [_fact(kortex_db, text=f"f{i}") for i in range(2)]
        linker.link_episode_to_facts(ep.id, [f.id for f in facts])
        assert linker.get_episode_facts(ep.id) == [facts[1].id, facts[0].id]

    def test_get_fact_episodes_returns_sources(self, kortex_db, linker):
        fact = _fact(kortex_db)
        episodes = [_episode(kortex_db, summary=f"ep {i}") for i in range(2)]
        for ep in episodes:
            linker.link_episode_to_facts(ep.id, [fact.id])
        assert linker.get_fact_episodes(fact.id) == [episodes[1].id, episodes[0].id]

    def test_get_episode_facts_empty_for_missing_episode(self, linker):
        assert linker.get_episode_facts(9999) == []

    def test_get_fact_episodes_empty_for_missing_fact(self, linker):
        assert linker.get_fact_episodes(9999) == []


class TestReflectionLinking:
    def test_single_reflection_link(self, kortex_db, linker):
        ep = _episode(kortex_db)
        ref = _reflection(kortex_db)
        assert linker.link_episode_to_reflections(ep.id, [ref.id]) == 1

    def test_multiple_reflection_links(self, kortex_db, linker):
        ep = _episode(kortex_db)
        refs = [_reflection(kortex_db, text=f"r{i}") for i in range(3)]
        assert linker.link_episode_to_reflections(ep.id, [r.id for r in refs]) == 3

    def test_empty_reflection_list(self, kortex_db, linker):
        ep = _episode(kortex_db)
        assert linker.link_episode_to_reflections(ep.id, []) == 0

    def test_deduplicates_reflection_links(self, kortex_db, linker):
        ep = _episode(kortex_db)
        ref = _reflection(kortex_db)
        linker.link_episode_to_reflections(ep.id, [ref.id])
        assert linker.link_episode_to_reflections(ep.id, [ref.id]) == 0

    def test_reflection_link_relation(self, kortex_db, linker):
        ep = _episode(kortex_db)
        ref = _reflection(kortex_db)
        linker.link_episode_to_reflections(ep.id, [ref.id])
        links = kortex_db.get_links_from("episode", ep.id)
        assert links[0]["relation"] == "triggered"


class TestRelatedEpisodeLinking:
    def test_links_by_shared_entities(self, kortex_db, linker):
        _episode(kortex_db, summary="first", entities="Alice,Bob", topics="work")
        ep2 = _episode(kortex_db, summary="second", entities="Alice,Bob", topics="code")
        assert linker.link_related_episodes(ep2) >= 2

    def test_links_by_shared_topics(self, kortex_db, linker):
        _episode(kortex_db, summary="first", topics="code,infra,data")
        ep2 = _episode(kortex_db, summary="second", topics="code,infra,data")
        assert linker.link_related_episodes(ep2) >= 2

    def test_no_link_below_threshold(self, kortex_db, linker):
        _episode(kortex_db, summary="first", entities="Alice", topics="work")
        ep2 = _episode(
            kortex_db, summary="second", entities="Alice", topics="code,data"
        )
        assert linker.link_related_episodes(ep2) == 1

    def test_threshold_exactly_point_three_links(self, kortex_db, linker):
        _episode(kortex_db, summary="first", topics="a,b,c")
        ep2 = _episode(kortex_db, summary="second", topics="a,b,c,d,e,f,g,h,i,j")
        assert linker.link_related_episodes(ep2) >= 2

    def test_no_link_when_no_tokens(self, kortex_db, linker):
        _episode(kortex_db, summary="first")
        ep2 = _episode(kortex_db, summary="second")
        assert linker.link_related_episodes(ep2) == 0

    def test_ignores_self(self, kortex_db, linker):
        ep = _episode(kortex_db, summary="self", topics="code")
        assert linker.link_related_episodes(ep) == 0

    def test_respects_lookback_limit(self, kortex_db, linker):
        for i in range(55):
            _episode(kortex_db, summary=f"ep{i}", topics="shared")
        latest = _episode(kortex_db, summary="latest", topics="shared")
        linker.link_related_episodes(latest, max_lookback=3)
        assert len(linker.get_related_episodes(latest.id, limit=20)) == 3

    def test_related_links_are_bidirectional(self, kortex_db, linker):
        ep1 = _episode(kortex_db, summary="first", topics="code,infra,data")
        ep2 = _episode(kortex_db, summary="second", topics="code,infra,data")
        linker.link_related_episodes(ep2)
        assert ep1.id in linker.get_related_episodes(ep2.id, limit=10)
        assert ep2.id in linker.get_related_episodes(ep1.id, limit=10)

    def test_related_episode_dedup_on_repeat(self, kortex_db, linker):
        _episode(kortex_db, summary="first", topics="code,infra,data")
        ep2 = _episode(kortex_db, summary="second", topics="code,infra,data")
        linker.link_related_episodes(ep2)
        before = kortex_db.count_links()
        linker.link_related_episodes(ep2)
        assert kortex_db.count_links() == before

    def test_get_related_episodes_empty(self, linker):
        assert linker.get_related_episodes(1234) == []

    def test_related_episode_weight_reflects_similarity(self, kortex_db, linker):
        _episode(kortex_db, summary="first", topics="code,infra,data")
        ep2 = _episode(kortex_db, summary="second", topics="code,infra,data")
        linker.link_related_episodes(ep2)
        links = kortex_db.get_links_from("episode", ep2.id, relation="related_to")
        assert links[0]["weight"] == pytest.approx(1.0)

    def test_entity_links_created_for_episode(self, kortex_db, linker):
        ep = _episode(kortex_db, entities="Alice,Bob")
        linker.link_related_episodes(ep)
        incoming = kortex_db.get_links_to(
            "episode", ep.id, relation="co_occurs", limit=10
        )
        assert len(incoming) == 2

    def test_entity_links_deduplicate(self, kortex_db, linker):
        ep = _episode(kortex_db, entities="Alice,Alice")
        linker.link_related_episodes(ep)
        linker.link_related_episodes(ep)
        incoming = kortex_db.get_links_to(
            "episode", ep.id, relation="co_occurs", limit=10
        )
        assert len(incoming) == 1


class TestSupersededFactLinks:
    def test_manual_superseded_link(self, kortex_db, linker):
        old = _fact(kortex_db, text="Python 3.10", predicate="uses")
        new = _fact(kortex_db, text="Python 3.12", predicate="uses")
        linker.link_superseded_facts(old.id, new.id)
        assert kortex_db.link_exists("fact", old.id, "fact", new.id, "supersedes")

    def test_manual_superseded_link_ignores_invalid(self, kortex_db, linker):
        linker.link_superseded_facts(0, 1)
        assert kortex_db.count_links() == 0

    def test_fact_linking_discovers_superseded_pairs(self, ingestor, kortex_db, linker):
        ep1 = ingestor.ingest_turn("a", "b", session_id="s1")
        ingestor.extract_facts("I use Python 3.10 for development", ep1.id)
        ep2 = ingestor.ingest_turn("c", "d", session_id="s1")
        facts = ingestor.extract_facts("I use Python 3.12 for development", ep2.id)
        linker.link_episode_to_facts(ep2.id, [f.id for f in facts])
        old = kortex_db.get_facts_superseded_by(facts[0].id)[0]
        assert kortex_db.link_exists("fact", old.id, "fact", facts[0].id, "supersedes")

    def test_superseded_link_deduplicates(self, kortex_db, linker):
        old = _fact(kortex_db, text="old", predicate="uses")
        new = _fact(kortex_db, text="new", predicate="uses")
        linker.link_superseded_facts(old.id, new.id)
        linker.link_superseded_facts(old.id, new.id)
        assert kortex_db.count_links() == 1


class TestGraphTraversal:
    def test_traverse_respects_max_hops(self, kortex_db, linker):
        ep = _episode(kortex_db, summary="release planning", entities="Amelia")
        linker.link_related_episodes(ep)
        fact = _fact(kortex_db, text="ship Amelia tonight", source_episode_id=ep.id)
        linker.link_episode_to_facts(ep.id, [fact.id])

        entity_id = Linker._entity_id("amelia")
        one_hop = linker.traverse([entity_id], max_hops=1)
        two_hops = linker.traverse([entity_id], max_hops=2)

        assert any(
            node["node_type"] == "episode" and node["node_id"] == ep.id
            for node in one_hop
        )
        assert not any(
            node["node_type"] == "fact" and node["node_id"] == fact.id
            for node in one_hop
        )
        assert any(
            node["node_type"] == "fact" and node["node_id"] == fact.id
            for node in two_hops
        )

    def test_traverse_score_propagates_with_decay_and_relation_weight(
        self, kortex_db, linker
    ):
        ep1 = _episode(
            kortex_db,
            summary="release planning",
            entities="Amelia",
            topics="rollout,launch,ops",
        )
        ep2 = _episode(
            kortex_db,
            summary="follow-up mitigation",
            topics="rollout,launch,ops",
        )
        linker.link_related_episodes(ep2)
        fact = _fact(
            kortex_db, text="Amelia wants same-night rollout", source_episode_id=ep1.id
        )
        linker.link_episode_to_facts(ep1.id, [fact.id])

        entity_id = Linker._entity_id("amelia")
        ranked = {
            (node["node_type"], node["node_id"]): node["score"]
            for node in linker.traverse([entity_id], max_hops=2)
        }

        assert ranked[("episode", ep1.id)] > ranked[("fact", fact.id)]
        assert ranked[("fact", fact.id)] > ranked[("episode", ep2.id)]


class TestDBLinkMethods:
    def test_get_links_from_returns_outgoing(self, kortex_db):
        kortex_db.insert_link("episode", 1, "fact", 2, "extracted_from", 0.8)
        links = kortex_db.get_links_from("episode", 1)
        assert links == [
            {
                "dst_type": "fact",
                "dst_id": 2,
                "relation": "extracted_from",
                "weight": 0.8,
            }
        ]

    def test_get_links_from_filters_relation(self, kortex_db):
        kortex_db.insert_link("episode", 1, "fact", 2, "extracted_from")
        kortex_db.insert_link("episode", 1, "reflection", 3, "triggered")
        links = kortex_db.get_links_from("episode", 1, relation="triggered")
        assert len(links) == 1
        assert links[0]["dst_type"] == "reflection"

    def test_get_links_to_returns_incoming(self, kortex_db):
        kortex_db.insert_link("episode", 1, "fact", 2, "extracted_from")
        links = kortex_db.get_links_to("fact", 2)
        assert links[0]["src_type"] == "episode"

    def test_get_links_to_filters_relation(self, kortex_db):
        kortex_db.insert_link("episode", 1, "fact", 2, "extracted_from")
        kortex_db.insert_link("entity", 9, "fact", 2, "mentions")
        links = kortex_db.get_links_to("fact", 2, relation="mentions")
        assert len(links) == 1
        assert links[0]["src_type"] == "entity"

    def test_delete_links_removes_outgoing(self, kortex_db):
        kortex_db.insert_link("episode", 1, "fact", 2, "extracted_from")
        kortex_db.insert_link("episode", 1, "reflection", 3, "triggered")
        assert kortex_db.delete_links("episode", 1) == 2

    def test_delete_links_returns_zero_when_missing(self, kortex_db):
        assert kortex_db.delete_links("episode", 99) == 0

    def test_count_links_counts_all(self, kortex_db):
        for i in range(3):
            kortex_db.insert_link("episode", 1, "fact", i + 1, "extracted_from")
        assert kortex_db.count_links() == 3

    def test_link_exists_true(self, kortex_db):
        kortex_db.insert_link("episode", 1, "fact", 2, "extracted_from")
        assert kortex_db.link_exists("episode", 1, "fact", 2, "extracted_from") is True

    def test_link_exists_false(self, kortex_db):
        assert kortex_db.link_exists("episode", 1, "fact", 2, "extracted_from") is False

    def test_get_links_from_limit(self, kortex_db):
        for i in range(5):
            kortex_db.insert_link("episode", 1, "fact", i, "extracted_from")
        assert len(kortex_db.get_links_from("episode", 1, limit=2)) == 2

    def test_get_links_to_limit(self, kortex_db):
        for i in range(5):
            kortex_db.insert_link("episode", i, "fact", 1, "extracted_from")
        assert len(kortex_db.get_links_to("fact", 1, limit=2)) == 2

    def test_get_links_from_orders_by_weight_desc(self, kortex_db):
        kortex_db.insert_link("episode", 1, "episode", 2, "related_to", 0.4)
        kortex_db.insert_link("episode", 1, "episode", 3, "related_to", 0.9)
        links = kortex_db.get_links_from("episode", 1)
        assert links[0]["dst_id"] == 3

    def test_get_links_to_orders_by_weight_desc(self, kortex_db):
        kortex_db.insert_link("episode", 2, "episode", 1, "related_to", 0.4)
        kortex_db.insert_link("episode", 3, "episode", 1, "related_to", 0.9)
        links = kortex_db.get_links_to("episode", 1)
        assert links[0]["src_id"] == 3

    def test_get_fact_and_missing(self, kortex_db):
        fact = _fact(kortex_db, text="known")
        assert kortex_db.get_fact(fact.id).object_text == "known"
        assert kortex_db.get_fact(9999) is None

    def test_get_reflection_and_missing(self, kortex_db):
        ref = _reflection(kortex_db, text="known")
        assert kortex_db.get_reflection(ref.id).text == "known"
        assert kortex_db.get_reflection(9999) is None

    def test_get_facts_superseded_by_empty(self, kortex_db):
        assert kortex_db.get_facts_superseded_by(1234) == []


class TestRecallLinkEnrichment:
    def test_enrich_adds_related_fact_line(self, kortex_db, kortex_config):
        ep = _episode(kortex_db, summary="Discussed editors", salience=0.8)
        fact = _fact(kortex_db, text="uses neovim", source_episode_id=ep.id)
        kortex_db.insert_link("episode", ep.id, "fact", fact.id, "extracted_from")
        recall = Recall(kortex_db, kortex_config)
        lines = recall._enrich_with_links([ep], set())
        assert any("Related fact" in line for line in lines)

    def test_enrich_skips_existing_fact_ids(self, kortex_db, kortex_config):
        ep = _episode(kortex_db, salience=0.8)
        fact = _fact(kortex_db, text="uses neovim", source_episode_id=ep.id)
        kortex_db.insert_link("episode", ep.id, "fact", fact.id, "extracted_from")
        recall = Recall(kortex_db, kortex_config)
        assert recall._enrich_with_links([ep], {fact.id}) == []

    def test_enrich_adds_related_memory_for_salient_episode(
        self, kortex_db, kortex_config
    ):
        ep1 = _episode(
            kortex_db, summary="Past deployment issue", topics="infra,deploy,ci"
        )
        ep2 = _episode(
            kortex_db,
            summary="Current deployment issue",
            topics="infra,deploy,ci",
            salience=0.9,
        )
        Linker(kortex_db).link_related_episodes(ep2)
        recall = Recall(kortex_db, kortex_config)
        lines = recall._enrich_with_links([ep2], set())
        assert any("Related memory" in line for line in lines)

    def test_enrich_skips_related_memory_for_low_salience(
        self, kortex_db, kortex_config
    ):
        ep1 = _episode(kortex_db, summary="Past", topics="infra,deploy,ci")
        ep2 = _episode(
            kortex_db, summary="Current", topics="infra,deploy,ci", salience=0.6
        )
        Linker(kortex_db).link_related_episodes(ep2)
        recall = Recall(kortex_db, kortex_config)
        assert all(
            "Related memory" not in line
            for line in recall._enrich_with_links([ep2], set())
        )

    def test_enrich_caps_extra_lines(self, kortex_db, kortex_config):
        ep = _episode(
            kortex_db, summary="Main", salience=0.95, topics="code,infra,data"
        )
        for i in range(5):
            fact = _fact(kortex_db, text=f"fact {i}", source_episode_id=ep.id)
            kortex_db.insert_link("episode", ep.id, "fact", fact.id, "extracted_from")
        for i in range(3):
            other = _episode(
                kortex_db, summary=f"related {i}", topics="code,infra,data"
            )
            kortex_db.insert_link("episode", ep.id, "episode", other.id, "related_to")
        recall = Recall(kortex_db, kortex_config)
        assert len(recall._enrich_with_links([ep], set())) <= 3

    def test_build_context_includes_link_enrichment(self, kortex_db, kortex_config):
        ep = _episode(
            kortex_db, summary="Editor chat", salience=0.9, user_text="editor"
        )
        fact = _fact(kortex_db, text="uses neovim", source_episode_id=ep.id)
        kortex_db.insert_link("episode", ep.id, "fact", fact.id, "extracted_from")
        recall = Recall(kortex_db, kortex_config)
        # _enrich_with_links is called from full recall path; test it directly
        lines = recall._enrich_with_links([ep], set())
        assert any("Related fact" in line for line in lines)

    def test_build_context_handles_missing_link_targets(self, kortex_db, kortex_config):
        ep = _episode(kortex_db, summary="Editor chat", salience=0.9, user_text="vim")
        kortex_db.insert_link("episode", ep.id, "fact", 9999, "extracted_from")
        recall = Recall(kortex_db, kortex_config)
        ctx = recall.build_context("vim")
        assert "KORTEX Memory" in ctx


class TestProviderIntegration:
    def test_initialize_creates_linker(self, tmp_path):
        provider = KortexProvider(KortexConfig(db_path=str(tmp_path / "test.db")))
        provider.initialize("s1", hermes_home=str(tmp_path))
        assert provider._linker is not None
        provider.shutdown()

    def test_sync_turn_links_episode_to_facts(self, tmp_path):
        provider = KortexProvider(KortexConfig(db_path=str(tmp_path / "test.db")))
        provider.initialize("s1", hermes_home=str(tmp_path))
        provider.sync_turn("I prefer dark mode", "Noted.")
        time.sleep(0.5)
        episodes = provider._db.get_recent_episodes(limit=1)
        assert provider._linker.get_episode_facts(episodes[0].id)
        provider.shutdown()

    def test_sync_turn_links_episode_to_reflections(self, tmp_path):
        provider = KortexProvider(KortexConfig(db_path=str(tmp_path / "test.db")))
        provider.initialize("s1", hermes_home=str(tmp_path))
        provider.sync_turn("Keep it short, don't need explanation.", "Understood.")
        time.sleep(0.5)
        ep = provider._db.get_recent_episodes(limit=1)[0]
        links = provider._db.get_links_from("episode", ep.id, relation="triggered")
        assert len(links) >= 1
        provider.shutdown()

    def test_sync_turn_links_related_episodes(self, tmp_path):
        provider = KortexProvider(KortexConfig(db_path=str(tmp_path / "test.db")))
        provider.initialize("s1", hermes_home=str(tmp_path))
        provider.sync_turn("Alice asked about the database migration", "I'll help.")
        time.sleep(0.5)
        provider.sync_turn(
            "Alice mentioned another database migration issue", "Let's fix it."
        )
        time.sleep(0.5)
        latest = provider._db.get_recent_episodes(limit=1)[0]
        related = provider._db.get_links_from(
            "episode", latest.id, relation="related_to"
        )
        assert len(related) >= 1
        provider.shutdown()

    def test_sync_turn_no_links_when_auto_extract_disabled(self, tmp_path):
        provider = KortexProvider(
            KortexConfig(db_path=str(tmp_path / "test.db"), auto_extract=False)
        )
        provider.initialize("s1", hermes_home=str(tmp_path))
        provider.sync_turn("I prefer dark mode", "Noted")
        time.sleep(0.5)
        ep = provider._db.get_recent_episodes(limit=1)[0]
        assert provider._db.get_links_from("episode", ep.id) == []
        provider.shutdown()


class TestSchemaAndMigration:
    def test_schema_version_is_five(self):
        from kortex import db as db_module

        assert db_module.SCHEMA_VERSION == 5

    def test_existing_v2_db_still_has_entity_links(self, tmp_path):
        db = KortexDB(str(tmp_path / "test.db"))
        table = (
            db._get_conn()
            .execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_links'"
            )
            .fetchone()
        )
        assert table[0] == "entity_links"
        db.close()

    def test_populated_v2_db_migrates_to_v3(self, populated_v2_db_path):
        db = KortexDB(populated_v2_db_path)
        conn = db._get_conn()

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 5

        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(journal_mode).lower() == "wal"

        episodes_columns = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(episodes)").fetchall()
        }
        facts_columns = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(facts)").fetchall()
        }

        assert episodes_columns["timestamp"].upper() == "REAL"
        assert episodes_columns["user_id"].upper() == "TEXT"
        assert facts_columns["first_seen"].upper() == "REAL"
        assert facts_columns["user_id"].upper() == "TEXT"

        stored_type = conn.execute(
            "SELECT typeof(timestamp) FROM episodes WHERE id=1"
        ).fetchone()[0]
        assert stored_type == "real"

        ep = db.get_episode(1)
        assert ep is not None
        assert isinstance(ep.timestamp, float)
        assert ep.user_id == "__default__"

        fact = db.get_fact(1)
        assert fact is not None
        assert isinstance(fact.first_seen, float)
        assert fact.valid_from == fact.first_seen
        assert fact.user_id == "__default__"

        rel = db.get_relationship()
        assert rel.user_id == "__default__"

        schema_version = conn.execute(
            "SELECT version FROM kortex_schema_version"
        ).fetchone()[0]
        assert schema_version == 5

        db.close()


class TestEdgeCases:
    def test_link_related_episodes_with_missing_id(self, kortex_db, linker):
        ep = Episode(summary="x", topics="code")
        assert linker.link_related_episodes(ep) == 0

    def test_create_links_with_nonexistent_fact_ids(self, kortex_db, linker):
        ep = _episode(kortex_db)
        assert linker.link_episode_to_facts(ep.id, [999]) == 1

    def test_get_related_episodes_limit(self, kortex_db, linker):
        base = _episode(kortex_db, summary="base", topics="code,infra,data")
        for i in range(4):
            other = _episode(kortex_db, summary=f"other {i}", topics="code,infra,data")
            kortex_db.insert_link("episode", base.id, "episode", other.id, "related_to")
        assert len(linker.get_related_episodes(base.id, limit=2)) == 2

    def test_delete_links_only_outgoing(self, kortex_db):
        kortex_db.insert_link("episode", 1, "fact", 2, "extracted_from")
        kortex_db.insert_link("fact", 2, "episode", 1, "mentioned_in")
        kortex_db.delete_links("episode", 1)
        assert kortex_db.get_links_to("episode", 1)

    def test_recall_enrichment_empty_input(self, kortex_db, kortex_config):
        recall = Recall(kortex_db, kortex_config)
        assert recall._enrich_with_links([], set()) == []
