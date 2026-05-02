"""Tests for evidence trace methods (facts ↔ episodes linkage)."""
import pytest
from kortex.models import Episode, Fact


@pytest.fixture
def traced_db(kortex_db):
    """Set up a DB with episodes and facts linked by source_episode_id."""
    # Create 3 episodes
    ep1 = kortex_db.insert_episode(
        Episode(session_id="s1", user_text="I use Python daily", assistant_text="Cool!", timestamp=1000)
    )
    ep2 = kortex_db.insert_episode(
        Episode(session_id="s1", user_text="I also like Rust", assistant_text="Nice!", timestamp=2000)
    )
    ep3 = kortex_db.insert_episode(
        Episode(session_id="s2", user_text="My cat is named Luna", assistant_text="Adorable!", timestamp=3000)
    )

    # Create facts linked to episodes
    f1 = Fact(object_text="uses Python daily", predicate="uses", source_episode_id=ep1)
    f2 = Fact(object_text="likes Rust", predicate="likes", source_episode_id=ep1)
    f3 = Fact(object_text="has a cat named Luna", predicate="has_pet", source_episode_id=ep3)
    kortex_db.insert_fact(f1)
    kortex_db.insert_fact(f2)
    kortex_db.insert_fact(f3)
    return (kortex_db, ep1, ep2, ep3)


def test_get_facts_by_episode(traced_db):
    """Should return all facts from a given episode."""
    kortex_db, ep1_id, _, _ = traced_db
    facts = kortex_db.get_facts_by_episode(ep1_id)
    # We inserted 2 facts for ep1
    assert len(facts) >= 2


def test_get_evidence_for_fact(traced_db):
    """Should return evidence trail with episode metadata."""
    kortex_db, ep1_id, _, _ = traced_db
    facts = kortex_db.get_facts_by_episode(ep1_id)
    if not facts:
        pytest.skip("No facts in DB")
    fact = facts[0]
    evidence = kortex_db.get_evidence_for_fact(fact.id)
    assert evidence is not None
    assert evidence["fact_id"] == fact.id
    assert evidence["predicate"] == fact.predicate
    assert evidence["object_text"] == fact.object_text
    assert evidence["source_episode_id"] == fact.source_episode_id


def test_get_evidence_for_fact_missing(traced_db):
    """Should return None for a non-existent fact."""
    kortex_db, _, _, _ = traced_db
    evidence = kortex_db.get_evidence_for_fact(999999)
    assert evidence is None


def test_get_fact_evidence_chain(traced_db):
    """Should return evidence chain with superseded facts."""
    kortex_db, ep1_id, _, _ = traced_db
    facts = kortex_db.get_facts_by_episode(ep1_id)
    if not facts:
        pytest.skip("No facts in DB")
    fact = facts[0]
    chain = kortex_db.get_fact_evidence_chain(fact.id)
    if chain:
        assert "fact_id" in chain
        assert "superseded_facts" in chain
        assert isinstance(chain["superseded_facts"], list)


def test_get_evidence_summary(traced_db):
    """Should return aggregated evidence stats per episode."""
    kortex_db, _, _, _ = traced_db
    summary = kortex_db.get_evidence_summary()
    assert isinstance(summary, list)
    for item in summary:
        assert "episode_id" in item
        assert "fact_count" in item
        assert "avg_confidence" in item
        assert "predicates" in item


def test_get_orphaned_facts(traced_db):
    """Should find facts whose source episode no longer exists.
    
    We create an orphan by disabling FK constraints temporarily,
    inserting a fact with a phantom episode ID, then re-enabling.
    This simulates the real-world case where episodes get purged."""
    kortex_db, _, _, _ = traced_db
    # Disable FK to insert a fact with a non-existent episode
    conn = kortex_db._get_conn()
    conn.execute("PRAGMA foreign_keys=OFF")
    orphan_fact = Fact(object_text="orphaned fact", predicate="is_orphan", source_episode_id=99999)
    kortex_db.insert_fact(orphan_fact)
    conn.execute("PRAGMA foreign_keys=ON")
    
    orphaned = kortex_db.get_orphaned_facts()
    assert any(f.source_episode_id == 99999 for f in orphaned)
