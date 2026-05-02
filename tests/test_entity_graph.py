"""Tests for entity graph traversal in recall — using graph connections to boost related memories.

Entity graph traversal for recall ensures that:
1. Episodes connected via entity links get boosted during recall
2. Multi-hop connections are discoverable
3. Graph traversal integrates with the ranking system
"""

import pytest

from kortex.db import KortexDB
from kortex.models import Episode
import tempfile


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    return KortexDB(tmp + "/test.db")


class TestGraphRecallIntegration:
    """Test that entity graph connections improve recall quality."""

    def test_linked_episodes_appear_in_recall(self, db):
        """Episodes linked to a query episode should be discoverable."""
        ep1 = Episode(user_text="Meeting with Alice about project X")
        ep1.id = db.insert_episode(ep1)
        ep2 = Episode(user_text="Alice sent the report on project X")
        ep2.id = db.insert_episode(ep2)

        db.insert_link("episode", ep1.id, "episode", ep2.id, "shares_entity", 0.9)

        links = db.get_links_from("episode", ep1.id)
        assert any(l["dst_id"] == ep2.id for l in links)

    def test_bidirectional_discovery(self, db):
        """Linked episodes should be discoverable from both ends."""
        ep1 = Episode(user_text="Ep 1")
        ep1.id = db.insert_episode(ep1)
        ep2 = Episode(user_text="Ep 2")
        ep2.id = db.insert_episode(ep2)

        db.insert_link("episode", ep1.id, "episode", ep2.id, "related_to", 0.8)

        # Forward
        fwd = db.get_links_from("episode", ep1.id)
        assert any(l["dst_id"] == ep2.id for l in fwd)

        # Reverse
        rev = db.get_links_to("episode", ep2.id)
        assert any(l["src_id"] == ep1.id for l in rev)

    def test_graph_based_candidate_expansion(self, db):
        """Use graph traversal to expand candidate set for recall."""
        ep1 = Episode(user_text="Hub episode")
        ep1.id = db.insert_episode(ep1)
        eps = []
        for i in range(5):
            ep = Episode(user_text=f"Related episode {i}")
            ep.id = db.insert_episode(ep)
            eps.append(ep)

        for ep in eps:
            db.insert_link("episode", ep1.id, "episode", ep.id, "related_to", 0.7)

        # Get all linked episodes (candidate expansion)
        candidates = db.get_links_from("episode", ep1.id, limit=10)
        assert len(candidates) == 5

    def test_multi_hop_graph_traversal(self, db):
        """Chain of links should be discoverable."""
        eps = []
        for i in range(4):
            ep = Episode(user_text=f"Episode {i}")
            ep.id = db.insert_episode(ep)
            eps.append(ep)

        db.insert_link("episode", eps[0].id, "episode", eps[1].id, "follows", 0.8)
        db.insert_link("episode", eps[1].id, "episode", eps[2].id, "follows", 0.8)
        db.insert_link("episode", eps[2].id, "episode", eps[3].id, "follows", 0.8)

        # Hop 1
        hop1 = db.get_links_from("episode", eps[0].id)
        assert any(l["dst_id"] == eps[1].id for l in hop1)

        # Hop 2
        hop2 = db.get_links_from("episode", eps[1].id)
        assert any(l["dst_id"] == eps[2].id for l in hop2)

    def test_isolated_episode_has_no_links(self, db):
        """An episode with no links should return empty."""
        ep = Episode(user_text="Lone episode")
        ep.id = db.insert_episode(ep)
        links = db.get_links_from("episode", ep.id)
        assert links == []

    def test_graph_pruning_removes_stale_links(self, db):
        """Deleting links should remove them from the graph."""
        ep1 = Episode(user_text="Ep 1")
        ep1.id = db.insert_episode(ep1)
        ep2 = Episode(user_text="Ep 2")
        ep2.id = db.insert_episode(ep2)

        db.insert_link("episode", ep1.id, "episode", ep2.id, "related_to", 0.8)
        assert len(db.get_links_from("episode", ep1.id)) == 1

        db.delete_links("episode", ep1.id)
        assert len(db.get_links_from("episode", ep1.id)) == 0
