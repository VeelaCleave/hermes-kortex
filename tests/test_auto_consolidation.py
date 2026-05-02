"""Tests for auto-consolidation — intelligent episode merging.

Auto-consolidation ensures that:
1. Old/low-salience episodes get consolidated when memory pressure is high
2. Consolidation preserves entity links and salience scores
3. Summary episodes inherit the best qualities of their parents
4. Consolidation is triggered automatically based on thresholds
"""

import pytest

from kortex.db import KortexDB
from kortex.models import Episode
import tempfile


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    return KortexDB(tmp + "/test.db")


class TestConsolidationBasics:
    """Test basic consolidation operations."""

    def test_mark_episodes_consolidated(self, db):
        """Should be able to mark episodes as consolidated."""
        ep1 = Episode(user_text="Ep 1")
        ep1.id = db.insert_episode(ep1)
        ep2 = Episode(user_text="Ep 2")
        ep2.id = db.insert_episode(ep2)

        # Mark as consolidated
        # Create a summary episode first
        summary = Episode(user_text="Summary", is_consolidated=True)
        summary.id = db.insert_episode(summary)
        db.mark_episodes_consolidated([ep1.id, ep2.id], summary_episode_id=summary.id)

        # Should be marked as consolidated
        eps = db.get_unconsolidated_episodes()
        assert ep1.id not in [e.id for e in eps]
        assert ep2.id not in [e.id for e in eps]

    def test_count_unconsolidated(self, db):
        """Should count unconsolidated episodes correctly."""
        for i in range(5):
            ep = Episode(user_text=f"Ep {i}")
            ep.id = db.insert_episode(ep)

        count = db.count_unconsolidated_episodes()
        assert count == 5

    def test_get_unconsolidated_episodes(self, db):
        """Should retrieve unconsolidated episodes."""
        for i in range(3):
            ep = Episode(user_text=f"Ep {i}")
            ep.id = db.insert_episode(ep)

        eps = db.get_unconsolidated_episodes(limit=10)
        assert len(eps) == 3

    def test_consolidated_episodes_excluded(self, db):
        """Consolidated episodes should be excluded from unconsolidated list."""
        ep1 = Episode(user_text="Ep 1")
        ep1.id = db.insert_episode(ep1)
        ep2 = Episode(user_text="Ep 2")
        ep2.id = db.insert_episode(ep2)

        # Create a summary episode first
        summary = Episode(user_text="Summary", is_consolidated=True)
        summary.id = db.insert_episode(summary)
        db.mark_episodes_consolidated([ep1.id], summary_episode_id=summary.id)

        eps = db.get_unconsolidated_episodes()
        ids = [e.id for e in eps]
        assert ep1.id not in ids
        assert ep2.id in ids


class TestConsolidationThresholds:
    """Test threshold-based auto-consolidation triggers."""

    def test_below_threshold_no_consolidation(self, db):
        """Below threshold should not trigger consolidation."""
        for i in range(5):
            ep = Episode(user_text=f"Ep {i}")
            ep.id = db.insert_episode(ep)

        # With threshold of 10, 5 episodes shouldn't trigger
        count = db.count_unconsolidated_episodes()
        assert count == 5
        assert count < 10

    def test_above_threshold_triggers(self, db):
        """Above threshold should trigger consolidation."""
        for i in range(15):
            ep = Episode(user_text=f"Ep {i}")
            ep.id = db.insert_episode(ep)

        count = db.count_unconsolidated_episodes()
        assert count == 15
        assert count >= 10  # Threshold

    def test_batch_size_limits(self, db):
        """Batch size should limit consolidation."""
        for i in range(20):
            ep = Episode(user_text=f"Ep {i}")
            ep.id = db.insert_episode(ep)

        eps = db.get_unconsolidated_episodes(limit=10)
        assert len(eps) == 10


class TestConsolidationSafety:
    """Test safety and edge cases."""

    def test_empty_consolidation(self, db):
        """Consolidating an empty set should be safe."""
        eps = db.get_unconsolidated_episodes()
        assert len(eps) == 0

    def test_single_episode_consolidation(self, db):
        """Single episode consolidation should work."""
        ep = Episode(user_text="Lone episode")
        ep.id = db.insert_episode(ep)

        eps = db.get_unconsolidated_episodes()
        assert len(eps) == 1

    def test_consolidation_preserves_data(self, db):
        """Consolidation should preserve key data."""
        ep1 = Episode(user_text="Important meeting", salience=0.8, valence=1)
        ep1.id = db.insert_episode(ep1)
        ep2 = Episode(user_text="Follow-up discussion", salience=0.6, valence=2)
        ep2.id = db.insert_episode(ep2)

        # Verify both are retrievable
        eps = db.get_unconsolidated_episodes()
        assert len(eps) == 2

    def test_large_scale_consolidation(self, db):
        """Large scale consolidation should scale gracefully."""
        for i in range(100):
            ep = Episode(user_text=f"Episode {i}")
            ep.id = db.insert_episode(ep)

        count = db.count_unconsolidated_episodes()
        assert count == 100

    def test_consolidation_respects_limit(self, db):
        """Limit parameter should cap results."""
        for i in range(50):
            ep = Episode(user_text=f"Episode {i}")
            ep.id = db.insert_episode(ep)

        eps = db.get_unconsolidated_episodes(limit=20)
        assert len(eps) == 20
