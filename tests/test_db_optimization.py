"""Tests for DB optimization methods."""
import pytest
from kortex.models import Episode, Fact


class TestOptimizeDatabase:
    """Test optimize_database() and get_db_stats()."""

    def test_optimize_returns_summary(self, kortex_db):
        result = kortex_db.optimize_database()
        assert isinstance(result, dict)
        assert "vacuumed" in result
        assert "reindexed" in result
        assert "analyzed" in result
        assert "indexes_added" in result

    def test_optimize_vacuums(self, kortex_db):
        result = kortex_db.optimize_database()
        assert result["vacuumed"] is True

    def test_optimize_reindexes(self, kortex_db):
        result = kortex_db.optimize_database()
        assert result["reindexed"] is True

    def test_optimize_analyzes(self, kortex_db):
        result = kortex_db.optimize_database()
        assert result["analyzed"] is True

    def test_optimize_creates_compound_indexes(self, kortex_db):
        result = kortex_db.optimize_database()
        assert len(result["indexes_added"]) > 0

    def test_optimize_is_idempotent(self, kortex_db):
        """Running twice shouldn't crash or duplicate indexes."""
        result1 = kortex_db.optimize_database()
        result2 = kortex_db.optimize_database()
        assert result2["vacuumed"] is True
        assert result2["reindexed"] is True

    def test_get_db_stats_returns_dict(self, kortex_db):
        stats = kortex_db.get_db_stats()
        assert isinstance(stats, dict)

    def test_get_db_stats_has_table_counts(self, kortex_db):
        stats = kortex_db.get_db_stats()
        assert "episodes_count" in stats
        assert "facts_count" in stats
        assert "open_loops_count" in stats

    def test_get_db_stats_schema_version(self, kortex_db):
        stats = kortex_db.get_db_stats()
        assert stats["schema_version"] == 5

    def test_get_db_stats_shows_inserted_data(self, kortex_db):
        ep = kortex_db.insert_episode(Episode(
            user_id="test", session_id="s1", timestamp=1000,
            user_text="test", salience=0.5, topics="test"
        ))
        kortex_db.insert_fact(Fact(
            object_text="test fact", user_id="test", source_episode_id=ep
        ))
        stats = kortex_db.get_db_stats()
        assert stats["episodes_count"] >= 1
        assert stats["facts_count"] >= 1
