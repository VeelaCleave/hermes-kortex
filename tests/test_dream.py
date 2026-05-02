"""Tests for kortex.dream — DayDream and REMSleep modes."""

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_db_with_data(tmp_path, name="test.db"):
    """Create a properly-initialized KortexDB with test data."""
    from kortex.db import KortexDB
    from kortex.models import Episode, Fact, OpenLoop, Reflection
    db_path = str(tmp_path / name)
    db = KortexDB(db_path)
    now = time.time()
    ep = Episode(session_id="test", user_text="hello", assistant_text="hi",
                 user_id="default", timestamp=now)
    db.insert_episode(ep)
    fact = Fact(object_text="test fact", user_id="default")
    db.insert_fact(fact)
    loop = OpenLoop(text="test loop", user_id="default")
    db.insert_open_loop(loop)
    refl = Reflection(text="test reflection", confidence=0.7, user_id="default")
    db.insert_reflection(refl)
    db.close()
    return db_path


def _make_empty_db(tmp_path, name="empty.db"):
    """Create an empty KortexDB."""
    from kortex.db import KortexDB
    db_path = str(tmp_path / name)
    db = KortexDB(db_path)
    db.close()
    return db_path


class TestGetDbStats:
    """Test get_db_stats helper function."""

    def test_returns_dict_with_expected_keys(self, tmp_path):
        from kortex.dream import get_db_stats
        from kortex.db import KortexDB
        db_path = str(tmp_path / "stats_test.db")
        db = KortexDB(db_path)
        stats = get_db_stats(db)
        assert isinstance(stats, dict)
        assert "episodes" in stats
        assert "facts" in stats
        assert "open_loops" in stats
        assert "entity_links" in stats
        db.close()

    def test_counts_are_zero_on_fresh_db(self, tmp_path):
        """Fresh DB should have zero counts."""
        from kortex.dream import get_db_stats
        from kortex.db import KortexDB
        db_path = str(tmp_path / "fresh.db")
        db = KortexDB(db_path)
        stats = get_db_stats(db)
        assert stats["episodes"] == 0
        assert stats["facts"] == 0
        assert stats["open_loops"] == 0
        assert stats["entity_links"] == 0
        db.close()


class TestDayDream:
    """Test DayDream mode — quick maintenance pass."""

    def test_daydream_runs_without_exception(self, tmp_path):
        """DayDream should complete without raising."""
        db_path = _make_db_with_data(tmp_path, "dd1.db")
        from kortex.dream import daydream
        result = daydream(db_path)
        assert result["mode"] == "daydream"
        assert "elapsed_seconds" in result
        assert result["elapsed_seconds"] > 0

    def test_daydream_returns_stats(self, tmp_path):
        """DayDream should return initial and final stats."""
        db_path = _make_db_with_data(tmp_path, "dd2.db")
        from kortex.dream import daydream
        result = daydream(db_path)
        assert "initial_stats" in result
        assert "final_stats" in result
        assert isinstance(result["initial_stats"], dict)
        assert isinstance(result["final_stats"], dict)

    def test_daydream_expired_loops_field_present(self, tmp_path):
        """DayDream should report expired loops."""
        db_path = _make_db_with_data(tmp_path, "dd3.db")
        from kortex.dream import daydream
        result = daydream(db_path, max_age_days=7)
        assert "expired_loops" in result
        assert isinstance(result["expired_loops"], int)

    def test_daydream_decayed_facts_field_present(self, tmp_path):
        """DayDream should report decayed facts."""
        db_path = _make_db_with_data(tmp_path, "dd4.db")
        from kortex.dream import daydream
        result = daydream(db_path, fact_decay_days=49)
        assert "decayed_facts" in result
        assert isinstance(result["decayed_facts"], int)

    def test_daydream_decayed_reflections_field_present(self, tmp_path):
        """DayDream should report decayed reflections."""
        db_path = _make_db_with_data(tmp_path, "dd5.db")
        from kortex.dream import daydream
        result = daydream(db_path, reflection_decay_days=98)
        assert "decayed_reflections" in result
        assert isinstance(result["decayed_reflections"], int)

    def test_daydream_considered_consolidation(self, tmp_path):
        """DayDream should include consolidation results."""
        db_path = _make_db_with_data(tmp_path, "dd6.db")
        from kortex.dream import daydream
        result = daydream(db_path)
        assert "consolidation" in result
        assert isinstance(result["consolidation"], dict)


class TestRemSleep:
    """Test REMSleep mode — deep optimization pass."""

    def test_rem_sleep_runs_without_exception(self, tmp_path):
        """REMSleep should complete without raising."""
        db_path = _make_db_with_data(tmp_path, "rs1.db")
        from kortex.dream import rem_sleep
        result = rem_sleep(db_path)
        assert result["mode"] == "rem_sleep"
        assert "elapsed_seconds" in result
        assert result["elapsed_seconds"] > 0

    def test_rem_sleep_returns_stats(self, tmp_path):
        """REMSleep should return initial and final stats."""
        db_path = _make_db_with_data(tmp_path, "rs2.db")
        from kortex.dream import rem_sleep
        result = rem_sleep(db_path)
        assert "initial_stats" in result
        assert "final_stats" in result

    def test_rem_sleep_considered_consolidation(self, tmp_path):
        """REMSleep should include consolidation results."""
        db_path = _make_db_with_data(tmp_path, "rs3.db")
        from kortex.dream import rem_sleep
        result = rem_sleep(db_path)
        assert "consolidation" in result
        assert isinstance(result["consolidation"], dict)

    def test_rem_sleep_includes_pruning_results(self, tmp_path):
        """REMSleep should report pruning results."""
        db_path = _make_db_with_data(tmp_path, "rs4.db")
        from kortex.dream import rem_sleep
        result = rem_sleep(db_path)
        assert "expired_loops" in result
        assert "decayed_facts" in result
        assert "decayed_reflections" in result


class TestDreamAlias:
    """Test legacy dream() alias."""

    def test_dream_is_alias_for_daydream(self, tmp_path):
        """dream() should be an alias for daydream()."""
        db_path = _make_empty_db(tmp_path, "alias.db")
        from kortex.dream import dream
        result = dream(db_path, max_age_days=7)
        assert result["mode"] == "daydream"


class TestMainCli:
    """Test CLI entry point."""

    def test_cli_help(self):
        """CLI --help should work."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "kortex.dream", "--help"],
            capture_output=True, text=True, timeout=5
        )
        assert result.returncode == 0


class TestProviderIntegration:
    """Test that KortexProvider._trigger_daydream works correctly."""

    def test_trigger_daydream_method_exists(self):
        """_trigger_daydream should exist on KortexProvider."""
        from kortex.provider import KortexProvider
        from kortex.config import KortexConfig
        provider = KortexProvider(KortexConfig())
        assert hasattr(provider, "_trigger_daydream")
        assert callable(getattr(provider, "_trigger_daydream"))

    def test_trigger_daydream_has_lock(self):
        """Provider should have daydream lock attributes."""
        from kortex.provider import KortexProvider
        from kortex.config import KortexConfig
        provider = KortexProvider(KortexConfig())
        assert hasattr(provider, "_daydream_active")
        assert hasattr(provider, "_daydream_lock")
        assert provider._daydream_active == False

    def test_trigger_daydream_noop_when_not_initialized(self):
        """_trigger_daydream should be a no-op when DB is not initialized."""
        from kortex.provider import KortexProvider
        from kortex.config import KortexConfig
        provider = KortexProvider(KortexConfig())
        provider._trigger_daydream()

    def test_trigger_daydream_spawns_thread(self, tmp_path):
        """_trigger_daydream should spawn a daemon thread."""
        from kortex.provider import KortexProvider
        from kortex.config import KortexConfig
        from kortex.db import KortexDB
        db_path = _make_empty_db(tmp_path, "prov.db")
        provider = KortexProvider(KortexConfig(db_path=db_path))
        provider._hermes_home = str(tmp_path)
        provider._user_id = "default"
        provider._db = KortexDB(db_path)
        provider._trigger_daydream()
        time.sleep(0.5)
        assert provider._daydream_active == False
        provider._db.close()


class TestSecurity:
    """Test that dream functions handle edge cases gracefully."""

    def test_daydream_handles_empty_db(self, tmp_path):
        """DayDream should handle an empty DB gracefully."""
        db_path = _make_empty_db(tmp_path, "sec1.db")
        from kortex.dream import daydream
        result = daydream(db_path)
        assert result["mode"] == "daydream"
        assert result["elapsed_seconds"] > 0

    def test_rem_sleep_handles_empty_db(self, tmp_path):
        """REMSleep should handle an empty DB gracefully."""
        db_path = _make_empty_db(tmp_path, "sec2.db")
        from kortex.dream import rem_sleep
        result = rem_sleep(db_path)
        assert result["mode"] == "rem_sleep"
        assert result["elapsed_seconds"] > 0

    def test_multiple_daydream_runs_are_idempotent(self, tmp_path):
        """Running DayDream twice should be safe."""
        db_path = _make_db_with_data(tmp_path, "sec3.db")
        from kortex.dream import daydream
        r1 = daydream(db_path)
        r2 = daydream(db_path)
        assert r1["mode"] == "daydream"
        assert r2["mode"] == "daydream"
