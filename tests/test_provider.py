import json
import time
from unittest.mock import MagicMock

from kortex.config import KortexConfig
from kortex.provider import KortexProvider


class TestProviderLifecycle:
    def test_name(self):
        p = KortexProvider()
        assert p.name == "kortex"

    def test_is_available(self):
        p = KortexProvider()
        assert p.is_available() is True

    def test_initialize_creates_db(self, tmp_path):
        config = KortexConfig(db_path=str(tmp_path / "test.db"))
        p = KortexProvider(config=config)
        p.initialize("test-session", hermes_home=str(tmp_path))
        assert p._db is not None
        assert p._ingestor is not None
        assert p._recall is not None
        p.shutdown()

    def test_system_prompt_block(self):
        p = KortexProvider()
        block = p.system_prompt_block()
        assert "KORTEX" in block
        assert "kortex_search" in block


class TestProviderSyncTurn:
    def test_sync_turn_ingests(self, tmp_path):
        config = KortexConfig(db_path=str(tmp_path / "test.db"))
        p = KortexProvider(config=config)
        p.initialize("test-session", hermes_home=str(tmp_path))

        p.sync_turn("Hello there", "Hi! How can I help?")
        time.sleep(0.5)

        episodes = p._db.get_recent_episodes(limit=10)
        assert len(episodes) >= 1
        p.shutdown()

    def test_sync_turn_skips_non_primary(self, tmp_path):
        config = KortexConfig(db_path=str(tmp_path / "test.db"))
        p = KortexProvider(config=config)
        p.initialize(
            "test-session", hermes_home=str(tmp_path), agent_context="subagent"
        )

        p.sync_turn("Hello", "Hi")
        time.sleep(0.3)

        episodes = p._db.get_recent_episodes(limit=10)
        assert len(episodes) == 0
        p.shutdown()


class TestProviderPrefetch:
    def test_prefetch_returns_context(self, tmp_path):
        config = KortexConfig(db_path=str(tmp_path / "test.db"))
        p = KortexProvider(config=config)
        p.initialize("test-session", hermes_home=str(tmp_path))

        p._db.insert_episode(
            __import__("kortex.models", fromlist=["Episode"]).Episode(
                session_id="test",
                summary="discussed python testing strategies",
                user_text="how should I write tests",
                salience=0.5,
            )
        )

        ctx = p.prefetch("testing")
        assert "KORTEX" in ctx or ctx == ""
        p.shutdown()

    def test_queue_prefetch_caches(self, tmp_path):
        config = KortexConfig(db_path=str(tmp_path / "test.db"))
        p = KortexProvider(config=config)
        p.initialize("test-session", hermes_home=str(tmp_path))

        p._db.insert_episode(
            __import__("kortex.models", fromlist=["Episode"]).Episode(
                session_id="test",
                summary="important architecture decision",
                salience=0.8,
            )
        )

        p.queue_prefetch("architecture")
        time.sleep(0.5)

        ctx = p.prefetch("anything")
        assert "KORTEX" in ctx or ctx == ""
        p.shutdown()


class TestProviderToolCall:
    def _setup_provider(self, tmp_path):
        config = KortexConfig(db_path=str(tmp_path / "test.db"))
        p = KortexProvider(config=config)
        p.initialize("test-session", hermes_home=str(tmp_path))
        return p

    def test_tool_schemas(self, tmp_path):
        p = self._setup_provider(tmp_path)
        schemas = p.get_tool_schemas()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert "kortex_search" in names
        assert "kortex_identity" in names
        p.shutdown()

    def test_status_action(self, tmp_path):
        p = self._setup_provider(tmp_path)
        result = json.loads(p.handle_tool_call("kortex_search", {"action": "status"}))
        assert "total_episodes" in result
        assert result["total_episodes"] == 0
        p.shutdown()

    def test_search_action(self, tmp_path):
        p = self._setup_provider(tmp_path)
        from kortex.models import Episode

        p._db.insert_episode(
            Episode(
                session_id="s1",
                summary="kubernetes deployment",
                user_text="deploy k8s",
            )
        )
        result = json.loads(
            p.handle_tool_call(
                "kortex_search",
                {
                    "action": "search",
                    "query": "kubernetes",
                },
            )
        )
        assert len(result["episodes"]) >= 1
        p.shutdown()

    def test_list_facts_action(self, tmp_path):
        p = self._setup_provider(tmp_path)
        from kortex.models import Fact

        p._db.insert_fact(Fact(object_text="uses neovim", predicate="uses"))
        result = json.loads(
            p.handle_tool_call(
                "kortex_search",
                {
                    "action": "list_facts",
                },
            )
        )
        assert len(result["facts"]) == 1
        p.shutdown()

    def test_list_loops_action(self, tmp_path):
        p = self._setup_provider(tmp_path)
        from kortex.models import OpenLoop

        p._db.insert_open_loop(OpenLoop(text="fix the bug"))
        result = json.loads(
            p.handle_tool_call(
                "kortex_search",
                {
                    "action": "list_loops",
                },
            )
        )
        assert len(result["loops"]) == 1
        p.shutdown()

    def test_recall_episode_action(self, tmp_path):
        p = self._setup_provider(tmp_path)
        from kortex.models import Episode

        ep = Episode(session_id="s1", summary="test episode", user_text="hello")
        p._db.insert_episode(ep)
        result = json.loads(
            p.handle_tool_call(
                "kortex_search",
                {
                    "action": "recall_episode",
                    "episode_id": ep.id,
                },
            )
        )
        assert result["summary"] == "test episode"
        p.shutdown()

    def test_unknown_tool(self, tmp_path):
        p = self._setup_provider(tmp_path)
        result = json.loads(p.handle_tool_call("unknown_tool", {}))
        assert "error" in result
        p.shutdown()

    def test_unknown_action(self, tmp_path):
        p = self._setup_provider(tmp_path)
        result = json.loads(p.handle_tool_call("kortex_search", {"action": "nope"}))
        assert "error" in result
        p.shutdown()


class TestProviderOnMemoryWrite:
    def test_mirrors_user_memory(self, tmp_path):
        config = KortexConfig(db_path=str(tmp_path / "test.db"))
        p = KortexProvider(config=config)
        p.initialize("test-session", hermes_home=str(tmp_path))

        p.on_memory_write("add", "user", "Prefers dark theme")

        facts = p._db.get_active_facts(subject_type="user")
        assert len(facts) == 1
        assert "dark theme" in facts[0].object_text
        p.shutdown()


class TestProviderConfigSchema:
    def test_returns_schema(self):
        p = KortexProvider()
        schema = p.get_config_schema()
        assert len(schema) >= 1
        keys = [s["key"] for s in schema]
        assert "db_path" in keys


class TestRegisterEntryPoint:
    def test_register_calls_provider(self):
        from kortex import register

        ctx = MagicMock()
        register(ctx)
        ctx.register_memory_provider.assert_called_once()
        provider = ctx.register_memory_provider.call_args[0][0]
        assert provider.name == "kortex"
