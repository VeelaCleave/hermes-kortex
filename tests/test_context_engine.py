import json

from kortex.context_engine import CHECKPOINT_PREFIX, KortexContextEngine
from kortex.db import KortexDB


class TestKortexContextEngine:
    def test_name(self):
        engine = KortexContextEngine()
        assert engine.name == "kortex"

    def test_compress_small_message_list_is_noop(self):
        engine = KortexContextEngine()
        messages = [{"role": "user", "content": "hello"}]
        assert engine.compress(messages) == messages

    def test_update_from_response_tracks_tokens(self):
        engine = KortexContextEngine()
        engine.update_from_response(
            {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
        )
        assert engine.last_prompt_tokens == 120
        assert engine.last_completion_tokens == 30
        assert engine.last_total_tokens == 150

    def test_default_tool_handler_errors(self):
        engine = KortexContextEngine()
        result = json.loads(engine.handle_tool_call("unknown", {}))
        assert "error" in result

    def test_status_reports_engine_identity(self):
        engine = KortexContextEngine()
        status = engine.get_status()
        assert status["engine"] == "kortex"

    def test_compress_archives_middle_and_emits_checkpoint(self, tmp_path):
        db_path = str(tmp_path / "kortex.db")
        engine = KortexContextEngine(db_path=db_path)
        engine.update_model("test", 1000)
        engine.on_session_start("sess-1", hermes_home=str(tmp_path), model="test")

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "decided keep gateway/run.py fix rollback"},
            {"role": "assistant", "content": "implement follow up tomorrow"},
            {"role": "user", "content": "tail1"},
            {"role": "assistant", "content": "tail2"},
        ]
        engine.protect_first_n = 1
        engine.protect_last_n = 2

        compressed = engine.compress(messages)
        assert compressed[1]["content"].startswith(CHECKPOINT_PREFIX)

        db = KortexDB(db_path)
        checkpoint = db.get_active_context_checkpoint(engine._conversation_id)
        assert checkpoint is not None
        hits = db.search_context_messages(engine._conversation_id, "gateway")
        assert len(hits) >= 1
        db.close()

    def test_recall_and_expand_tools_return_archived_content(self, tmp_path):
        db_path = str(tmp_path / "kortex.db")
        engine = KortexContextEngine(db_path=db_path)
        engine.update_model("test", 1000)
        engine.on_session_start("sess-1", hermes_home=str(tmp_path), model="test")
        engine.protect_first_n = 1
        engine.protect_last_n = 1
        engine.compress(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "investigate provider retry failure"},
                {"role": "assistant", "content": "decision use retry wrapper"},
                {"role": "user", "content": "tail"},
            ]
        )

        recall = json.loads(
            engine.handle_tool_call("kortex_recall", {"query": "retry"})
        )
        assert recall["refs"] or recall["message_hits"]

        if recall["refs"]:
            expanded = engine.expand_ref(recall["refs"][0]["ref_id"], limit=4)
            assert expanded["messages"]
