import json

from kortex.context_engine import KortexContextEngine


class TestKortexContextEngine:
    def test_name(self):
        engine = KortexContextEngine()
        assert engine.name == "kortex"

    def test_passthrough_compress_without_hermes_delegate(self):
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
        assert status["delegate"] in {"compressor", "passthrough"}
