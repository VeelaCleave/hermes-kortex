from kortex.extract_llm import extract_structured_memory


class FakeAuxClient:
    def extract_structured(self, prompt):
        return {
            "summary": "LLM summary",
            "topics": ["infra", "deploy"],
            "entities": ["Kubernetes", "PostgreSQL"],
            "facts": [{"predicate": "uses", "object_text": "Kubernetes"}],
            "open_loops": [{"kind": "question", "text": "Need rollback plan"}],
            "reflections": ["User values clear rollout plans"],
        }


class TestExtractLLM:
    def test_returns_none_without_client(self):
        assert extract_structured_memory("user", "assistant") is None

    def test_normalizes_structured_output(self):
        result = extract_structured_memory(
            "user", "assistant", auxiliary_client=FakeAuxClient()
        )
        assert result["summary"] == "LLM summary"
        assert result["topics"] == ["infra", "deploy"]
        assert result["entities"] == ["Kubernetes", "PostgreSQL"]
        assert result["facts"] == [("uses", "Kubernetes")]
        assert result["open_loops"][0]["text"] == "Need rollback plan"
