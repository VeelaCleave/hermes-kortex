import json
import threading
import time
from pathlib import Path

import pytest

from kortex.config import KortexConfig
from kortex.db import KortexDB
from kortex.models import Episode, IdentityDelta
from kortex.promote import Promoter
from kortex.provider import KortexProvider


def _episode(kortex_db, summary="episode summary", session_id="s1"):
    episode = Episode(
        session_id=session_id,
        user_text="user message",
        assistant_text="assistant message",
        summary=summary,
        timestamp=time.time(),
        salience=0.6,
    )
    kortex_db.insert_episode(episode)
    return episode


def _delta(
    kortex_db,
    text="She tends to overcorrect after making mistakes",
    confidence=0.65,
    source_episode_id=None,
    applied=False,
):
    delta = IdentityDelta(
        text=text,
        confidence=confidence,
        source_episode_id=source_episode_id,
        applied=applied,
    )
    kortex_db.insert_identity_delta(delta)
    return delta


def _provider(tmp_path, *, soul_path=None):
    config = KortexConfig(db_path=str(tmp_path / "kortex.db"), soul_path=soul_path)
    provider = KortexProvider(config=config)
    provider.initialize("test-session", hermes_home=str(tmp_path))
    return provider


class TestPromoterListPending:
    def test_list_pending_returns_only_unapplied_deltas(self, kortex_db, tmp_path):
        _delta(kortex_db, text="pending one", applied=False)
        _delta(kortex_db, text="already applied", applied=True)

        pending = Promoter(
            kortex_db, soul_path=str(tmp_path / "SOUL.md")
        ).list_pending()

        assert [delta.text for delta in pending] == ["pending one"]

    def test_list_pending_respects_limit(self, kortex_db, tmp_path):
        for index in range(5):
            _delta(kortex_db, text=f"delta {index}", confidence=0.5 + index / 100)

        pending = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).list_pending(
            limit=2
        )

        assert len(pending) == 2

    def test_list_pending_sorted_by_confidence_desc(self, kortex_db, tmp_path):
        _delta(kortex_db, text="low", confidence=0.3)
        _delta(kortex_db, text="high", confidence=0.9)
        _delta(kortex_db, text="mid", confidence=0.6)

        pending = Promoter(
            kortex_db, soul_path=str(tmp_path / "SOUL.md")
        ).list_pending()

        assert [delta.text for delta in pending] == ["high", "mid", "low"]

    def test_list_pending_returns_empty_when_none(self, kortex_db, tmp_path):
        pending = Promoter(
            kortex_db, soul_path=str(tmp_path / "SOUL.md")
        ).list_pending()
        assert pending == []

    def test_list_pending_ignores_applied_even_when_high_confidence(
        self, kortex_db, tmp_path
    ):
        _delta(kortex_db, text="applied high", confidence=0.99, applied=True)
        _delta(kortex_db, text="pending lower", confidence=0.5, applied=False)

        pending = Promoter(
            kortex_db, soul_path=str(tmp_path / "SOUL.md")
        ).list_pending()

        assert [delta.text for delta in pending] == ["pending lower"]

    def test_list_pending_limit_zero_returns_empty(self, kortex_db, tmp_path):
        _delta(kortex_db, text="delta")
        pending = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).list_pending(
            limit=0
        )
        assert pending == []


class TestPromoterPreview:
    def test_preview_delta_with_valid_id_returns_correct_info(
        self, kortex_db, tmp_path
    ):
        delta = _delta(kortex_db, text="preview me", confidence=0.77)
        result = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).preview_delta(
            delta.id
        )

        assert result["id"] == delta.id
        assert result["text"] == "preview me"
        assert result["confidence"] == pytest.approx(0.77)
        assert result["applied"] is False

    def test_preview_delta_with_source_episode_includes_episode_summary(
        self, kortex_db, tmp_path
    ):
        episode = _episode(kortex_db, summary="source episode summary")
        delta = _delta(kortex_db, source_episode_id=episode.id)

        result = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).preview_delta(
            delta.id
        )

        assert result["source_episode"]["summary"] == "source episode summary"
        assert result["source_episode"]["id"] == episode.id

    def test_preview_delta_with_invalid_id_returns_error(self, kortex_db, tmp_path):
        result = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).preview_delta(
            9999
        )
        assert "error" in result

    def test_preview_delta_without_source_episode_omits_episode_context(
        self, kortex_db, tmp_path
    ):
        delta = _delta(kortex_db, source_episode_id=None)
        result = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).preview_delta(
            delta.id
        )
        assert "source_episode" not in result

    def test_preview_delta_returns_created_at_string(self, kortex_db, tmp_path):
        delta = _delta(kortex_db)
        result = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).preview_delta(
            delta.id
        )
        assert isinstance(result["created_at"], str)
        assert "T" in result["created_at"]

    def test_preview_delta_keeps_applied_flag_for_applied_delta(
        self, kortex_db, tmp_path
    ):
        delta = _delta(kortex_db, applied=True)
        result = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).preview_delta(
            delta.id
        )
        assert result["applied"] is True

    def test_preview_delta_truncates_very_long_text_to_500_chars(
        self, kortex_db, tmp_path
    ):
        delta = _delta(kortex_db, text="x" * 700)
        result = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).preview_delta(
            delta.id
        )
        assert len(result["text"]) == 500


class TestPromoterApproveApply:
    def test_approve_and_apply_creates_soul_md_if_missing(self, kortex_db, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        delta = _delta(kortex_db)

        result = Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(
            delta.id
        )

        assert result["success"] is True
        assert soul_path.exists()

    def test_approve_and_apply_adds_learned_traits_section_if_missing(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text(
            "# SOUL\n\n## Core Identity\nStable self\n", encoding="utf-8"
        )
        delta = _delta(kortex_db, text="Learns from criticism")

        Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(delta.id)
        content = soul_path.read_text(encoding="utf-8")

        assert "## Learned Traits" in content
        assert "- Learns from criticism [confidence: 0.65]" in content

    def test_approve_and_apply_appends_under_existing_learned_traits_section(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text(
            "# SOUL\n\n## Learned Traits\n- Existing trait [confidence: 0.90]\n",
            encoding="utf-8",
        )
        delta = _delta(kortex_db, text="New trait")

        Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(delta.id)
        content = soul_path.read_text(encoding="utf-8")

        assert content.count("## Learned Traits") == 1
        assert "- Existing trait [confidence: 0.90]" in content
        assert "- New trait [confidence: 0.65]" in content

    def test_approve_and_apply_marks_delta_as_applied_in_db(self, kortex_db, tmp_path):
        delta = _delta(kortex_db)
        promoter = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md"))

        promoter.approve_and_apply(delta.id)

        assert kortex_db.get_identity_delta_by_id(delta.id).applied is True

    def test_approve_and_apply_with_invalid_id_returns_error(self, kortex_db, tmp_path):
        result = Promoter(
            kortex_db, soul_path=str(tmp_path / "SOUL.md")
        ).approve_and_apply(9999)
        assert "error" in result

    def test_approve_and_apply_with_already_applied_delta_returns_error(
        self, kortex_db, tmp_path
    ):
        delta = _delta(kortex_db, applied=True)
        result = Promoter(
            kortex_db, soul_path=str(tmp_path / "SOUL.md")
        ).approve_and_apply(delta.id)
        assert "already applied" in result["error"]

    def test_get_soul_content_reads_existing_file(self, kortex_db, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("existing soul", encoding="utf-8")
        content = Promoter(kortex_db, soul_path=str(soul_path)).get_soul_content()
        assert content == "existing soul"

    def test_get_soul_content_returns_empty_for_missing_file(self, kortex_db, tmp_path):
        content = Promoter(
            kortex_db, soul_path=str(tmp_path / "SOUL.md")
        ).get_soul_content()
        assert content == ""

    def test_resolve_soul_path_uses_config_override(self, kortex_db, tmp_path):
        custom_path = tmp_path / "custom" / "SOUL.md"
        promoter = Promoter(kortex_db, soul_path=str(custom_path))
        assert promoter._resolve_soul_path() == custom_path

    def test_resolve_soul_path_uses_hermes_home(self, kortex_db, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        promoter = Promoter(kortex_db)
        assert promoter._resolve_soul_path() == (tmp_path / "hermes-home" / "SOUL.md")

    def test_resolve_soul_path_falls_back_to_home_hermes(
        self, kortex_db, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        promoter = Promoter(kortex_db)
        assert promoter._resolve_soul_path() == (tmp_path / ".hermes" / "SOUL.md")

    def test_approve_and_apply_handles_soul_without_trailing_newline(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# SOUL", encoding="utf-8")
        delta = _delta(kortex_db, text="No newline case")

        Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(delta.id)
        content = soul_path.read_text(encoding="utf-8")

        assert content.endswith("\n")
        assert "## Learned Traits" in content

    def test_approve_and_apply_with_existing_traits_preserves_other_traits(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text(
            "# SOUL\n\n## Learned Traits\n- Trait A [confidence: 0.80]\n- Trait B [confidence: 0.70]\n",
            encoding="utf-8",
        )
        delta = _delta(kortex_db, text="Trait C")

        Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(delta.id)
        content = soul_path.read_text(encoding="utf-8")

        assert "Trait A" in content
        assert "Trait B" in content
        assert "Trait C" in content

    def test_multiple_approve_operations_in_sequence(self, kortex_db, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        promoter = Promoter(kortex_db, soul_path=str(soul_path))
        first = _delta(kortex_db, text="First")
        second = _delta(kortex_db, text="Second")

        promoter.approve_and_apply(first.id)
        promoter.approve_and_apply(second.id)
        content = soul_path.read_text(encoding="utf-8")

        assert content.count("## Learned Traits") == 1
        assert "- First [confidence: 0.65]" in content
        assert "- Second [confidence: 0.65]" in content

    def test_approve_and_apply_supports_unicode_content(self, kortex_db, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        delta = _delta(kortex_db, text="She says café, 你好, and λ naturally")

        Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(delta.id)
        content = soul_path.read_text(encoding="utf-8")

        assert "café" in content
        assert "你好" in content
        assert "λ" in content

    def test_approve_and_apply_truncates_very_long_delta_text_to_500_chars(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        delta = _delta(kortex_db, text="a" * 700)

        result = Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(
            delta.id
        )
        content = soul_path.read_text(encoding="utf-8")

        assert len(result["applied_text"]) == len("- ") + 500 + len(
            " [confidence: 0.65]"
        )
        assert "a" * 500 in content
        assert "a" * 501 not in content

    def test_approve_and_apply_inserts_before_next_section(self, kortex_db, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text(
            "# SOUL\n\n## Learned Traits\n- Existing [confidence: 0.90]\n## Values\nStay steady\n",
            encoding="utf-8",
        )
        delta = _delta(kortex_db, text="Inserted before next header")

        Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(delta.id)
        content = soul_path.read_text(encoding="utf-8")

        assert content.index("Inserted before next header") < content.index("## Values")

    def test_approve_and_apply_on_blank_soul_file_writes_header_and_trait(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("", encoding="utf-8")
        delta = _delta(kortex_db, text="Blank file trait")

        Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(delta.id)
        content = soul_path.read_text(encoding="utf-8")

        assert content == "## Learned Traits\n- Blank file trait [confidence: 0.65]\n"


class TestPromoterRejectAndBulk:
    def test_reject_delta_removes_delta_from_db(self, kortex_db, tmp_path):
        delta = _delta(kortex_db)
        promoter = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md"))

        result = promoter.reject_delta(delta.id)

        assert result == {"success": True, "rejected_id": delta.id}
        assert kortex_db.get_identity_delta_by_id(delta.id) is None

    def test_reject_delta_with_invalid_id_returns_error(self, kortex_db, tmp_path):
        result = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).reject_delta(
            9999
        )
        assert "error" in result

    def test_approve_multiple_applies_multiple_deltas(self, kortex_db, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        promoter = Promoter(kortex_db, soul_path=str(soul_path))
        first = _delta(kortex_db, text="First bulk")
        second = _delta(kortex_db, text="Second bulk")

        result = promoter.approve_multiple([first.id, second.id])
        content = soul_path.read_text(encoding="utf-8")

        assert result["applied_count"] == 2
        assert "First bulk" in content
        assert "Second bulk" in content

    def test_approve_multiple_skips_already_applied_deltas(self, kortex_db, tmp_path):
        promoter = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md"))
        applied = _delta(kortex_db, text="done", applied=True)
        pending = _delta(kortex_db, text="todo")

        result = promoter.approve_multiple([applied.id, pending.id])

        assert result["applied_count"] == 1
        assert any(item["delta_id"] == applied.id for item in result["failed"])

    def test_approve_multiple_with_empty_list_returns_appropriate_result(
        self, kortex_db, tmp_path
    ):
        result = Promoter(
            kortex_db, soul_path=str(tmp_path / "SOUL.md")
        ).approve_multiple([])
        assert result["requested"] == 0
        assert result["applied_count"] == 0

    def test_approve_multiple_handles_invalid_ids(self, kortex_db, tmp_path):
        promoter = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md"))
        valid = _delta(kortex_db, text="valid")

        result = promoter.approve_multiple([valid.id, 9999])

        assert result["applied_count"] == 1
        assert any(item["delta_id"] == 9999 for item in result["failed"])

    def test_approve_multiple_does_not_duplicate_header(self, kortex_db, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        promoter = Promoter(kortex_db, soul_path=str(soul_path))
        deltas = [_delta(kortex_db, text=f"trait {index}") for index in range(3)]

        promoter.approve_multiple([delta.id for delta in deltas])
        content = soul_path.read_text(encoding="utf-8")

        assert content.count("## Learned Traits") == 1

    def test_approve_multiple_preserves_existing_non_identity_sections(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# SOUL\n\n## Values\nHold course\n", encoding="utf-8")
        promoter = Promoter(kortex_db, soul_path=str(soul_path))
        delta = _delta(kortex_db, text="Bulk preserve")

        promoter.approve_multiple([delta.id])
        content = soul_path.read_text(encoding="utf-8")

        assert "## Values" in content
        assert "Hold course" in content

    def test_approve_multiple_returns_success_false_on_partial_failure(
        self, kortex_db, tmp_path
    ):
        promoter = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md"))
        valid = _delta(kortex_db)

        result = promoter.approve_multiple([valid.id, 9999])

        assert result["success"] is False

    def test_approve_multiple_marks_each_applied_in_db(self, kortex_db, tmp_path):
        promoter = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md"))
        first = _delta(kortex_db, text="db1")
        second = _delta(kortex_db, text="db2")

        promoter.approve_multiple([first.id, second.id])

        assert kortex_db.get_identity_delta_by_id(first.id).applied is True
        assert kortex_db.get_identity_delta_by_id(second.id).applied is True


class TestDBIdentityDeltaStage6Methods:
    def test_mark_identity_delta_applied_sets_applied_true(self, kortex_db):
        delta = _delta(kortex_db)
        assert kortex_db.mark_identity_delta_applied(delta.id) is True
        assert kortex_db.get_identity_delta_by_id(delta.id).applied is True

    def test_mark_identity_delta_applied_returns_false_for_nonexistent_id(
        self, kortex_db
    ):
        assert kortex_db.mark_identity_delta_applied(9999) is False

    def test_delete_identity_delta_removes_the_row(self, kortex_db):
        delta = _delta(kortex_db)
        assert kortex_db.delete_identity_delta(delta.id) is True
        assert kortex_db.get_identity_delta_by_id(delta.id) is None

    def test_delete_identity_delta_returns_false_for_nonexistent_id(self, kortex_db):
        assert kortex_db.delete_identity_delta(9999) is False

    def test_get_identity_delta_by_id_returns_correct_delta(self, kortex_db):
        delta = _delta(kortex_db, text="lookup me", confidence=0.88)
        found = kortex_db.get_identity_delta_by_id(delta.id)
        assert found.text == "lookup me"
        assert found.confidence == pytest.approx(0.88)

    def test_get_identity_delta_by_id_returns_none_for_nonexistent_id(self, kortex_db):
        assert kortex_db.get_identity_delta_by_id(9999) is None

    def test_reject_identity_delta_alias_deletes_row(self, kortex_db):
        delta = _delta(kortex_db)
        assert kortex_db.reject_identity_delta(delta.id) is True
        assert kortex_db.get_identity_delta_by_id(delta.id) is None

    def test_reject_identity_delta_alias_returns_false_when_missing(self, kortex_db):
        assert kortex_db.reject_identity_delta(9999) is False


class TestProviderIdentityIntegration:
    def test_get_tool_schemas_returns_all_tools(self, tmp_path):
        provider = _provider(tmp_path)
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 3
        names = {s["name"] for s in schemas}
        assert "kortex_query" in names
        assert "kortex_recall" in names
        assert "kortex_expand" in names
        provider.shutdown()

    def test_handle_tool_call_dispatches_kortex_query_correctly(self, tmp_path):
        provider = _provider(tmp_path, soul_path=str(tmp_path / "SOUL.md"))
        result = json.loads(
            provider.handle_tool_call("kortex_query", {"action": "list_pending"})
        )
        assert "pending" in result
        provider.shutdown()

    def test_handle_tool_call_list_pending_returns_pending_deltas(self, tmp_path):
        provider = _provider(tmp_path)
        _delta(provider._db, text="pending from provider")
        result = json.loads(
            provider.handle_tool_call("kortex_query", {"action": "list_pending"})
        )
        assert result["pending"][0]["text"] == "pending from provider"
        provider.shutdown()

    def test_handle_tool_call_preview_returns_delta_info(self, tmp_path):
        provider = _provider(tmp_path)
        episode = _episode(provider._db, summary="provider source")
        delta = _delta(
            provider._db, text="provider preview", source_episode_id=episode.id
        )
        result = json.loads(
            provider.handle_tool_call(
                "kortex_query", {"action": "preview", "delta_id": delta.id}
            )
        )
        assert result["text"] == "provider preview"
        assert result["source_episode"]["summary"] == "provider source"
        provider.shutdown()

    def test_handle_tool_call_approve_applies_to_soul(self, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        provider = _provider(tmp_path, soul_path=str(soul_path))
        delta = _delta(provider._db, text="provider approve")

        result = json.loads(
            provider.handle_tool_call(
                "kortex_query", {"action": "approve", "delta_id": delta.id}
            )
        )

        assert result["success"] is True
        assert "provider approve" in soul_path.read_text(encoding="utf-8")
        provider.shutdown()

    def test_handle_tool_call_reject_removes_delta(self, tmp_path):
        provider = _provider(tmp_path)
        delta = _delta(provider._db, text="reject me")

        result = json.loads(
            provider.handle_tool_call(
                "kortex_query", {"action": "reject", "delta_id": delta.id}
            )
        )

        assert result["success"] is True
        assert provider._db.get_identity_delta_by_id(delta.id) is None
        provider.shutdown()

    def test_handle_tool_call_approve_all_with_min_confidence_filter(self, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        provider = _provider(tmp_path, soul_path=str(soul_path))
        low = _delta(provider._db, text="low", confidence=0.4)
        high = _delta(provider._db, text="high", confidence=0.8)

        result = json.loads(
            provider.handle_tool_call(
                "kortex_query",
                {"action": "approve_all", "min_confidence": 0.6},
            )
        )

        assert result["applied_count"] == 1
        assert provider._db.get_identity_delta_by_id(high.id).applied is True
        assert provider._db.get_identity_delta_by_id(low.id).applied is False
        provider.shutdown()

    def test_handle_tool_call_show_soul_returns_soul_content(self, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# SOUL\n", encoding="utf-8")
        provider = _provider(tmp_path, soul_path=str(soul_path))
        result = json.loads(
            provider.handle_tool_call("kortex_query", {"action": "show_soul"})
        )
        assert result["content"] == "# SOUL\n"
        provider.shutdown()

    def test_handle_tool_call_unknown_action_returns_error(self, tmp_path):
        provider = _provider(tmp_path)
        result = json.loads(
            provider.handle_tool_call("kortex_query", {"action": "wat"})
        )
        assert "error" in result
        provider.shutdown()

    @pytest.mark.parametrize("action", ["preview", "approve", "reject"])
    def test_handle_tool_call_requires_delta_id_for_specific_actions(
        self, tmp_path, action
    ):
        provider = _provider(tmp_path)
        result = json.loads(
            provider.handle_tool_call("kortex_query", {"action": action})
        )
        assert result["error"] == "delta_id required"
        provider.shutdown()

    def test_handle_tool_call_show_soul_returns_path(self, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        provider = _provider(tmp_path, soul_path=str(soul_path))
        result = json.loads(
            provider.handle_tool_call("kortex_query", {"action": "show_soul"})
        )
        assert result["soul_path"] == str(soul_path)
        provider.shutdown()

    def test_handle_tool_call_approve_all_with_no_matches_returns_zero(self, tmp_path):
        provider = _provider(tmp_path, soul_path=str(tmp_path / "SOUL.md"))
        _delta(provider._db, text="too low", confidence=0.2)
        result = json.loads(
            provider.handle_tool_call(
                "kortex_query", {"action": "approve_all", "min_confidence": 0.9}
            )
        )
        assert result["applied_count"] == 0
        provider.shutdown()


class TestStage6EdgeCases:
    def test_concurrent_approve_operations_same_delta_are_thread_safe(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        promoter = Promoter(kortex_db, soul_path=str(soul_path))
        delta = _delta(kortex_db, text="thread safe")
        results = []

        def _run():
            results.append(promoter.approve_and_apply(delta.id))

        threads = [threading.Thread(target=_run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sum(1 for result in results if result.get("success")) == 1
        assert sum(1 for result in results if result.get("error")) == 1

    def test_concurrent_approve_operations_different_deltas_both_succeed(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        promoter = Promoter(kortex_db, soul_path=str(soul_path))
        first = _delta(kortex_db, text="first thread")
        second = _delta(kortex_db, text="second thread")
        results = []

        threads = [
            threading.Thread(
                target=lambda: results.append(promoter.approve_and_apply(first.id))
            ),
            threading.Thread(
                target=lambda: results.append(promoter.approve_and_apply(second.id))
            ),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sum(1 for result in results if result.get("success")) == 2

    def test_show_soul_returns_empty_content_when_missing(self, tmp_path):
        provider = _provider(tmp_path, soul_path=str(tmp_path / "SOUL.md"))
        result = json.loads(
            provider.handle_tool_call("kortex_query", {"action": "show_soul"})
        )
        assert result["content"] == ""
        provider.shutdown()

    def test_approve_all_applies_in_confidence_order(self, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        provider = _provider(tmp_path, soul_path=str(soul_path))
        _delta(provider._db, text="low first", confidence=0.61)
        _delta(provider._db, text="high first", confidence=0.95)

        provider.handle_tool_call(
            "kortex_query", {"action": "approve_all", "min_confidence": 0.6}
        )
        content = soul_path.read_text(encoding="utf-8")

        assert content.index("high first") < content.index("low first")
        provider.shutdown()

    def test_list_pending_through_provider_respects_limit(self, tmp_path):
        provider = _provider(tmp_path)
        for index in range(4):
            _delta(provider._db, text=f"delta {index}", confidence=0.9 - index / 10)
        result = json.loads(
            provider.handle_tool_call(
                "kortex_query", {"action": "list_pending", "limit": 2}
            )
        )
        assert len(result["pending"]) == 2
        provider.shutdown()

    def test_approve_preserves_existing_text_before_learned_traits(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# SOUL\nIdentity core\n", encoding="utf-8")
        delta = _delta(kortex_db, text="preserve prefix")

        Promoter(kortex_db, soul_path=str(soul_path)).approve_and_apply(delta.id)
        content = soul_path.read_text(encoding="utf-8")

        assert content.startswith("# SOUL\nIdentity core")

    def test_approve_multiple_can_be_called_after_single_approve(
        self, kortex_db, tmp_path
    ):
        soul_path = tmp_path / "SOUL.md"
        promoter = Promoter(kortex_db, soul_path=str(soul_path))
        first = _delta(kortex_db, text="single first")
        second = _delta(kortex_db, text="bulk second")

        promoter.approve_and_apply(first.id)
        result = promoter.approve_multiple([first.id, second.id])

        assert result["applied_count"] == 1
        assert any(item["delta_id"] == first.id for item in result["failed"])

    def test_preview_delta_handles_missing_source_episode_gracefully(
        self, kortex_db, tmp_path
    ):
        episode = _episode(kortex_db, summary="temporary source")
        delta = _delta(kortex_db, source_episode_id=episode.id)
        conn = kortex_db._get_conn()
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "UPDATE identity_deltas SET source_episode_id=? WHERE id=?",
            (9999, delta.id),
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys=ON")
        result = Promoter(kortex_db, soul_path=str(tmp_path / "SOUL.md")).preview_delta(
            delta.id
        )
        assert "source_episode" not in result
