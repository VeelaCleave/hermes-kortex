import json
import time

from kortex.export import export_to_json, import_from_json
from kortex.models import Episode, Fact, OpenLoop, Reflection
from kortex.provider import KortexProvider


def test_export_json_includes_metadata_and_iso_timestamps(kortex_db):
    kortex_db.insert_episode(
        Episode(session_id="s1", summary="export me", timestamp=time.time())
    )
    kortex_db.insert_fact(Fact(object_text="uses neovim", predicate="uses"))

    exported = json.loads(export_to_json(kortex_db))
    assert exported["metadata"]["kortex_schema_version"] == 5
    assert "T" in exported["metadata"]["exported_at"]
    assert "T" in exported["episodes"][0]["timestamp"]


def test_selective_export_by_user_and_date_range(kortex_db):
    old_ts = time.time() - (10 * 86400)
    new_ts = time.time()
    kortex_db.insert_episode(
        Episode(user_id="alice", session_id="s1", summary="old", timestamp=old_ts)
    )
    kortex_db.insert_episode(
        Episode(user_id="alice", session_id="s1", summary="new", timestamp=new_ts)
    )
    kortex_db.insert_episode(
        Episode(user_id="bob", session_id="s2", summary="bob", timestamp=new_ts)
    )

    exported = json.loads(
        export_to_json(
            kortex_db,
            user_id="alice",
            start=new_ts - 60,
            end=new_ts + 60,
            memory_types=["episodes"],
        )
    )
    assert [ep["summary"] for ep in exported["episodes"]] == ["new"]


def test_import_rejects_incompatible_schema(kortex_db):
    payload = json.dumps({"metadata": {"kortex_schema_version": 999}, "episodes": []})
    result = import_from_json(kortex_db, payload)
    assert result["ok"] is False
    assert "Schema version mismatch" in result["error"]


def test_import_round_trip_restores_records(tmp_path):
    from kortex.db import KortexDB

    source = KortexDB(str(tmp_path / "source.db"))
    source.insert_episode(Episode(session_id="s1", summary="restore me"))
    source.insert_open_loop(OpenLoop(text="ship export tooling"))
    source.insert_reflection(Reflection(text="capture backups", kind="pattern"))

    payload = export_to_json(source)

    target = KortexDB(str(tmp_path / "target.db"))
    result = import_from_json(target, payload)

    assert result["ok"] is True
    assert target.count_episodes() == 1
    assert len(target.get_open_loops(limit=10)) == 1
    assert len(target.get_reflections(limit=10)) == 1

    source.close()
    target.close()


def test_provider_export_tool_integration(tmp_path):
    p = KortexProvider()
    p.initialize("test-session", hermes_home=str(tmp_path))
    p._db.insert_episode(Episode(session_id="s1", summary="provider export"))

    exported = json.loads(p.handle_tool_call("kortex_export", {"action": "export"}))
    assert exported["episodes"][0]["summary"] == "provider export"

    imported = json.loads(
        p.handle_tool_call(
            "kortex_export",
            {
                "action": "import",
                "payload": json.dumps({"metadata": {"kortex_schema_version": 999}}),
            },
        )
    )
    assert imported["ok"] is False
    p.shutdown()
