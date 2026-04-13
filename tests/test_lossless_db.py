from kortex.db import KortexDB


class TestLosslessContextStorage:
    def test_session_alias_and_archive_roundtrip(self, tmp_db_path):
        db = KortexDB(tmp_db_path)
        db.ensure_context_conversation("conv_1")
        db.map_session_alias("sess_1", "conv_1")
        assert db.get_context_conversation_id("sess_1") == "conv_1"

        start_seq, end_seq = db.archive_context_messages(
            "conv_1",
            [
                {"role": "user", "content": "hello gateway failure"},
                {"role": "assistant", "content": "decision keep retry wrapper"},
            ],
        )
        assert start_seq == 1
        assert end_seq == 2

        span_id = db.create_context_span(
            "conv_1", start_seq=1, end_seq=2, kind="compressed"
        )
        db.insert_context_ref(
            "conv_1",
            ref_id="ref_test",
            ref_type="decision",
            label="keep retry wrapper",
            payload={"source_span_id": span_id},
            source_span_id=span_id,
            salience=0.8,
        )
        db.insert_context_checkpoint(
            "conv_1",
            checkpoint_id="ckpt_1",
            replaced_start_seq=1,
            replaced_end_seq=2,
            export_text="checkpoint text",
            source_span_ids=[span_id],
            hot_ref_ids=["ref_test"],
        )

        checkpoint = db.get_active_context_checkpoint("conv_1")
        assert checkpoint["checkpoint_id"] == "ckpt_1"
        assert db.search_context_refs("conv_1", "retry")
        assert db.search_context_messages("conv_1", "gateway")
        assert len(db.get_context_messages_by_seq_range("conv_1", 1, 2)) == 2
        db.close()
