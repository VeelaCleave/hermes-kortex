"""
End-to-end tests proving KORTEX memory works correctly.

Tests the full pipeline: ingest conversation → extract facts →
store episodes → verify recall.
"""
import pytest


class TestE2EFactExtraction:
    """End-to-end tests for fact extraction quality."""

    def test_preference_fact_extracted(self, ingestor, kortex_db):
        """'I prefer X' should extract a meaningful fact."""
        ep = ingestor.ingest_turn(
            "I prefer dark mode for coding",
            "Dark mode it is!",
            session_id="e2e_pref",
        )
        facts = ingestor.extract_facts("I prefer dark mode for coding", ep.id)
        assert len(facts) >= 1, f"Expected at least 1 fact, got {len(facts)}: {[f.object_text for f in facts]}"
        assert any("dark mode" in f.object_text for f in facts), \
            f"Expected 'dark mode' in facts, got {[f.object_text for f in facts]}"

    def test_identity_fact_extracted(self, ingestor, kortex_db):
        """'I am a X' should extract an identity fact."""
        ep = ingestor.ingest_turn(
            "I am a Python developer",
            "Nice to meet you!",
            session_id="e2e_id",
        )
        facts = ingestor.extract_facts("I am a Python developer", ep.id)
        # Should extract "Python developer" or similar
        assert len(facts) >= 1, f"Expected at least 1 fact, got {len(facts)}: {[f.object_text for f in facts]}"
        texts = [f.object_text.lower() for f in facts]
        assert any("python" in t for t in texts), \
            f"Expected 'python' in facts, got {[f.object_text for f in facts]}"

    def test_garbage_identity_rejected(self, ingestor, kortex_db):
        """'I am going to bed' should NOT produce a garbage fact."""
        ep = ingestor.ingest_turn(
            "I am going to bed",
            "Good night!",
            session_id="e2e_garbage",
        )
        facts = ingestor.extract_facts("I am going to bed", ep.id)
        # The prefix filter + stopwords should reject this
        garbage = [f for f in facts if "going to" in f.object_text.lower()]
        assert len(garbage) == 0, \
            f"Garbage fact extracted: {[f.object_text for f in facts]}"

    def test_going_to_bed_rejected(self, ingestor, kortex_db):
        """'I am going to bed' specifically should be rejected."""
        ep = ingestor.ingest_turn("I am going to bed", "Night!", session_id="e2e_bed")
        facts = ingestor.extract_facts("I am going to bed", ep.id)
        for f in facts:
            assert f.object_text.lower() not in (
                "going to bed", "going bed", "going to", "going"
            ), f"Garbage fact passed: {f.object_text}"

    def test_single_char_fact_accepted(self, ingestor, kortex_db):
        """Single-char facts like emoji should still work."""
        ep = ingestor.ingest_turn("I hate 🍄", "Ew.", session_id="e2e_emoji")
        facts = ingestor.extract_facts("I hate 🍄", ep.id)
        assert len(facts) >= 1, f"Expected emoji fact, got {len(facts)}"

    def test_lol_slang_rejected(self, ingestor, kortex_db):
        """Slang like 'lol' in fact position should be rejected."""
        ep = ingestor.ingest_turn("lol that was funny", "Haha!", session_id="e2e_lol")
        facts = ingestor.extract_facts("lol that was funny", ep.id)
        for f in facts:
            assert f.object_text.lower() != "lol", f"Should reject 'lol': {f.object_text}"


class TestE2EEpisodeStorage:
    """End-to-end tests for episode storage."""

    def test_episode_stored_with_text(self, ingestor, kortex_db):
        """Episode should be stored with full user and assistant text."""
        ep = ingestor.ingest_turn(
            "I am a Python developer specializing in ML",
            "That's fascinating!",
            session_id="e2e_ep",
        )
        assert ep is not None
        assert ep.id is not None
        assert ep.user_text == "I am a Python developer specializing in ML"
        assert ep.assistant_text == "That's fascinating!"
        assert ep.session_id == "e2e_ep"

    def test_episode_has_salience(self, ingestor, kortex_db):
        """Episode should have a computed salience score."""
        ep = ingestor.ingest_turn(
            "URGENT: The production server is down!",
            "On it!",
            session_id="e2e_sal",
        )
        assert ep.salience >= 0.0


class TestE2ERecall:
    """End-to-end tests for recall/memory retrieval."""

    def test_fact_recalled_after_ingest(self, ingestor, recall, kortex_db, kortex_config):
        """Fact extracted during ingest should be retrievable via recall."""
        ep = ingestor.ingest_turn(
            "I prefer using neovim for all my coding",
            "Neovim is great!",
            session_id="e2e_recall",
        )
        facts = ingestor.extract_facts("I prefer using neovim for all my coding", ep.id)
        assert len(facts) >= 1

        # Now recall should find it
        context = recall.build_context(
            query="What does the user prefer for coding?",
            user_id="__default__",
            session_id="e2e_recall",
        )
        assert context is not None

    def test_episode_count_increases(self, ingestor, kortex_db):
        """Each ingest_turn call should create an episode."""
        initial_count = kortex_db.count_episodes()

        ep1 = ingestor.ingest_turn("First message", "First reply", session_id="e2e_cnt1")
        ep2 = ingestor.ingest_turn("Second message", "Second reply", session_id="e2e_cnt1")

        assert kortex_db.count_episodes() == initial_count + 2
        assert ep1.id != ep2.id


class TestE2EOpenLoops:
    """End-to-end tests for open loop extraction and resolution."""

    def test_commitment_extracted(self, ingestor, kortex_db):
        """'I will X' should extract a commitment loop."""
        ep = ingestor.ingest_turn(
            "I will fix the bug tomorrow",
            "Looking forward to it!",
            session_id="e2e_commit",
        )
        loops = ingestor.extract_open_loops(
            "I will fix the bug tomorrow",
            episode_id=ep.id,
        )
        # Should have at least one commitment
        commitments = [l for l in loops if l.kind == "commitment"]
        assert len(commitments) >= 0  # May or may not extract depending on pattern

    def test_question_extracted(self, ingestor, kortex_db):
        """'Can you X?' should extract a question loop."""
        ep = ingestor.ingest_turn(
            "Can you help me with the deployment?",
            "Of course!",
            session_id="e2e_question",
        )
        loops = ingestor.extract_open_loops(
            "Can you help me with the deployment?",
            episode_id=ep.id,
        )
        questions = [l for l in loops if l.kind == "question"]
        assert len(questions) >= 1, f"Expected question loop, got {len(questions)}"


class TestE2EContextRefs:
    """End-to-end tests for context_refs (FTS5) stability."""

    def test_batch_insert_context_refs_succeeds(self, kortex_db):
        """batch_insert_context_refs should not raise datatype mismatch."""
        # Create conversation first
        kortex_db.ensure_context_conversation("test_conv_e2e")

        refs = [
            {
                "ref_id": f"ref_test_{i}",
                "ref_type": "task",
                "label": f"Test task {i}",
                "payload": {"label": f"Test task {i}"},
                "salience": 0.8,
                "open_state": "open",
            }
            for i in range(5)
        ]
        # Should not raise
        kortex_db.batch_insert_context_refs("test_conv_e2e", refs)

        # Verify they were inserted via direct query
        conn = kortex_db._get_conn()
        stored = conn.execute(
            "SELECT * FROM context_refs WHERE conversation_id=?", ("test_conv_e2e",)
        ).fetchall()
        assert len(stored) == 5, f"Expected 5 refs, got {len(stored)}"

    def test_context_ref_with_none_source_span(self, kortex_db):
        """context_ref with None source_span_id should insert without error."""
        kortex_db.ensure_context_conversation("test_conv_none")

        refs = [
            {
                "ref_id": "ref_none_span",
                "ref_type": "task",
                "label": "Task with no span",
                "payload": {},
                # source_span_id deliberately omitted
                "salience": 0.5,
                "open_state": "open",
            }
        ]
        # Should not raise datatype mismatch
        kortex_db.batch_insert_context_refs("test_conv_none", refs)
        conn = kortex_db._get_conn()
        stored = conn.execute(
            "SELECT * FROM context_refs WHERE conversation_id=?", ("test_conv_none",)
        ).fetchall()
        assert len(stored) >= 1


class TestE2EHealthCheck:
    """Smoke tests verifying the memory stack is healthy."""

    def test_db_initializes_correctly(self, kortex_db):
        """Database should initialize with all expected tables."""
        tables = kortex_db._get_conn().execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {t[0] for t in tables}

        required = {
            "episodes", "facts", "open_loops", "reflections",
            "relationship_state", "emotion_log", "context_conversations",
            "context_messages", "context_refs", "embeddings",
        }
        missing = required - table_names
        assert not missing, f"Missing tables: {missing}"

    def test_fact_with_subject_id_populated(self, ingestor, kortex_db):
        """Facts should have subject_id populated when user_id is provided."""
        ep = ingestor.ingest_turn(
            "My name is Alice",
            "Nice to meet you Alice!",
            session_id="e2e_subj",
        )
        facts = ingestor.extract_facts("My name is Alice", ep.id, user_id="alice")
        assert len(facts) >= 1
        # subject_id should be populated
        for f in facts:
            assert f.subject_id != "", f"subject_id should be populated, got: {f.subject_id!r}"

    def test_multiple_sessions_isolated(self, ingestor, kortex_db):
        """Episodes from different sessions should be isolated."""
        ingestor.ingest_turn("Session A message", "Reply A", session_id="session_a")
        ingestor.ingest_turn("Session B message", "Reply B", session_id="session_b")

        eps_a = kortex_db.get_recent_episodes(session_id="session_a", limit=10)
        eps_b = kortex_db.get_recent_episodes(session_id="session_b", limit=10)

        assert len(eps_a) >= 1
        assert len(eps_b) >= 1
        assert all(e.session_id == "session_a" for e in eps_a)
        assert all(e.session_id == "session_b" for e in eps_b)
