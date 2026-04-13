from kortex.ingest import Ingestor


class TestIngestTurn:
    def test_basic_ingest(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn(
            "Hello, how are you?",
            "I'm doing well, thanks for asking!",
            session_id="test",
        )
        assert ep.id is not None
        assert ep.session_id == "test"
        assert ep.turn_index == 0
        assert ep.summary != ""

    def test_turn_index_increments(self, ingestor):
        ep1 = ingestor.ingest_turn("a", "b", session_id="s1")
        ep2 = ingestor.ingest_turn("c", "d", session_id="s1")
        assert ep1.turn_index == 0
        assert ep2.turn_index == 1

    def test_separate_sessions(self, ingestor):
        ep1 = ingestor.ingest_turn("a", "b", session_id="s1")
        ep2 = ingestor.ingest_turn("c", "d", session_id="s2")
        assert ep1.turn_index == 0
        assert ep2.turn_index == 0

    def test_text_truncation(self, ingestor):
        long_text = "x" * 10000
        ep = ingestor.ingest_turn(long_text, long_text, session_id="s1")
        assert len(ep.user_text) <= 4000
        assert len(ep.assistant_text) <= 4000


class TestSalienceScoring:
    def test_mundane_message(self, ingestor):
        ep = ingestor.ingest_turn(
            "What's the weather like?",
            "I don't have weather data.",
            session_id="s1",
        )
        assert ep.salience < 0.3

    def test_emotional_message(self, ingestor):
        ep = ingestor.ingest_turn(
            "You're such an idiot, you got everything wrong!",
            "I'm sorry, let me fix that.",
            session_id="s1",
        )
        assert ep.salience >= 0.5

    def test_caps_boost(self, ingestor):
        ep = ingestor.ingest_turn(
            "THIS IS REALLY IMPORTANT STUFF",
            "I understand.",
            session_id="s1",
        )
        assert ep.salience >= 0.3


class TestValenceScoring:
    def test_negative_valence(self, ingestor):
        ep = ingestor.ingest_turn(
            "I hate this, it's terrible and broken",
            "Sorry about that.",
            session_id="s1",
        )
        assert ep.valence < 0

    def test_positive_valence(self, ingestor):
        ep = ingestor.ingest_turn(
            "This is amazing! Thank you so much, great work!",
            "Happy to help!",
            session_id="s1",
        )
        assert ep.valence > 0

    def test_neutral_valence(self, ingestor):
        ep = ingestor.ingest_turn(
            "Please list the files in the directory.",
            "Here are the files: ...",
            session_id="s1",
        )
        assert ep.valence == 0


class TestTopicExtraction:
    def test_code_topic(self, ingestor):
        ep = ingestor.ingest_turn(
            "Help me debug this function",
            "Sure, let me look at the code.",
            session_id="s1",
        )
        assert "code" in ep.topics

    def test_multiple_topics(self, ingestor):
        ep = ingestor.ingest_turn(
            "Deploy the database migration to the server",
            "I'll handle the deployment.",
            session_id="s1",
        )
        topics = ep.topics.split(",")
        assert len(topics) >= 1


class TestOpenLoopExtraction:
    def test_commitment_extraction(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        loops = ingestor.extract_open_loops(
            "I will fix the authentication bug by Friday",
            ep.id,
        )
        assert len(loops) >= 1
        assert loops[0].kind == "commitment"

    def test_question_extraction(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        loops = ingestor.extract_open_loops(
            "Can you remind me about the deployment tomorrow?",
            ep.id,
        )
        assert len(loops) >= 1

    def test_no_loops_for_plain_text(self, ingestor, kortex_db):
        ep = ingestor.ingest_turn("msg", "resp", session_id="s1")
        loops = ingestor.extract_open_loops("The sky is blue.", ep.id)
        assert len(loops) == 0
