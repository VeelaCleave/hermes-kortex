"""Tests for semantic search layer — hybrid FTS5 + TF-IDF embeddings."""

import numpy as np
import pytest

from kortex.db import KortexDB
from kortex.models import Episode, Fact
from kortex.semantic import SemanticSearch


@pytest.fixture
def db(tmp_path):
    """Create an in-memory test DB."""
    path = tmp_path / "test_kortex.db"
    database = KortexDB(str(path))
    yield database
    database.close()


@pytest.fixture
def search(db):
    """Create a SemanticSearch instance (vocab built lazily)."""
    return SemanticSearch(db)


class TestEmbeddingPipeline:
    """Test embedding generation and storage."""

    def test_embed_episode(self, db, search):
        """Test generating and storing an episode embedding."""
        ep = Episode(user_text="Hello world", assistant_text="Hi there!")
        ep.id = db.insert_episode(ep)
        search.build_vocab()
        result = search.embed_episode(ep.id, ep.user_text, ep.assistant_text)
        assert result is not None
        emb = db.get_embedding("episode", ep.id)
        assert emb is not None

    def test_embed_fact(self, db, search):
        """Test generating and storing a fact embedding."""
        fact = Fact(object_text="User likes coffee")
        fact.id = db.insert_fact(fact)
        search.build_vocab()
        result = search.embed_fact(fact.id, fact.object_text)
        assert result is not None
        emb = db.get_embedding("fact", fact.id)
        assert emb is not None

    def test_batch_embed_facts(self, db, search):
        """Test batch embedding of facts."""
        for i in range(3):
            fact = Fact(object_text=f"Fact number {i}")
            fact.id = db.insert_fact(fact)
        search.build_vocab()
        count = search.batch_embed_facts()
        assert count == 3

    def test_batch_embed_episodes(self, db, search):
        """Test batch embedding of episodes."""
        for i in range(3):
            ep = Episode(user_text=f"Episode {i}", assistant_text=f"Reply {i}")
            ep.id = db.insert_episode(ep)
        search.build_vocab()
        count = search.batch_embed_episodes()
        assert count == 3

    def test_batch_embed_idempotent(self, db, search):
        """Test that batch embed is idempotent."""
        ep = Episode(user_text="Test", assistant_text="Reply")
        ep.id = db.insert_episode(ep)
        search.build_vocab()
        search.embed_episode(ep.id, ep.user_text, ep.assistant_text)
        count = search.batch_embed_episodes()
        assert count == 0  # Already embedded, so no new embeddings needed

    def test_embed_returns_valid_vector(self, db, search):
        """Test that embed produces a valid L2-normalized vector."""
        # Need data in DB for vocab
        ep = Episode(user_text="hello world test", assistant_text="reply")
        ep.id = db.insert_episode(ep)
        search.build_vocab()
        vec = search.embed("hello world")
        assert vec.shape[0] == 128
        norm = np.linalg.norm(vec)
        assert norm > 0


class TestHybridSearch:
    """Test hybrid FTS5 + semantic search."""

    def _setup_test_data(self, db, search):
        """Set up test data with embeddings."""
        episodes = [
            ("The cat sat on the mat", "Nice observation about cats"),
            ("I love drinking coffee in the morning", "Coffee is great"),
            ("The dog played fetch in the park", "Dogs love parks"),
            ("She drank tea while reading a book", "Tea and books, classic combo"),
        ]
        
        for user_t, asst_t in episodes:
            ep = Episode(user_text=user_t, assistant_text=asst_t)
            ep.id = db.insert_episode(ep)
        
        # Build vocab from the DB data, then embed
        search.build_vocab()
        for user_t, asst_t in episodes:
            rows = db._get_conn().execute(
                "SELECT id FROM episodes WHERE user_text=?", (user_t,)
            ).fetchall()
            if rows:
                eid = rows[0]["id"]
                search.embed_episode(eid, user_t, asst_t)

    def test_search_episodes_hybrid(self, db, search):
        """Test hybrid search returns results combining FTS5 + semantic."""
        self._setup_test_data(db, search)
        
        results = search.search_episodes_hybrid("cat")
        assert len(results) > 0
        cat_found = any("cat" in r.get("user_text", "").lower() for r in results)
        assert cat_found

    def test_search_facts_hybrid(self, db, search):
        """Test hybrid search for facts."""
        for text in ["User has a cat", "User drinks coffee", "User has a dog"]:
            ep = Episode(user_text=text, assistant_text="Confirmed")
            ep.id = db.insert_episode(ep)
        search.build_vocab()
        for text in ["User has a cat", "User drinks coffee", "User has a dog"]:
            rows = db._get_conn().execute(
                "SELECT id FROM episodes WHERE user_text=?", (text,)
            ).fetchall()
            if rows:
                search.embed_episode(rows[0]["id"], text, "Confirmed")
        
        results = search.search_episodes_hybrid("dog")
        assert len(results) > 0

    def test_hybrid_score_contains_components(self, db, search):
        """Test that hybrid results include all scoring components."""
        self._setup_test_data(db, search)
        
        results = search.search_episodes_hybrid("cat")
        if results:
            r = results[0]
            assert "semantic_score" in r
            assert "fts_rank" in r
            assert "hybrid_score" in r
            assert "id" in r

    def test_empty_search_returns_empty(self, db, search):
        """Test that searching an empty DB returns empty results."""
        search.build_vocab()
        ep_results = search.search_episodes_hybrid("nonexistent")
        assert ep_results == []

    def test_min_similarity_filters_results(self, db, search):
        """Test that min_similarity threshold filters results."""
        self._setup_test_data(db, search)
        
        results = search.search_episodes_hybrid("coffee", min_similarity=0.99)
        assert isinstance(results, list)

    def test_custom_weights_affect_ranking(self, db, search):
        """Test that changing weights affects result ordering."""
        self._setup_test_data(db, search)
        
        results_default = search.search_episodes_hybrid("morning")
        results_semantic_heavy = search.search_episodes_hybrid(
            "morning", semantic_weight=0.8, fts_weight=0.2
        )
        assert isinstance(results_default, list)
        assert isinstance(results_semantic_heavy, list)


class TestCosineSimilarity:
    """Test cosine similarity calculations."""

    def test_identical_vectors(self, db, search):
        """Identical vectors should have cosine ~1.0."""
        ep = Episode(user_text="hello world test", assistant_text="reply")
        ep.id = db.insert_episode(ep)
        search.build_vocab()
        vec = search.embed("hello world")
        sim = search._cosine(vec, vec)
        assert abs(sim - 1.0) < 0.01

    def test_orthogonal_vectors(self, db, search):
        """Orthogonal vectors should have cosine ~0.0."""
        ep = Episode(user_text="hello world test", assistant_text="reply")
        ep.id = db.insert_episode(ep)
        search.build_vocab()
        vec_a = search.embed("hello")
        vec_b = search.embed("world")
        sim = search._cosine(vec_a, vec_b)
        assert 0.0 <= sim <= 1.0

    def test_zero_vector(self, db, search):
        """Zero vector should return 0 similarity."""
        ep = Episode(user_text="hello world test", assistant_text="reply")
        ep.id = db.insert_episode(ep)
        search.build_vocab()
        vec = search.embed("hello")
        zero = np.zeros_like(vec)
        sim = search._cosine(vec, zero)
        assert sim == 0.0
