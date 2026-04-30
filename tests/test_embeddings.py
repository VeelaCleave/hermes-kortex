"""Tests for semantic embeddings table and DB methods."""

import math
import numpy as np
import pytest

from kortex.db import KortexDB
from kortex.models import Episode, Fact


@pytest.fixture
def db(tmp_path):
    db = KortexDB(str(tmp_path / "test_embeddings.db"))
    yield db
    db.close()


class TestEmbeddingsCRUD:
    """Test basic embedding CRUD operations."""

    def test_insert_and_get_embedding(self, db):
        """Test inserting and retrieving an embedding."""
        vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        row_id = db.insert_embedding(1, "episode", vec.tobytes())
        assert row_id > 0

        result = db.get_embedding("episode", 1)
        assert result is not None
        assert result["entity_id"] == 1
        assert result["entity_type"] == "episode"
        retrieved = np.frombuffer(bytes(result["embedding_vector"]), dtype=np.float32)
        np.testing.assert_array_almost_equal(retrieved, vec)

    def test_insert_multiple_embeddings_different_types(self, db):
        """Test inserting embeddings for different entity types."""
        vec1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vec2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        vec3 = np.array([0.7, 0.8, 0.9], dtype=np.float32)

        db.insert_embedding(1, "episode", vec1.tobytes())
        db.insert_embedding(1, "fact", vec2.tobytes())
        db.insert_embedding(1, "reflection", vec3.tobytes())

        ep = db.get_embedding("episode", 1)
        fact = db.get_embedding("fact", 1)
        refl = db.get_embedding("reflection", 1)

        assert ep is not None
        assert fact is not None
        assert refl is not None
        assert ep["embedding_vector"] != fact["embedding_vector"]
        assert ep["embedding_vector"] != refl["embedding_vector"]

    def test_get_nonexistent_embedding(self, db):
        """Test retrieving an embedding that doesn't exist."""
        result = db.get_embedding("episode", 999)
        assert result is None

    def test_delete_embedding(self, db):
        """Test deleting an embedding."""
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        db.insert_embedding(1, "episode", vec.tobytes())

        assert db.get_embedding("episode", 1) is not None
        deleted = db.delete_embedding("episode", 1)
        assert deleted == 1
        assert db.get_embedding("episode", 1) is None

    def test_delete_nonexistent_embedding(self, db):
        """Test deleting an embedding that doesn't exist."""
        deleted = db.delete_embedding("episode", 999)
        assert deleted == 0

    def test_embedding_stores_user_id(self, db):
        """Test that embeddings are scoped by user_id."""
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        db.insert_embedding(1, "episode", vec.tobytes(), user_id="user_a")
        db.insert_embedding(1, "episode", vec.tobytes(), user_id="user_b")

        result_a = db.get_embedding("episode", 1, user_id="user_a")
        result_b = db.get_embedding("episode", 1, user_id="user_b")

        assert result_a is not None
        assert result_b is not None
        assert result_a["user_id"] == "user_a"
        assert result_b["user_id"] == "user_b"

    def test_embedding_table_exists_in_schema(self, db):
        """Test that the embeddings table is created in the schema."""
        conn = db._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
        ).fetchall()
        assert len(tables) == 1

    def test_embedding_table_has_correct_columns(self, db):
        """Test that the embeddings table has the expected columns."""
        conn = db._get_conn()
        columns = conn.execute("PRAGMA table_info(embeddings)").fetchall()
        col_names = [c["name"] for c in columns]
        expected = ["id", "entity_id", "entity_type", "embedding_vector", "created_at", "user_id"]
        for e in expected:
            assert e in col_names

    def test_embedding_table_has_indexes(self, db):
        """Test that the embeddings table has the expected indexes."""
        conn = db._get_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='embeddings'"
        ).fetchall()
        index_names = [i["name"] for i in indexes]
        assert "idx_embeddings_entity" in index_names
        assert "idx_embeddings_user" in index_names
