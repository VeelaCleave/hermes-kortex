"""Semantic memory graph for KORTEX.

Uses TF-IDF vectorization to create lightweight embeddings for episodes and facts.
Enables semantic similarity search without requiring a sentence transformer model.

Architecture:
- TF-IDF embeddings (pure numpy, ~5ms per turn)
- Cosine similarity search
- Hybrid scoring (FTS5 + semantic)
- Graph enhancement with semantic edges
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .db import DEFAULT_USER_ID, KortexDB

logger = logging.getLogger(__name__)

# ── TF-IDF parameters ────────────────────────────────────────────────────
VECTOR_DIM = 128  # embedding dimensionality
MIN_TOKEN_LEN = 2  # filter out single-char tokens


class SemanticGraph:
    """Semantic memory graph using TF-IDF embeddings.

    Embeds episodes and facts into a shared vector space, enabling:
    - Semantic similarity search (cosine distance)
    - Nearest-neighbor traversal
    - Hybrid scoring (FTS5 + semantic)
    """

    def __init__(self, db: KortexDB):
        self._db = db
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(VECTOR_DIM, dtype=np.float32)
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, strip punctuation, filter short tokens."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) >= MIN_TOKEN_LEN]

    def build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from existing episodes in the DB."""
        try:
            conn = self._db._get_conn()
            rows = conn.execute(
                "SELECT user_text, assistant_text FROM episodes"
            ).fetchall()

            texts = [row["user_text"] + " " + row["assistant_text"] for row in rows]
            if not texts:
                self._built = True
                return self._vocab

            # Build vocabulary from corpus
            all_tokens = [self._tokenize(t) for t in texts]
            doc_freq = Counter()
            for tokens in all_tokens:
                doc_freq.update(set(tokens))

            n_docs = max(len(texts), 1)
            # Filter: appear in at least 2 docs, at most 80% of docs
            valid_tokens = {
                token for token, df in doc_freq.items()
                if df >= 2 and df <= n_docs * 0.8
            }

            # Sort by frequency and index
            sorted_tokens = sorted(valid_tokens, key=lambda t: doc_freq[t], reverse=True)
            self._vocab = {token: idx for idx, token in enumerate(sorted_tokens[:VECTOR_DIM])}

            # Compute IDF: log(N / df)
            self._idf = np.zeros(max(len(self._vocab), VECTOR_DIM), dtype=np.float32)
            for token, idx in self._vocab.items():
                df = doc_freq.get(token, 1)
                self._idf[idx] = math.log(n_docs / df)

            self._built = True
            return self._vocab
        except Exception:
            logger.debug("SemanticGraph build_vocab fallback", exc_info=True)
            self._built = True
            return self._vocab

    def embed(self, text: str) -> np.ndarray:
        """Generate a TF-IDF embedding for a text string."""
        if not self._built:
            self.build_vocab()

        if not self._vocab:
            return np.zeros(self._idf.shape[0], dtype=np.float32)

        tokens = self._tokenize(text)
        vec = np.zeros(self._idf.shape[0], dtype=np.float32)

        # Count term frequencies
        counts = Counter(tokens)

        for token, count in counts.items():
            idx = self._vocab.get(token)
            if idx is not None:
                # Smooth TF: 0.5 + log(tf)
                tf = 0.5 + math.log(count)
                vec[idx] = tf * self._idf[idx]

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def find_similar_episodes(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Tuple[int, float]]:
        """Find episodes semantically similar to a query.

        Returns list of (episode_id, cosine_similarity) sorted by similarity.
        """
        query_vec = self.embed(query)
        results = []

        # Get candidate episodes from DB
        episodes = self._db.get_salient_episodes(
            min_salience=0.1, limit=100, user_id=user_id
        )

        for ep in episodes:
            if not ep.id:
                continue
            # Get stored embedding from DB
            emb_row = self._db.get_semantic_embedding("episode", ep.id, user_id=user_id)
            if emb_row and emb_row.get("embedding"):
                ep_vec = np.frombuffer(
                    bytes(emb_row["embedding"]), dtype=np.float32
                )
            else:
                continue

            sim = self._cosine(query_vec, ep_vec)
            if sim >= min_similarity:
                results.append((ep.id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def find_similar_facts(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        user_id: str = DEFAULT_USER_ID,
    ) -> List[Tuple[int, float]]:
        """Find facts semantically similar to a query."""
        query_vec = self.embed(query)
        results = []

        facts = self._db.get_active_facts(
            subject_type="user", limit=100, user_id=user_id
        )

        for fact in facts:
            if not fact.id:
                continue
            emb_row = self._db.get_semantic_embedding("fact", fact.id, user_id=user_id)
            if emb_row and emb_row.get("embedding"):
                fact_vec = np.frombuffer(
                    bytes(emb_row["embedding"]), dtype=np.float32
                )
            else:
                continue

            sim = self._cosine(query_vec, fact_vec)
            if sim >= min_similarity:
                results.append((fact.id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized vectors."""
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
