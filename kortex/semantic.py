"""Semantic search layer for KORTEX.

Provides embedding-based similarity search alongside FTS5 full-text search.
Uses TF-IDF vectorization for lightweight embeddings without requiring a sentence transformer.

Architecture:
- TF-IDF embeddings (pure numpy, ~5ms per turn)
- Cosine similarity search
- Hybrid scoring (FTS5 + semantic combined)
- Auto-embedding on ingestion
- Batch embedding for existing episodes/facts
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .db import DEFAULT_USER_ID, KortexDB

logger = logging.getLogger(__name__)

# ── TF-IDF parameters ────────────────────────────────────────────────────
VECTOR_DIM = 128  # embedding dimensionality
MIN_TOKEN_LEN = 2  # filter out single-char tokens
EMBEDDING_DIM = VECTOR_DIM  # alias for clarity


class SemanticSearch:
    """Semantic search layer: embedding-based similarity search alongside FTS5.

    Features:
    - Auto-embed episodes and facts during ingestion
    - Hybrid search combining FTS5 rank + cosine similarity
    - Batch embedding for retroactive coverage
    - Semantic nearest-neighbor lookup
    """

    def __init__(self, db: KortexDB):
        self._db = db
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(EMBEDDING_DIM, dtype=np.float32)
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
            # Filter: appear in at least 1 doc, at most 80% of docs
            # For small corpora (<10 docs), relax to at least 1 doc
            min_df = 1 if n_docs < 10 else 2
            # For small corpora, also relax max_df to 90%
            max_df_ratio = 0.9 if n_docs < 10 else 0.8
            valid_tokens = {
                token for token, df in doc_freq.items()
                if df >= min_df and df <= n_docs * max_df_ratio
            }

            # If still empty (very small corpus), accept any token
            if not valid_tokens and doc_freq:
                valid_tokens = set(doc_freq.keys())

            # Sort by frequency and index
            sorted_tokens = sorted(valid_tokens, key=lambda t: doc_freq[t], reverse=True)
            self._vocab = {token: idx for idx, token in enumerate(sorted_tokens[:EMBEDDING_DIM])}

            # Compute IDF: log(N / df) with smoothing to avoid division issues
            self._idf = np.zeros(max(len(self._vocab), EMBEDDING_DIM), dtype=np.float32)
            for token, idx in self._vocab.items():
                df = doc_freq.get(token, 1)
                # Smooth IDF: log((N + 1) / (df + 1)) + 1 avoids zero IDF
                self._idf[idx] = math.log((n_docs + 1) / (df + 1)) + 1.0

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

    def embed_episode(self, episode_id: int, user_text: str, assistant_text: str, user_id: str = DEFAULT_USER_ID) -> Optional[bytes]:
        """Generate and store embedding for an episode."""
        combined = f"{user_text} {assistant_text}"
        vec = self.embed(combined)
        self._db.insert_embedding(episode_id, "episode", vec.tobytes(), user_id=user_id)
        return vec.tobytes()

    def embed_fact(self, fact_id: int, text: str, user_id: str = DEFAULT_USER_ID) -> Optional[bytes]:
        """Generate and store embedding for a fact."""
        vec = self.embed(text)
        self._db.insert_embedding(fact_id, "fact", vec.tobytes(), user_id=user_id)
        return vec.tobytes()

    def batch_embed_episodes(self, user_id: str = DEFAULT_USER_ID) -> int:
        """Embed all unembedded episodes. Returns count of newly embedded."""
        conn = self._db._get_conn()
        # Find episodes without embeddings
        rows = conn.execute("""
            SELECT e.id, e.user_text, e.assistant_text
            FROM episodes e
            LEFT JOIN embeddings emb ON emb.entity_type='episode' AND emb.entity_id=e.id AND emb.user_id=?
            WHERE emb.entity_id IS NULL
        """, (user_id,)).fetchall()
        
        count = 0
        for row in rows:
            self.embed_episode(row["id"], row["user_text"], row["assistant_text"], user_id)
            count += 1
        return count

    def batch_embed_facts(self, user_id: str = DEFAULT_USER_ID) -> int:
        """Embed all unembedded facts. Returns count of newly embedded."""
        conn = self._db._get_conn()
        rows = conn.execute("""
            SELECT f.id, f.object_text
            FROM facts f
            LEFT JOIN embeddings emb ON emb.entity_type='fact' AND emb.entity_id=f.id AND emb.user_id=?
            WHERE emb.entity_id IS NULL
        """, (user_id,)).fetchall()
        
        count = 0
        for row in rows:
            self.embed_fact(row["id"], row["object_text"], user_id)
            count += 1
        return count

    def search_episodes_hybrid(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        user_id: str = DEFAULT_USER_ID,
        semantic_weight: float = 0.6,
        fts_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: combine FTS5 rank + cosine similarity.

        Returns list of dicts with episode data, scores, and ranking.
        """
        query_vec = self.embed(query)
        
        # Get FTS5 results
        fts_results = self._db.search_episodes(query, limit=limit * 2, user_id=user_id)
        fts_ids = {ep.id: rank for rank, ep in enumerate(fts_results)}
        
        # Get all episodes (not just salient ones) for semantic comparison
        conn = self._db._get_conn()
        rows = conn.execute(
            "SELECT id, user_text, assistant_text, salience, timestamp FROM episodes WHERE user_id=? AND is_consolidated=0 ORDER BY timestamp DESC LIMIT 200",
            (user_id,)
        ).fetchall()
        
        results = []
        for row in rows:
            ep_id = row["id"]
            ep_user_text = row["user_text"] or ""
            ep_assistant_text = row["assistant_text"] or ""
            ep_salience = row["salience"] or 0.5
            ep_timestamp = row["timestamp"] or 0
            
            # Check if this episode is in FTS results
            fts_rank = fts_ids.get(ep_id)
            
            # Get stored embedding
            emb_row = self._db.get_embedding("episode", ep_id, user_id=user_id)
            if emb_row and emb_row.get("embedding_vector"):
                ep_vec = np.frombuffer(bytes(emb_row["embedding_vector"]), dtype=np.float32)
                sim = self._cosine(query_vec, ep_vec)
                
                if sim >= min_similarity:
                    # Hybrid score: normalize FTS rank + combine
                    fts_score = 1.0 / (1 + fts_rank) if fts_rank is not None else 0.0
                    hybrid_score = (fts_score * fts_weight) + (sim * semantic_weight)
                    
                    results.append({
                        "id": ep_id,
                        "user_text": ep_user_text,
                        "assistant_text": ep_assistant_text,
                        "salience": ep_salience,
                        "timestamp": ep_timestamp,
                        "semantic_score": round(sim, 4),
                        "fts_rank": fts_rank,
                        "hybrid_score": round(hybrid_score, 4),
                    })
        
        # Sort by hybrid score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:limit]

    def search_facts_hybrid(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        user_id: str = DEFAULT_USER_ID,
        semantic_weight: float = 0.6,
        fts_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Hybrid search for facts: combine FTS5 rank + cosine similarity."""
        query_vec = self.embed(query)
        
        # Get FTS5 results on facts
        fts_results = self._db.search_facts(query, limit=limit * 2, user_id=user_id)
        fts_ids = {fact.id: rank for rank, fact in enumerate(fts_results)}
        
        # Get all active facts
        conn = self._db._get_conn()
        rows = conn.execute(
            "SELECT id, object_text, confidence FROM facts WHERE user_id=? AND status='active' LIMIT 200",
            (user_id,)
        ).fetchall()
        
        results = []
        for row in rows:
            fact_id = row["id"]
            fact_text = row["object_text"] or ""
            fact_confidence = row["confidence"] or 0.5
            
            fts_rank = fts_ids.get(fact_id)
            
            emb_row = self._db.get_embedding("fact", fact_id, user_id=user_id)
            if emb_row and emb_row.get("embedding_vector"):
                fact_vec = np.frombuffer(bytes(emb_row["embedding_vector"]), dtype=np.float32)
                sim = self._cosine(query_vec, fact_vec)
                
                if sim >= min_similarity:
                    fts_score = 1.0 / (1 + fts_rank) if fts_rank is not None else 0.0
                    hybrid_score = (fts_score * fts_weight) + (sim * semantic_weight)
                    
                    results.append({
                        "id": fact_id,
                        "text": fact_text,
                        "confidence": fact_confidence,
                        "semantic_score": round(sim, 4),
                        "fts_rank": fts_rank,
                        "hybrid_score": round(hybrid_score, 4),
                    })
        
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
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
