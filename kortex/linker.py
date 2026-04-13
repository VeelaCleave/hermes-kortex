"""Heuristic graph linking for KORTEX memories."""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Dict, List, Set, Tuple

from .db import KortexDB
from .models import Episode


class Linker:
    """Creates and traverses lightweight memory graph edges."""

    def __init__(self, db: KortexDB):
        self._db = db

    def link_episode_to_facts(self, episode_id: int, fact_ids: List[int]) -> int:
        """Create episode→fact links. Returns count of links created."""
        created = 0
        for fact_id in self._unique_positive_ids(fact_ids):
            created += self._create_link(
                "episode", episode_id, "fact", fact_id, "extracted_from"
            )
            for old_fact in self._db.get_facts_superseded_by(fact_id, limit=10):
                if old_fact.id:
                    self.link_superseded_facts(old_fact.id, fact_id)
        return created

    def link_episode_to_reflections(
        self, episode_id: int, reflection_ids: List[int]
    ) -> int:
        """Create episode→reflection links. Returns count of links created."""
        created = 0
        for reflection_id in self._unique_positive_ids(reflection_ids):
            created += self._create_link(
                "episode", episode_id, "reflection", reflection_id, "triggered"
            )
        return created

    def link_related_episodes(self, episode: Episode, max_lookback: int = 50) -> int:
        """Find and link related episodes by shared entities/topics. Returns count."""
        if not episode.id:
            return 0

        created = self._link_entities_to_episode(episode)
        tokens = self._episode_tokens(episode)
        if not tokens:
            return created

        recent = self._db.get_recent_episodes(limit=max_lookback + 1)
        for other in recent:
            if not other.id or other.id == episode.id:
                continue

            self._link_entities_to_episode(other)
            other_tokens = self._episode_tokens(other)
            if not other_tokens:
                continue

            score = self._jaccard(tokens, other_tokens)
            if score < 0.3:
                continue

            created += self._create_link(
                "episode", episode.id, "episode", other.id, "related_to", weight=score
            )
            created += self._create_link(
                "episode", other.id, "episode", episode.id, "related_to", weight=score
            )

        return created

    def link_superseded_facts(self, old_fact_id: int, new_fact_id: int) -> None:
        """Create fact→fact supersession link."""
        if old_fact_id <= 0 or new_fact_id <= 0:
            return
        self._create_link("fact", old_fact_id, "fact", new_fact_id, "supersedes")

    def link_contradicting_facts(self, old_fact_id: int, new_fact_id: int) -> None:
        """Create fact↔fact contradiction links."""
        if old_fact_id <= 0 or new_fact_id <= 0:
            return
        self._create_link("fact", old_fact_id, "fact", new_fact_id, "contradicts")
        self._create_link("fact", new_fact_id, "fact", old_fact_id, "contradicts")

    def link_episode_to_loops(self, episode_id: int, loop_ids: List[int]) -> int:
        """Create episode→open_loop resolution links."""
        created = 0
        for loop_id in self._unique_positive_ids(loop_ids):
            created += self._create_link(
                "episode", episode_id, "open_loop", loop_id, "resolves"
            )
        return created

    def get_related_episodes(self, episode_id: int, limit: int = 5) -> List[int]:
        """Get episode IDs linked to this episode."""
        return [
            link["dst_id"]
            for link in self._db.get_links_from(
                "episode", episode_id, relation="related_to", limit=limit
            )
            if link["dst_type"] == "episode"
        ]

    def get_episode_facts(self, episode_id: int) -> List[int]:
        """Get fact IDs linked to this episode."""
        return [
            link["dst_id"]
            for link in self._db.get_links_from(
                "episode", episode_id, relation="extracted_from", limit=100
            )
            if link["dst_type"] == "fact"
        ]

    def get_fact_episodes(self, fact_id: int) -> List[int]:
        """Get episode IDs where this fact was extracted."""
        return [
            link["src_id"]
            for link in self._db.get_links_to(
                "fact", fact_id, relation="extracted_from", limit=100
            )
            if link["src_type"] == "episode"
        ]

    def traverse(
        self,
        entity_ids: List[int],
        max_hops: int = 2,
        max_results: int = 50,
        hop_decay: float = 0.5,
    ) -> List[dict]:
        """Traverse the memory graph from seed entities and return ranked nodes."""
        seeds = [
            ("entity", entity_id) for entity_id in self._unique_positive_ids(entity_ids)
        ]
        if not seeds or max_hops <= 0:
            return []

        queue = deque((node_type, node_id, 0, 1.0) for node_type, node_id in seeds)
        seed_nodes = set(seeds)
        best_scores: Dict[Tuple[str, int], float] = {seed: 1.0 for seed in seeds}

        while queue:
            node_type, node_id, hops, score = queue.popleft()
            if hops >= max_hops:
                continue

            for next_type, next_id, relation, edge_weight in self._neighbors(
                node_type, node_id
            ):
                next_score = (
                    score * hop_decay * edge_weight * self._relation_weight(relation)
                )
                if next_score <= 0:
                    continue
                key = (next_type, next_id)
                if next_score <= best_scores.get(key, 0.0):
                    continue
                best_scores[key] = next_score
                queue.append((next_type, next_id, hops + 1, next_score))

        ranked = [
            {"node_type": node_type, "node_id": node_id, "score": score}
            for (node_type, node_id), score in best_scores.items()
            if (node_type, node_id) not in seed_nodes
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:max_results]

    def _link_entities_to_episode(self, episode: Episode) -> int:
        if not episode.id:
            return 0
        created = 0
        for entity_name in self._split_csv(episode.entities):
            created += self._create_link(
                "entity",
                self._entity_id(entity_name),
                "episode",
                episode.id,
                "co_occurs",
            )
        return created

    def _neighbors(
        self, node_type: str, node_id: int
    ) -> List[Tuple[str, int, str, float]]:
        neighbors: List[Tuple[str, int, str, float]] = []
        for link in self._db.get_links_from(node_type, node_id, limit=100):
            neighbors.append(
                (link["dst_type"], link["dst_id"], link["relation"], link["weight"])
            )
        for link in self._db.get_links_to(node_type, node_id, limit=100):
            neighbors.append(
                (link["src_type"], link["src_id"], link["relation"], link["weight"])
            )
        return neighbors

    def _create_link(
        self,
        src_type: str,
        src_id: int,
        dst_type: str,
        dst_id: int,
        relation: str,
        weight: float = 1.0,
    ) -> int:
        if src_id <= 0 or dst_id <= 0:
            return 0
        if self._db.link_exists(src_type, src_id, dst_type, dst_id, relation):
            return 0
        self._db.insert_link(
            src_type, src_id, dst_type, dst_id, relation, weight=weight
        )
        return 1

    @classmethod
    def _episode_tokens(cls, episode: Episode) -> Set[str]:
        return set(cls._split_csv(episode.entities)) | set(
            cls._split_csv(episode.topics)
        )

    @staticmethod
    def _split_csv(value: str) -> List[str]:
        return [part.strip().lower() for part in value.split(",") if part.strip()]

    @staticmethod
    def _jaccard(left: Set[str], right: Set[str]) -> float:
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @staticmethod
    def _relation_weight(relation: str) -> float:
        return {
            "extracted_from": 1.0,
            "related_to": 0.7,
            "co_occurs": 0.6,
            "triggered": 0.6,
            "resolves": 0.7,
            "supersedes": 0.4,
            "contradicts": 0.3,
        }.get(relation, 0.5)

    @staticmethod
    def _entity_id(name: str) -> int:
        digest = hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()
        return int(digest[:15], 16)

    @staticmethod
    def _unique_positive_ids(values: List[int]) -> List[int]:
        seen = set()
        result = []
        for value in values:
            if not isinstance(value, int) or value <= 0 or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result
