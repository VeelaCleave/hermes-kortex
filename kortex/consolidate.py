"""Episode consolidation / compaction for KORTEX."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .config import KortexConfig
from .db import DEFAULT_USER_ID, KortexDB
from .linker import Linker
from .models import Episode
from .summaries import build_conversation_summary


class Consolidator:
    """Merge raw episodes into summary episodes while preserving links."""

    def __init__(self, db: KortexDB, linker: Linker, config: KortexConfig):
        self._db = db
        self._linker = linker
        self._config = config

    def maybe_consolidate(self) -> Dict[str, int | bool | List[int]]:
        active_count = self._db.count_unconsolidated_episodes()
        if active_count <= self._config.consolidation_threshold:
            return {
                "triggered": False,
                "active_episodes": active_count,
                "threshold": self._config.consolidation_threshold,
                "summary_episode_ids": [],
                "episodes_consolidated": 0,
                "summary_episodes_created": 0,
            }

        result = self.consolidate(limit=self._config.consolidation_batch_size)
        result["triggered"] = True
        result["active_episodes"] = active_count
        result["threshold"] = self._config.consolidation_threshold
        return result

    def consolidate(self, limit: int | None = None) -> Dict[str, int | List[int]]:
        batch_limit = limit or self._config.consolidation_batch_size
        episodes = self._db.get_unconsolidated_episodes(limit=batch_limit)
        if not episodes:
            return {
                "summary_episode_ids": [],
                "episodes_consolidated": 0,
                "summary_episodes_created": 0,
            }

        grouped: Dict[str, List[Episode]] = defaultdict(list)
        for episode in episodes:
            grouped[episode.session_id].append(episode)

        created_summary_ids: List[int] = []
        consolidated_count = 0
        episode_to_summary: Dict[int, int] = {}

        for session_id, session_episodes in grouped.items():
            summary_episode_id = self._create_summary_episode(
                session_id, session_episodes
            )
            if not summary_episode_id:
                continue
            created_summary_ids.append(summary_episode_id)
            for episode in session_episodes:
                if episode.id:
                    episode_to_summary[episode.id] = summary_episode_id

        for session_episodes in grouped.values():
            source_episode_ids = [
                episode.id for episode in session_episodes if episode.id
            ]
            if not source_episode_ids:
                continue
            summary_episode_id = episode_to_summary[source_episode_ids[0]]
            self._db.mark_episodes_consolidated(source_episode_ids, summary_episode_id)
            self._copy_links(
                source_episode_ids,
                summary_episode_id,
                session_episodes[0].user_id,
                episode_to_summary,
            )
            consolidated_count += len(session_episodes)

        return {
            "summary_episode_ids": created_summary_ids,
            "episodes_consolidated": consolidated_count,
            "summary_episodes_created": len(created_summary_ids),
        }

    def _create_summary_episode(
        self, session_id: str, session_episodes: List[Episode]
    ) -> Optional[int]:
        if not session_episodes:
            return None

        latest_summary = self._latest_summary_for_session(session_id)
        summary_text = None
        key_entities = ""
        if latest_summary:
            summary_text = latest_summary.get("summary_text", "").strip() or None
            key_entities = latest_summary.get("key_entities", "")

        if not summary_text:
            fallback = build_conversation_summary(
                session_id,
                session_episodes,
                user_id=session_episodes[0].user_id
                if session_episodes
                else DEFAULT_USER_ID,
            )
            if fallback:
                summary_text = fallback.get("summary_text", "").strip() or None
                key_entities = fallback.get("key_entities", "")

        if not summary_text:
            summary_text = "Conversation summary unavailable"

        merged_topics = _merge_csv([episode.topics for episode in session_episodes])
        merged_entities = _merge_csv(
            [episode.entities for episode in session_episodes] + [key_entities]
        )
        average_count = max(len(session_episodes), 1)
        summary_episode = Episode(
            user_id=session_episodes[0].user_id,
            session_id=session_id,
            turn_index=max(
                (episode.turn_index for episode in session_episodes), default=0
            ),
            timestamp=max(
                (episode.timestamp for episode in session_episodes), default=0.0
            ),
            user_text="",
            assistant_text="",
            summary=summary_text,
            salience=max(
                0.6,
                max((episode.salience for episode in session_episodes), default=0.0),
            ),
            valence=round(
                sum(episode.valence for episode in session_episodes) / average_count
            ),
            arousal=sum(episode.arousal for episode in session_episodes)
            / average_count,
            topics=merged_topics,
            entities=merged_entities,
            raw_preserved=False,
        )
        summary_episode_id = self._db.insert_episode(summary_episode)

        return summary_episode_id

    def _latest_summary_for_session(self, session_id: str) -> Optional[dict]:
        summaries = self._db.list_conversation_summaries(limit=1, session_id=session_id)
        return summaries[0] if summaries else None

    def _copy_links(
        self,
        source_episode_ids: List[int],
        summary_episode_id: int,
        user_id: str,
        episode_to_summary: Dict[int, int],
    ) -> None:
        copied_edges = set()

        for source_episode_id in source_episode_ids:
            for link in self._db.get_links_from(
                "episode", source_episode_id, limit=None
            ):
                destination_type = link["dst_type"]
                destination_id = link["dst_id"]
                if destination_type == "episode":
                    destination_id = episode_to_summary.get(
                        destination_id, destination_id
                    )
                if (
                    destination_type == "episode"
                    and destination_id == summary_episode_id
                ):
                    continue
                edge = (
                    "episode",
                    summary_episode_id,
                    destination_type,
                    destination_id,
                    link["relation"],
                )
                if edge in copied_edges:
                    continue
                if self._db.link_exists(
                    "episode",
                    summary_episode_id,
                    destination_type,
                    destination_id,
                    link["relation"],
                ):
                    copied_edges.add(edge)
                    continue
                copied_edges.add(edge)
                self._db.insert_link(
                    "episode",
                    summary_episode_id,
                    destination_type,
                    destination_id,
                    link["relation"],
                    weight=link["weight"],
                    user_id=user_id,
                )

            for link in self._db.get_links_to("episode", source_episode_id, limit=None):
                source_type = link["src_type"]
                source_id = link["src_id"]
                if source_type == "episode":
                    source_id = episode_to_summary.get(source_id, source_id)
                if source_type == "episode" and source_id == summary_episode_id:
                    continue
                edge = (
                    source_type,
                    source_id,
                    "episode",
                    summary_episode_id,
                    link["relation"],
                )
                if edge in copied_edges:
                    continue
                if self._db.link_exists(
                    source_type,
                    source_id,
                    "episode",
                    summary_episode_id,
                    link["relation"],
                ):
                    copied_edges.add(edge)
                    continue
                copied_edges.add(edge)
                self._db.insert_link(
                    source_type,
                    source_id,
                    "episode",
                    summary_episode_id,
                    link["relation"],
                    weight=link["weight"],
                    user_id=user_id,
                )


def _merge_csv(values: List[str]) -> str:
    merged: List[str] = []
    seen = set()
    for value in values:
        for part in [item.strip() for item in value.split(",") if item.strip()]:
            lowered = part.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(part)
    return ",".join(merged)
