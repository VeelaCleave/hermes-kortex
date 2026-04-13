"""KORTEX MemoryProvider — the Hermes integration layer.

Implements the MemoryProvider ABC to wire KORTEX into Hermes's agent loop.
One pipeline: ingest → consolidate → recall → inject.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import KortexConfig, load_kortex_config
from .calibrate import calibrate_affect, update_baseline
from .consolidate import Consolidator
from .db import KortexDB
from .affect import score_affect
from .ingest import Ingestor
from .linker import Linker
from .models import AffectSignal, Episode, Fact, OpenLoop, RelationshipState
from .promote import Promoter
from .recall import Recall
from .reflect import process_reflections
from .relationship import update_relationship
from .summaries import build_conversation_summary
from .time_utils import epoch_to_iso

logger = logging.getLogger(__name__)


KORTEX_SEARCH_SCHEMA = {
    "name": "kortex_search",
    "description": (
        "Search KORTEX experiential memory. Use this to recall past conversations, "
        "events, emotional moments, commitments, or facts about the user.\n\n"
        "Actions:\n"
        "- search: Full-text search across all memories\n"
        "- recall_episode: Get details of a specific episode by ID\n"
        "- list_facts: List known durable facts\n"
        "- list_loops: List open commitments/threads\n"
        "- list_conversations: List stored whole-conversation summaries\n"
        "- consolidate: Merge old raw episodes into summary episodes\n"
        "- status: Show memory statistics"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "search",
                    "recall_episode",
                    "list_facts",
                    "list_loops",
                    "list_conversations",
                    "consolidate",
                    "status",
                ],
            },
            "query": {
                "type": "string",
                "description": "Search query (for 'search' action)",
            },
            "episode_id": {
                "type": "integer",
                "description": "Episode ID (for 'recall_episode' action)",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 5)",
            },
        },
        "required": ["action"],
    },
}

KORTEX_IDENTITY_SCHEMA = {
    "name": "kortex_identity",
    "description": (
        "Manage identity evolution via KORTEX. Review learned personality traits "
        "and optionally promote them to SOUL.md for permanent identity changes.\n\n"
        "Actions:\n"
        "- list_pending: Show identity deltas awaiting review\n"
        "- preview: Preview a specific delta with source context\n"
        "- approve: Apply a delta to SOUL.md (makes it permanent)\n"
        "- reject: Discard a delta\n"
        "- approve_all: Apply all pending deltas above confidence threshold\n"
        "- show_soul: Display current SOUL.md content"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_pending",
                    "preview",
                    "approve",
                    "reject",
                    "approve_all",
                    "show_soul",
                ],
            },
            "delta_id": {
                "type": "integer",
                "description": "Identity delta ID (for preview/approve/reject)",
            },
            "min_confidence": {
                "type": "number",
                "description": "Minimum confidence for approve_all (default: 0.6)",
            },
        },
        "required": ["action"],
    },
}


try:
    from agent.memory_provider import MemoryProvider
except ImportError:

    class MemoryProvider:
        @property
        def name(self) -> str:
            return ""

        def is_available(self) -> bool:
            return False

        def initialize(self, session_id: str, **kwargs) -> None:
            pass

        def get_tool_schemas(self) -> list:
            return []


class KortexProvider(MemoryProvider):
    def __init__(self, config: Optional[KortexConfig] = None):
        self._config = config or KortexConfig()
        self._db: Optional[KortexDB] = None
        self._ingestor: Optional[Ingestor] = None
        self._linker: Optional[Linker] = None
        self._recall: Optional[Recall] = None
        self._promoter: Optional[Promoter] = None
        self._consolidator: Optional[Consolidator] = None
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._prefetch_cache: str = ""
        self._prefetch_lock = threading.Lock()
        self._agent_context: str = "primary"

    @property
    def name(self) -> str:
        return "kortex"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
        self._agent_context = kwargs.get("agent_context", "primary")

        db_path = self._config.db_path
        if not db_path:
            db_path = str(Path(self._hermes_home) / "kortex.db")

        self._db = KortexDB(db_path)
        self._ingestor = Ingestor(self._db)
        self._ingestor.configure_extraction(
            mode=self._config.extraction_mode,
            auxiliary_client=kwargs.get("auxiliary_client"),
        )
        self._linker = Linker(self._db)
        self._recall = Recall(self._db, self._config)
        self._promoter = Promoter(self._db, soul_path=self._config.soul_path)
        self._consolidator = Consolidator(self._db, self._linker, self._config)

        logger.info("KORTEX initialized (session=%s, db=%s)", session_id, db_path)

    def system_prompt_block(self) -> str:
        return (
            "You have KORTEX experiential memory active. You remember past conversations, "
            "emotional moments, user preferences, and commitments across sessions. "
            "Use the kortex_search tool to actively recall specific memories when relevant."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._recall:
            return ""
        with self._prefetch_lock:
            if self._prefetch_cache:
                cached = self._prefetch_cache
                self._prefetch_cache = ""
                return cached

        return self._recall.build_context(
            query, session_id=session_id or self._session_id
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._recall:
            return

        def _bg():
            try:
                ctx = self._recall.build_context(
                    query, session_id=session_id or self._session_id
                )
                with self._prefetch_lock:
                    self._prefetch_cache = ctx
            except Exception:
                logger.exception("KORTEX prefetch failed")

        threading.Thread(target=_bg, daemon=True).start()

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        if self._agent_context != "primary":
            return
        if not self._ingestor or not self._db:
            return

        sid = session_id or self._session_id

        def _bg():
            try:
                ep = self._ingestor.ingest_turn(
                    user_content, assistant_content, session_id=sid
                )

                facts = []
                reflections = []

                if self._config.auto_extract:
                    self._ingestor.extract_open_loops(user_content, ep.id)
                    facts = self._ingestor.extract_facts(user_content, ep.id)
                    resolved_loops = self._ingestor.resolve_answered_loops(
                        assistant_content, resolving_episode_id=ep.id
                    )
                    resolved_loops.extend(
                        self._ingestor.resolve_completed_commitments(
                            assistant_content, resolving_episode_id=ep.id
                        )
                    )
                else:
                    resolved_loops = []

                affect = score_affect(user_content, assistant_content)
                baseline = self._db.get_affect_baseline()
                if affect.is_significant:
                    self._db.insert_emotion_log(affect, ep.id, session_id=sid)
                updated_baseline = update_baseline(baseline, affect)
                self._db.upsert_affect_baseline(updated_baseline)
                calibrated_affect = calibrate_affect(
                    affect,
                    updated_baseline,
                    minimum_samples=self._config.affect_calibration_min_samples,
                )

                rel = self._db.get_relationship()
                days_since = 0.0
                if rel.total_turns > 0:
                    days_since = max(ep.timestamp - rel.last_updated, 0.0) / 86400
                updated_rel = update_relationship(calibrated_affect, rel, days_since)
                self._db.upsert_relationship(updated_rel)

                if self._config.auto_extract:
                    reflections = process_reflections(
                        self._db,
                        user_content,
                        assistant_content,
                        calibrated_affect,
                        ep.id,
                    )

                if self._linker:
                    self._linker.link_episode_to_facts(
                        ep.id, [fact.id for fact in facts if fact.id is not None]
                    )
                    self._linker.link_episode_to_reflections(
                        ep.id,
                        [ref.id for ref in reflections if ref.id is not None],
                    )
                    self._linker.link_episode_to_loops(
                        ep.id,
                        [loop.id for loop in resolved_loops if loop.id is not None],
                    )
                    self._linker.link_related_episodes(ep)

                if self._consolidator:
                    self._consolidator.maybe_consolidate()

                logger.debug(
                    "KORTEX ingested turn %d (salience=%.2f, valence=%d, affect=%s)",
                    ep.turn_index,
                    ep.salience,
                    ep.valence,
                    calibrated_affect.dominant_emotion,
                )
            except Exception:
                logger.exception("KORTEX sync_turn failed")

        threading.Thread(target=_bg, daemon=True).start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [KORTEX_SEARCH_SCHEMA, KORTEX_IDENTITY_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name not in {"kortex_search", "kortex_identity"}:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        if not self._db:
            return json.dumps({"error": "KORTEX not initialized"})

        try:
            if tool_name == "kortex_search":
                action = args.get("action", "")
                query = args.get("query", "")
                limit = args.get("limit", 5)

                if action == "search":
                    return self._handle_search(query, limit)
                elif action == "recall_episode":
                    return self._handle_recall_episode(args.get("episode_id"))
                elif action == "list_facts":
                    return self._handle_list_facts(limit)
                elif action == "list_loops":
                    return self._handle_list_loops(limit)
                elif action == "list_conversations":
                    return self._handle_list_conversations(limit)
                elif action == "consolidate":
                    return self._handle_consolidate(limit)
                elif action == "status":
                    return self._handle_status()
                else:
                    return json.dumps({"error": f"Unknown action: {action}"})

            return self._handle_identity_call(args)
        except Exception as exc:
            logger.exception("KORTEX tool call failed")
            return json.dumps({"error": str(exc)})

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._db:
            return
        try:
            episodes = self._db.get_episodes_for_session(self._session_id)
            summary = build_conversation_summary(
                self._session_id,
                episodes,
                messages=messages,
            )
            if summary:
                self._db.insert_conversation_summary(summary)
        except Exception:
            logger.exception("KORTEX session summary generation failed")
        logger.info("KORTEX session ended with %d messages", len(messages))

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._db:
            return ""
        logger.info("KORTEX pre-compress with %d messages", len(messages))
        return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._db or self._agent_context != "primary":
            return

        if target == "user" and action == "add":
            fact = Fact(
                subject_type="user",
                predicate="hermes_memory",
                object_text=content[:500],
                confidence=0.7,
            )
            self._db.insert_fact(fact)
            logger.debug("KORTEX mirrored memory write: %s %s", action, target)

    def shutdown(self) -> None:
        if self._db:
            self._db.close()
            logger.info("KORTEX shutdown")

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "db_path",
                "description": "Path to KORTEX SQLite database (leave empty for default)",
                "required": False,
            },
            {
                "key": "auto_extract",
                "description": "Automatically extract facts and open loops from turns",
                "required": False,
                "default": True,
                "choices": [True, False],
            },
        ]

    # -- Tool handlers -------------------------------------------------------

    def _handle_search(self, query: str, limit: int) -> str:
        if not query:
            return json.dumps({"error": "query required for search"})

        episodes = self._db.search_episodes(query, limit=limit)
        facts = self._db.search_facts(query, limit=limit)
        reflections = self._db.search_reflections(query, limit=limit)

        results = {
            "episodes": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp_iso,
                    "summary": e.summary,
                    "salience": e.salience,
                    "valence": e.valence,
                }
                for e in episodes
            ],
            "facts": [
                {"id": f.id, "text": f.object_text, "confidence": f.confidence}
                for f in facts
            ],
            "reflections": [
                {"id": r.id, "text": r.text, "kind": r.kind, "confidence": r.confidence}
                for r in reflections
            ],
        }
        return json.dumps(results)

    def _handle_recall_episode(self, episode_id: Optional[int]) -> str:
        if episode_id is None:
            return json.dumps({"error": "episode_id required"})

        ep = self._db.get_episode(episode_id)
        if not ep:
            return json.dumps({"error": f"Episode {episode_id} not found"})

        return json.dumps(
            {
                "id": ep.id,
                "session_id": ep.session_id,
                "timestamp": ep.timestamp_iso,
                "user_text": ep.user_text[:500],
                "assistant_text": ep.assistant_text[:500],
                "summary": ep.summary,
                "salience": ep.salience,
                "valence": ep.valence,
                "arousal": ep.arousal,
                "topics": ep.topics,
                "entities": ep.entities,
            }
        )

    def _handle_list_facts(self, limit: int) -> str:
        facts = self._db.get_active_facts(limit=limit)
        return json.dumps(
            {
                "facts": [
                    {
                        "id": f.id,
                        "type": f.subject_type,
                        "predicate": f.predicate,
                        "text": f.object_text,
                        "confidence": f.confidence,
                    }
                    for f in facts
                ]
            }
        )

    def _handle_list_loops(self, limit: int) -> str:
        loops = self._db.get_open_loops(limit=limit)
        return json.dumps(
            {
                "loops": [
                    {
                        "id": l.id,
                        "kind": l.kind,
                        "text": l.text,
                        "created": epoch_to_iso(l.created_at),
                    }
                    for l in loops
                ]
            }
        )

    def _handle_list_conversations(self, limit: int) -> str:
        summaries = self._db.list_conversation_summaries(limit=limit)
        return json.dumps(
            {
                "conversations": [
                    {
                        "id": summary["id"],
                        "session_id": summary["session_id"],
                        "summary_text": summary["summary_text"],
                        "episode_count": summary["episode_count"],
                        "key_entities": summary["key_entities"],
                        "updated_at": epoch_to_iso(summary["updated_at"]),
                    }
                    for summary in summaries
                ]
            }
        )

    def _handle_status(self) -> str:
        total_episodes = self._db.count_episodes()
        unconsolidated_raw_episodes = self._db.count_unconsolidated_episodes()
        facts = self._db.get_active_facts(limit=1000)
        loops = self._db.get_open_loops(limit=1000)
        reflections = self._db.get_reflections(limit=1000)
        rel = self._db.get_relationship()
        recent_emotions = self._db.get_recent_emotions(limit=5)
        baseline = self._db.get_affect_baseline()

        return json.dumps(
            {
                "total_episodes": total_episodes,
                "active_episodes": unconsolidated_raw_episodes,
                "unconsolidated_raw_episodes": unconsolidated_raw_episodes,
                "active_facts": len(facts),
                "open_loops": len(loops),
                "reflections": len(reflections),
                "relationship": {
                    "warmth": round(rel.warmth, 3),
                    "trust": round(rel.trust, 3),
                    "tension": round(rel.tension, 3),
                    "familiarity": round(rel.familiarity, 3),
                    "humor": round(rel.humor, 3),
                    "total_turns": rel.total_turns,
                },
                "recent_emotional_state": [
                    {
                        "emotion": calibrate_affect(
                            e,
                            baseline,
                            minimum_samples=self._config.affect_calibration_min_samples,
                        ).dominant_emotion,
                        "valence": calibrate_affect(
                            e,
                            baseline,
                            minimum_samples=self._config.affect_calibration_min_samples,
                        ).valence,
                        "arousal": calibrate_affect(
                            e,
                            baseline,
                            minimum_samples=self._config.affect_calibration_min_samples,
                        ).arousal,
                    }
                    for e in recent_emotions[:3]
                ],
            }
        )

    def _handle_consolidate(self, limit: Optional[int]) -> str:
        if not self._consolidator:
            return json.dumps({"error": "KORTEX consolidator not initialized"})
        return json.dumps(self._consolidator.consolidate(limit=limit))

    def _handle_identity_call(self, args: Dict[str, Any]) -> str:
        if not self._promoter:
            return json.dumps({"error": "KORTEX promoter not initialized"})

        action = args.get("action", "")
        delta_id = args.get("delta_id")

        if action == "list_pending":
            limit = args.get("limit", 20)
            deltas = self._promoter.list_pending(limit=limit)
            return json.dumps(
                {
                    "pending": [
                        {
                            "id": delta.id,
                            "text": delta.text[:500],
                            "confidence": delta.confidence,
                            "created_at": epoch_to_iso(delta.created_at),
                            "source_episode_id": delta.source_episode_id,
                        }
                        for delta in deltas
                    ]
                }
            )

        if action == "preview":
            if delta_id is None:
                return json.dumps({"error": "delta_id required"})
            return json.dumps(self._promoter.preview_delta(delta_id))

        if action == "approve":
            if delta_id is None:
                return json.dumps({"error": "delta_id required"})
            return json.dumps(self._promoter.approve_and_apply(delta_id))

        if action == "reject":
            if delta_id is None:
                return json.dumps({"error": "delta_id required"})
            return json.dumps(self._promoter.reject_delta(delta_id))

        if action == "approve_all":
            min_confidence = float(args.get("min_confidence", 0.6))
            pending = self._promoter.list_pending(limit=1000)
            eligible_ids = [
                delta.id
                for delta in pending
                if delta.id is not None and delta.confidence >= min_confidence
            ]
            return json.dumps(
                {
                    "min_confidence": min_confidence,
                    **self._promoter.approve_multiple(eligible_ids),
                }
            )

        if action == "show_soul":
            soul_content = self._promoter.get_soul_content()
            return json.dumps(
                {
                    "soul_path": str(self._promoter._resolve_soul_path()),
                    "content": soul_content,
                }
            )

        return json.dumps({"error": f"Unknown action: {action}"})
