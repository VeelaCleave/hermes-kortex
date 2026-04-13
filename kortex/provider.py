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
from .db import KortexDB
from .ingest import Ingestor
from .models import Episode, Fact, OpenLoop
from .recall import Recall

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
        self._recall: Optional[Recall] = None
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
        self._recall = Recall(self._db, self._config)

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

                if self._config.auto_extract:
                    self._ingestor.extract_open_loops(user_content, ep.id)
                    self._ingestor.extract_facts(user_content, ep.id)
                    self._ingestor.resolve_answered_loops(assistant_content)
                    self._ingestor.resolve_completed_commitments(assistant_content)

                logger.debug(
                    "KORTEX ingested turn %d (salience=%.2f, valence=%d)",
                    ep.turn_index,
                    ep.salience,
                    ep.valence,
                )
            except Exception:
                logger.exception("KORTEX sync_turn failed")

        threading.Thread(target=_bg, daemon=True).start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [KORTEX_SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "kortex_search":
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        if not self._db:
            return json.dumps({"error": "KORTEX not initialized"})

        action = args.get("action", "")
        query = args.get("query", "")
        limit = args.get("limit", 5)

        try:
            if action == "search":
                return self._handle_search(query, limit)
            elif action == "recall_episode":
                return self._handle_recall_episode(args.get("episode_id"))
            elif action == "list_facts":
                return self._handle_list_facts(limit)
            elif action == "list_loops":
                return self._handle_list_loops(limit)
            elif action == "status":
                return self._handle_status()
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as exc:
            logger.exception("KORTEX tool call failed")
            return json.dumps({"error": str(exc)})

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._db:
            return
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
                        "created": l.created_at.isoformat(),
                    }
                    for l in loops
                ]
            }
        )

    def _handle_status(self) -> str:
        total_episodes = self._db.count_episodes()
        facts = self._db.get_active_facts(limit=1000)
        loops = self._db.get_open_loops(limit=1000)
        reflections = self._db.get_reflections(limit=1000)
        rel = self._db.get_relationship()

        return json.dumps(
            {
                "total_episodes": total_episodes,
                "active_facts": len(facts),
                "open_loops": len(loops),
                "reflections": len(reflections),
                "relationship": {
                    "warmth": rel.warmth,
                    "trust": rel.trust,
                    "tension": rel.tension,
                    "familiarity": rel.familiarity,
                    "total_turns": rel.total_turns,
                },
            }
        )
