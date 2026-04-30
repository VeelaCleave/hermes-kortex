"""KORTEX MemoryProvider — the Hermes integration layer.

Implements the MemoryProvider ABC to wire KORTEX into Hermes's agent loop.
One pipeline: ingest → consolidate → recall → inject.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import KortexConfig, load_kortex_config
from .calibrate import calibrate_affect, update_baseline
from .consolidate import Consolidator
from .db import DEFAULT_USER_ID, KortexDB
from .affect import score_affect
from .ingest import Ingestor
from .linker import Linker
from .models import AffectSignal, Episode, Fact, OpenLoop, RelationshipState
from .promote import Promoter
from .recall import Recall
from .reflect import process_reflections
from .relationship import update_relationship
from .summaries import build_conversation_summary
from .time_utils import epoch_to_display, epoch_to_iso
from .export import export_to_json, import_from_json
from .ocean import score_turn as score_ocean_turn

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

KORTEX_EXPORT_SCHEMA = {
    "name": "kortex_export",
    "description": (
        "Export or import KORTEX memory as JSON backups.\n\n"
        "Actions:\n"
        "- export: Dump memories to JSON\n"
        "- import: Restore memories from JSON"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["export", "import"]},
            "user_id": {"type": "string", "description": "Optional user scope"},
            "start": {"type": "string", "description": "Optional start timestamp"},
            "end": {"type": "string", "description": "Optional end timestamp"},
            "types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional memory types to include",
            },
            "payload": {
                "type": "string",
                "description": "JSON payload for import action",
            },
            "allow_override": {
                "type": "boolean",
                "description": "Allow schema mismatch on import",
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

        def get_tool_schemas(self):
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
        self._user_id: str = DEFAULT_USER_ID

    @property
    def name(self) -> str:
        return "kortex"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
        self._agent_context = kwargs.get("agent_context", "primary")
        agent_identity = kwargs.get("agent_identity", "")
        self._user_id = kwargs.get("user_id", DEFAULT_USER_ID)
        if agent_identity and self._user_id == DEFAULT_USER_ID:
            self._user_id = agent_identity

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
        self._recall = Recall(self._db, self._config, linker=self._linker)
        self._promoter = Promoter(self._db, soul_path=self._config.soul_path)
        self._consolidator = Consolidator(self._db, self._linker, self._config)

        logger.info("KORTEX initialized (session=%s, db=%s)", session_id, db_path)

    def system_prompt_block(self) -> str:
        if not self._config.passive_context_hint:
            return ""
        return (
            "You have KORTEX experiential memory active. Relevant memory context is "
            "usually injected automatically before each turn. Use KORTEX tools only "
            "when you need to inspect, expand, or export memory beyond the passive context."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._recall or not self._config.passive_recall:
            return ""
        with self._prefetch_lock:
            if self._prefetch_cache:
                cached = self._prefetch_cache
                self._prefetch_cache = ""
                return cached

        return self._recall.build_context(
            query, session_id=session_id or self._session_id, user_id=self._user_id
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._recall or not self._config.passive_recall:
            return

        def _bg():
            try:
                ctx = self._recall.build_context(
                    query,
                    session_id=session_id or self._session_id,
                    user_id=self._user_id,
                )
                with self._prefetch_lock:
                    self._prefetch_cache = ctx
            except Exception:
                logger.exception("KORTEX prefetch failed")

        threading.Thread(target=_bg, daemon=True).start()

    def _filter_user_content(self, content: str) -> tuple[str, bool]:
        """Filter out system prompts, tool outputs, and other non-conversation content from user messages.
        
        Returns (cleaned_content, skip_entire_turn).
        skip_entire_turn is True if the user content is purely system noise.
        """
        if not content:
            return ("", True)
        
        # Patterns that indicate system-generated content (not real user input)
        system_patterns = [
            "[System note:",
            "[memory-context]",
            "Recalled memories:",
            "Open threads:",
            "Learned behaviors:",
            "Known facts:",
            "What general pattern of task did the user just complete",
            "Review the conversation above",
            "Consider whether a skill should be saved",
            "check my work over at",
            "make sure it's working as intended",
            "maybe you could try out",
            "you can then deep dive",
            "figure out what's not working",
            "what's over-engineered",
            "what's missing",
        ]
        
        for pattern in system_patterns:
            if pattern.lower() in content.lower():
                # Strip the system noise but keep any real user text
                lines = content.split("\n")
                real_lines = []
                for line in lines:
                    skip = False
                    for p in system_patterns:
                        if p.lower() in line.lower():
                            skip = True
                            break
                    if not skip:
                        real_lines.append(line)
                
                cleaned = "\n".join(real_lines).strip()
                if cleaned:
                    return (cleaned, False)
                return ("", True)
        
        return (content, False)
    
    def _filter_assistant_content(self, content: str) -> str:
        """Filter out tool outputs and system noise from assistant responses."""
        if not content:
            return content
        
        # Remove tool result blocks
        lines = content.split("\n")
        clean_lines = []
        in_tool_block = False
        
        for line in lines:
            # Skip lines that are clearly tool/system output
            if any(pattern in line for pattern in [
                "EXIT_CODE", "EXIT: 124", "EXIT: 0",
                "passed in", "failed in",
                "KORTEX Memory", "KORTEX initialized",
                "PRAGMA", "sqlite3",
            ]):
                continue
            clean_lines.append(line)
        
        return "\n".join(clean_lines)
    
    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        if self._agent_context != "primary":
            return
        if not self._ingestor or not self._db:
            return

        # ── Filter out garbage before ingesting ──────────────────────────
        user_clean, skip_user = self._filter_user_content(user_content)
        assistant_clean = self._filter_assistant_content(assistant_content)

        # If both sides are empty after filtering, skip the turn entirely
        if not user_clean.strip() and not assistant_clean.strip():
            return

        sid = session_id or self._session_id

        def _bg():
            try:
                ep = self._ingestor.ingest_turn(
                    user_clean,
                    assistant_clean,
                    session_id=sid,
                    user_id=self._user_id,
                )

                facts = []
                reflections = []

                if self._config.auto_extract:
                    self._ingestor.extract_open_loops(
                        user_clean, ep.id, user_id=self._user_id
                    )
                    facts = self._ingestor.extract_facts(
                        user_clean, ep.id, user_id=self._user_id
                    )
                    resolved_loops = self._ingestor.resolve_answered_loops(
                        assistant_clean,
                        resolving_episode_id=ep.id,
                        user_id=self._user_id,
                    )
                    resolved_loops.extend(
                        self._ingestor.resolve_completed_commitments(
                            assistant_clean,
                            resolving_episode_id=ep.id,
                            user_id=self._user_id,
                        )
                    )
                else:
                    resolved_loops = []

                affect = score_affect(user_clean, assistant_clean)
                baseline = self._db.get_affect_baseline(user_id=self._user_id)
                if affect.is_significant:
                    self._db.insert_emotion_log(
                        affect, ep.id, session_id=sid, user_id=self._user_id
                    )
                updated_baseline = update_baseline(baseline, affect)
                updated_baseline.user_id = self._user_id
                self._db.upsert_affect_baseline(updated_baseline)
                calibrated_affect = calibrate_affect(
                    affect,
                    updated_baseline,
                    minimum_samples=self._config.affect_calibration_min_samples,
                )

                rel = self._db.get_relationship(user_id=self._user_id)
                days_since = 0.0
                if rel.total_turns > 0:
                    days_since = max(ep.timestamp - rel.last_updated, 0.0) / 86400
                updated_rel = update_relationship(calibrated_affect, rel, days_since)
                updated_rel.user_id = self._user_id
                self._db.upsert_relationship(updated_rel)

                if self._config.auto_extract:
                    reflections = process_reflections(
                        self._db,
                        user_clean,
                        assistant_clean,
                        calibrated_affect,
                        ep.id,
                        user_id=self._user_id,
                    )

                if self._linker:
                    self._linker.link_episode_to_facts(
                        ep.id,
                        [fact.id for fact in facts if fact.id is not None],
                        user_id=self._user_id,
                    )
                    self._linker.link_episode_to_reflections(
                        ep.id,
                        [ref.id for ref in reflections if ref.id is not None],
                        user_id=self._user_id,
                    )
                    self._linker.link_episode_to_loops(
                        ep.id,
                        [loop.id for loop in resolved_loops if loop.id is not None],
                        user_id=self._user_id,
                    )
                    self._linker.link_related_episodes(ep, user_id=self._user_id)

                if self._consolidator:
                    self._consolidator.maybe_consolidate(user_id=self._user_id)

                # ── OCEAN personality scoring ────────────────────────────
                self._update_ocean(user_clean, assistant_clean)

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

    def _update_ocean(self, user_content: str, assistant_content: str) -> None:
        """Update OCEAN personality profile for the current user."""
        if not self._db:
            return
        try:
            # Get existing profile or start fresh
            existing = self._db.get_ocean_profile(self._user_id)
            if existing:
                current = self._make_ocean_score(existing)
            else:
                from .ocean import OCEANScore
                current = OCEANScore()

            # Score this turn and smooth
            scored = score_ocean_turn(user_content, assistant_content, current)

            # Persist to DB
            self._db.upsert_ocean_profile(
                user_id=self._user_id,
                openness=scored.openness,
                conscientiousness=scored.conscientiousness,
                extraversion=scored.extraversion,
                agreeableness=scored.agreeableness,
                neuroticism=scored.neuroticism,
                confidence=scored.confidence,
                turn_count=scored.turn_count,
            )
        except Exception:
            logger.exception("KORTEX _update_ocean failed")

    def _make_ocean_score(self, data: Dict[str, Any]) -> Any:
        """Reconstruct an OCEANScore from DB row."""
        from .ocean import OCEANScore
        return OCEANScore(
            openness=data.get("openness", 0.5),
            conscientiousness=data.get("conscientiousness", 0.5),
            extraversion=data.get("extraversion", 0.5),
            agreeableness=data.get("agreeableness", 0.5),
            neuroticism=data.get("neuroticism", 0.5),
            confidence=data.get("confidence", 0.5),
            turn_count=data.get("turn_count", 0),
            user_id=data.get("user_id", self._user_id),
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [KORTEX_SEARCH_SCHEMA, KORTEX_IDENTITY_SCHEMA, KORTEX_EXPORT_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name not in {"kortex_search", "kortex_identity", "kortex_export"}:
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

            if tool_name == "kortex_export":
                return self._handle_export_call(args)

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
            episodes = self._db.get_episodes_for_session(
                self._session_id, user_id=self._user_id
            )
            summary = build_conversation_summary(
                self._session_id,
                episodes,
                messages=messages,
            )
            if summary:
                summary["user_id"] = self._user_id
                self._db.insert_conversation_summary(summary)
        except Exception:
            logger.exception("KORTEX session summary generation failed")
        logger.info("KORTEX session ended with %d messages", len(messages))

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Pre-compression hook: archive middle messages and return reduced list.

        CRITICAL: Must return a REDUCED message list (or empty string to skip).
        Hermes uses this return value to replace the full context — returning
        JSON text caused the LLM compressor to still process ALL messages and
        time out. The fix: delegate to context_engine.compress() which returns
        [head + checkpoint + tail].
        """
        if not self._db:
            return ""

        try:
            from .context_engine import KortexContextEngine

            if not messages or len(messages) < 2:
                return ""

            engine = KortexContextEngine(
                db_path=self._config.db_path,
                user_id=self._user_id
            )

            result = engine.compress(
                messages=messages,
                focus_topic=self._config.focus_topic or "",
            )

            # Safety: if compress() returned the original messages list (no-op),
            # returning it would cause the LLM compressor to still process all
            # messages and time out. Skip compression instead.
            if result is messages or (isinstance(result, list) and len(result) == len(messages)):
                return ""

            return json.dumps(result)

        except Exception as e:
            logger.error("KORTEX on_pre_compress failed: %s", e)
            return ""

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._db or self._agent_context != "primary":
            return

        if target == "user":
            subject_type = "user"
        elif target == "agent":
            subject_type = "agent"
        elif target == "project":
            subject_type = "project"
        else:
            subject_type = target

        predicate = metadata.get("predicate", "hermes_memory") if metadata else "hermes_memory"
        fact_id = metadata.get("fact_id") if metadata else None

        if action == "add":
            fact = Fact(
                user_id=self._user_id,
                subject_type=subject_type,
                predicate=predicate,
                object_text=content[:500],
                confidence=metadata.get("confidence", 0.7) if metadata else 0.7,
            )
            self._db.insert_fact(fact)

        elif action == "replace":
            if fact_id:
                existing = self._db.get_fact(fact_id)
            else:
                existing = self._db.get_most_recent_fact(
                    subject_type=subject_type,
                    predicate=predicate,
                    user_id=self._user_id,
                )
            if existing:
                new_fact = Fact(
                    user_id=self._user_id,
                    subject_type=subject_type,
                    predicate=predicate,
                    object_text=content[:500],
                    confidence=metadata.get("confidence", existing.confidence) if metadata else existing.confidence,
                    source_episode_id=existing.source_episode_id,
                )
                self._db.insert_fact(new_fact)
                self._db.supersede_fact(existing.id, new_fact.id)
            else:
                new_fact = Fact(
                    user_id=self._user_id,
                    subject_type=subject_type,
                    predicate=predicate,
                    object_text=content[:500],
                    confidence=metadata.get("confidence", 0.7) if metadata else 0.7,
                )
                self._db.insert_fact(new_fact)

        elif action == "remove":
            if fact_id:
                with self._db._tx() as conn:
                    conn.execute(
                        "UPDATE facts SET status='retracted' WHERE id=?",
                        (fact_id,),
                    )
            else:
                existing = self._db.get_most_recent_fact(
                    subject_type=subject_type,
                    predicate=predicate,
                    user_id=self._user_id,
                )
                if existing:
                    with self._db._tx() as conn:
                        conn.execute(
                            "UPDATE facts SET status='retracted' WHERE id=?",
                            (existing.id,),
                        )

        logger.debug(
            "KORTEX mirrored memory write: %s %s %s metadata=%s",
            action,
            target,
            subject_type,
            metadata,
        )

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        if not self._db or self._agent_context != "primary":
            return
        try:
            ep = self._ingestor.ingest_turn(
                f"[delegated task] {task}",
                f"[delegated result] {result}",
                session_id=self._session_id,
                user_id=self._user_id,
            )
            if ep and self._linker:
                self._linker.link_episode_to_loops(
                    ep.id, [], user_id=self._user_id
                )
            logger.debug(
                "KORTEX recorded delegation (session=%s, child=%s)",
                self._session_id,
                child_session_id,
            )
        except Exception:
            logger.exception("KORTEX on_delegation failed")

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
            {
                "key": "search_format",
                "description": "Search tool response format",
                "required": False,
                "default": "narrative",
                "choices": ["narrative", "json"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        import yaml
        config_path = Path(hermes_home) / "config.yaml"
        try:
            with open(config_path) as f:
                full_config = yaml.safe_load(f) or {}
        except Exception:
            full_config = {}
        if "plugins" not in full_config:
            full_config["plugins"] = {}
        if "kortex" not in full_config["plugins"]:
            full_config["plugins"]["kortex"] = {}
        for key in ("db_path", "auto_extract", "search_format"):
            if key in values and values[key] is not None:
                full_config["plugins"]["kortex"][key] = values[key]
        with open(config_path, "w") as f:
            yaml.dump(full_config, f)
        logger.info("KORTEX config saved to %s", config_path)

    # -- Tool handlers -------------------------------------------------------

    def _handle_search(self, query: str, limit: int) -> str:
        if not query:
            return json.dumps({"error": "query required for search"})

        episodes = self._db.search_episodes(query, limit=limit, user_id=self._user_id)
        facts = self._db.search_facts(query, limit=limit, user_id=self._user_id)
        reflections = self._db.search_reflections(
            query, limit=limit, user_id=self._user_id
        )

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

        if self._config.search_format == "json":
            return json.dumps(results)

        return self._format_search_narrative(query, episodes, facts, reflections)

    def _format_search_narrative(
        self,
        query: str,
        episodes: List[Episode],
        facts: List[Fact],
        reflections: List[Any],
    ) -> str:
        if not episodes and not facts and not reflections:
            return f"I couldn't recall anything relevant to '{query}'."

        lines = [
            f"Here's what I remember about '{query}':",
        ]

        recent_episodes = []
        older_episodes = []
        now = time.time()
        for episode in episodes:
            age_days = max((now - episode.timestamp) / 86400, 0.0)
            if age_days <= 14:
                recent_episodes.append(episode)
            else:
                older_episodes.append(episode)

        if recent_episodes:
            lines.append("Recent memories:")
            for episode in recent_episodes:
                lines.append(
                    "- "
                    f"{episode.to_recall_text(now)} "
                    f"(source: episode #{episode.id} on {epoch_to_display(episode.timestamp)})"
                )

        if older_episodes:
            lines.append("Older related memories:")
            for episode in older_episodes:
                lines.append(
                    "- "
                    f"{episode.to_recall_text(now)} "
                    f"(source: episode #{episode.id} on {epoch_to_display(episode.timestamp)})"
                )

        if facts:
            lines.append("Durable facts:")
            for fact in facts:
                predicate = f"[{fact.predicate}] " if fact.predicate else ""
                source = (
                    f"source episode #{fact.source_episode_id}"
                    if fact.source_episode_id
                    else "no direct episode source"
                )
                lines.append(
                    "- "
                    f"I'm {self._confidence_phrase(fact.confidence)} that {predicate}{fact.object_text} "
                    f"(fact #{fact.id}, {source})"
                )

        if reflections:
            lines.append("Learned patterns:")
            for reflection in reflections:
                source = (
                    f"source episode #{reflection.source_episode_id}"
                    if reflection.source_episode_id
                    else "no direct episode source"
                )
                lines.append(
                    "- "
                    f"I'm {self._confidence_phrase(reflection.confidence)} this pattern matters: "
                    f"{reflection.text} ({reflection.kind}, reflection #{reflection.id}, {source})"
                )

        return "\n".join(lines)

    @staticmethod
    def _confidence_phrase(confidence: float) -> str:
        if confidence >= 0.8:
            return "very confident"
        if confidence >= 0.6:
            return "fairly certain"
        if confidence >= 0.4:
            return "reasonably confident"
        return "only loosely confident"

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
        facts = self._db.get_active_facts(limit=limit, user_id=self._user_id)
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
        loops = self._db.get_open_loops(limit=limit, user_id=self._user_id)
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
        summaries = self._db.list_conversation_summaries(
            limit=limit, user_id=self._user_id
        )
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
        total_episodes = self._db.count_episodes(user_id=self._user_id)
        unconsolidated_raw_episodes = self._db.count_unconsolidated_episodes(
            user_id=self._user_id
        )
        facts = self._db.get_active_facts(limit=1000, user_id=self._user_id)
        loops = self._db.get_open_loops(limit=1000, user_id=self._user_id)
        reflections = self._db.get_reflections(limit=1000, user_id=self._user_id)
        rel = self._db.get_relationship(user_id=self._user_id)
        recent_emotions = self._db.get_recent_emotions(limit=5, user_id=self._user_id)
        baseline = self._db.get_affect_baseline(user_id=self._user_id)

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

    def _handle_export_call(self, args: Dict[str, Any]) -> str:
        if not self._db:
            return json.dumps({"error": "KORTEX not initialized"})

        action = args.get("action", "")
        target_user = args.get("user_id", self._user_id)

        if action == "export":
            return export_to_json(
                self._db,
                user_id=target_user,
                start=args.get("start"),
                end=args.get("end"),
                memory_types=args.get("types"),
            )

        if action == "import":
            payload = args.get("payload")
            if not payload:
                return json.dumps({"error": "payload required for import"})
            return json.dumps(
                import_from_json(
                    self._db,
                    payload,
                    allow_override=bool(args.get("allow_override", False)),
                )
            )

        return json.dumps({"error": f"Unknown export action: {action}"})

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
