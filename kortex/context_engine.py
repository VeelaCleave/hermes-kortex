"""KORTEX lossless context engine for Hermes.

Archives exact dropped transcript spans at compression time, emits a bounded
deterministic checkpoint message, and exposes engine-owned retrieval tools for
later bounded re-expansion.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any, Dict, List, Optional

from .db import DEFAULT_USER_ID, KortexDB

try:
    from agent.context_engine import ContextEngine as HermesContextEngine
except ImportError:

    class HermesContextEngine:
        last_prompt_tokens: int = 0
        last_completion_tokens: int = 0
        last_total_tokens: int = 0
        threshold_tokens: int = 0
        context_length: int = 0
        compression_count: int = 0
        threshold_percent: float = 0.50
        protect_first_n: int = 3
        protect_last_n: int = 20

        def on_session_reset(self) -> None:
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            self.last_total_tokens = 0
            self.compression_count = 0

        def get_tool_schemas(self):
            return []

        def handle_tool_call(self, name, args, **kwargs):
            import json
            return json.dumps({"error": f"Unknown context engine tool: {name}"})

        def get_status(self):
            usage_percent = (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length
                else 0
            )
            return {
                "last_prompt_tokens": self.last_prompt_tokens,
                "threshold_tokens": self.threshold_tokens,
                "context_length": self.context_length,
                "usage_percent": usage_percent,
                "compression_count": self.compression_count,
            }

        def update_model(self, model, context_length, base_url="", api_key="", provider=""):
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self.threshold_percent)


CHECKPOINT_PREFIX = "[KORTEX CHECKPOINT — REFERENCE ONLY]"
LOSSY_MARKERS = ("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]:")

KORTEX_RECALL_SCHEMA = {
    "name": "kortex_recall",
    "description": "Search archived KORTEX lossless conversation history and checkpoint refs.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results", "default": 5},
        },
        "required": ["query"],
    },
}

KORTEX_EXPAND_SCHEMA = {
    "name": "kortex_expand",
    "description": "Expand exact archived KORTEX messages for a checkpoint ref or sequence range.",
    "parameters": {
        "type": "object",
        "properties": {
            "ref_id": {
                "type": "string",
                "description": "Reference id from kortex_recall",
            },
            "start_seq": {"type": "integer", "description": "Optional start sequence"},
            "end_seq": {"type": "integer", "description": "Optional end sequence"},
            "limit": {"type": "integer", "description": "Max messages", "default": 8},
        },
    },
}


class KortexContextEngine(HermesContextEngine):
    threshold_percent: float = 0.50
    protect_first_n: int = 3
    protect_last_n: int = 20

    def __init__(self, db_path: Optional[str] = None, user_id: str = DEFAULT_USER_ID):
        self._db_path = db_path
        self._db: Optional[KortexDB] = None
        self._session_id: str = ""
        self._conversation_id: str = ""
        self._parent_session_id: str = ""
        self._user_id: str = user_id
        self._model: str = "unknown"

    @property
    def name(self) -> str:
        return "kortex"

    def is_available(self) -> bool:
        return True

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._parent_session_id = str(kwargs.get("parent_session_id") or "")
        self._user_id = str(kwargs.get("user_id") or self._user_id or DEFAULT_USER_ID)
        self._model = str(kwargs.get("model") or self._model or "unknown")
        hermes_home = str(kwargs.get("hermes_home") or "")
        db_path = self._db_path or (
            f"{hermes_home.rstrip('/')}/kortex.db" if hermes_home else None
        )
        if db_path:
            self._db = KortexDB(db_path)

        self._conversation_id = self._derive_conversation_id(session_id)
        if self._db:
            self._db.ensure_context_conversation(
                self._conversation_id, user_id=self._user_id
            )
            self._db.map_session_alias(
                session_id,
                self._conversation_id,
                lineage_parent_session_id=self._parent_session_id or None,
            )

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if self._db:
            self._db.map_session_alias(
                session_id,
                self._conversation_id,
                lineage_parent_session_id=self._parent_session_id or None,
            )

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._session_id = ""
        self._parent_session_id = ""

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return bool(self.threshold_tokens and tokens >= self.threshold_tokens)

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.compression_count += 1
        if (
            not messages
            or not self._db
            or len(messages) <= (self.protect_first_n + self.protect_last_n + 1)
        ):
            return messages

        head = messages[: self.protect_first_n]
        tail = messages[-self.protect_last_n :]
        middle = messages[self.protect_first_n : len(messages) - self.protect_last_n]
        if not middle:
            return messages

        if any(self._contains_lossy_marker(msg) for msg in middle):
            start_seq = self._db.get_next_context_seq(self._conversation_id)
            span_id = self._db.create_context_span(
                self._conversation_id,
                start_seq=start_seq,
                end_seq=start_seq,
                kind="lossy_boundary",
            )
            self._db.insert_context_ref(
                self._conversation_id,
                ref_id=f"lossy_{span_id}",
                ref_type="lossy_boundary",
                label="Earlier context was already lossy before KORTEX lossless engine activation.",
                payload={"focus_topic": focus_topic or ""},
                source_span_id=span_id,
                salience=1.0,
                open_state="closed",
            )

        start_seq, end_seq = self._db.archive_context_messages(
            self._conversation_id, middle
        )
        span_id = self._db.create_context_span(
            self._conversation_id,
            start_seq=start_seq,
            end_seq=end_seq,
            kind="compressed",
        )
        refs = self._extract_refs(
            middle, source_span_id=span_id, focus_topic=focus_topic
        )
        checkpoint_id = self._checkpoint_id(start_seq, end_seq, middle)
        checkpoint_text = self._build_checkpoint_text(
            checkpoint_id,
            start_seq,
            end_seq,
            refs,
            focus_topic=focus_topic,
        )
        hot_ref_ids = [ref["ref_id"] for ref in refs[:6]]
        self._db.insert_context_checkpoint(
            self._conversation_id,
            checkpoint_id=checkpoint_id,
            replaced_start_seq=start_seq,
            replaced_end_seq=end_seq,
            export_text=checkpoint_text,
            source_span_ids=[span_id],
            hot_ref_ids=hot_ref_ids,
        )
        return head + [{"role": "assistant", "content": checkpoint_text}] + tail

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [KORTEX_RECALL_SCHEMA, KORTEX_EXPAND_SCHEMA]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._db:
            return json.dumps({"error": "KORTEX context engine not initialized"})
        if name == "kortex_recall":
            return self._handle_recall(args)
        if name == "kortex_expand":
            return self._handle_expand(args)
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update(
            {
                "engine": self.name,
                "conversation_id": self._conversation_id,
                "session_id": self._session_id,
            }
        )
        return status

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        self._model = model
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

    def _derive_conversation_id(self, session_id: str) -> str:
        if self._db:
            existing = self._db.get_context_conversation_id(session_id)
            if existing:
                return existing
            if self._parent_session_id:
                parent = self._db.get_context_conversation_id(self._parent_session_id)
                if parent:
                    return parent
        seed = self._parent_session_id or session_id
        digest = sha256(f"{self._user_id}:{seed}".encode("utf-8")).hexdigest()[:16]
        return f"conv_{digest}"

    @staticmethod
    def _contains_lossy_marker(message: Dict[str, Any]) -> bool:
        content = message.get("content", "")
        if not isinstance(content, str):
            return False
        return any(marker in content for marker in LOSSY_MARKERS)

    def _extract_refs(
        self,
        messages: List[Dict[str, Any]],
        *,
        source_span_id: int,
        focus_topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        text_blob = "\n".join(self._message_text_content(msg) for msg in messages)
        tasks = re.findall(
            r"\b(?:todo|fix|implement|investigate|follow up|remember)\b.*",
            text_blob,
            flags=re.IGNORECASE,
        )
        decisions = re.findall(
            r"\b(?:decided|decision|chosen|keep|use|switch to)\b.*",
            text_blob,
            flags=re.IGNORECASE,
        )
        files = sorted(set(re.findall(r"(?:[\w.-]+/)+[\w.-]+", text_blob)))
        errors = re.findall(
            r"\b(?:error|failed|exception|traceback)\b.*",
            text_blob,
            flags=re.IGNORECASE,
        )

        for idx, item in enumerate(tasks[:4]):
            refs.append(self._make_ref("task", item.strip(), source_span_id, idx, 0.9))
        for idx, item in enumerate(decisions[:4], start=len(refs)):
            refs.append(
                self._make_ref("decision", item.strip(), source_span_id, idx, 0.85)
            )
        for idx, item in enumerate(files[:4], start=len(refs)):
            refs.append(self._make_ref("file", item.strip(), source_span_id, idx, 0.75))
        for idx, item in enumerate(errors[:3], start=len(refs)):
            refs.append(self._make_ref("error", item.strip(), source_span_id, idx, 0.8))

        if focus_topic:
            refs.insert(
                0,
                self._make_ref("focus", focus_topic.strip(), source_span_id, 999, 1.0),
            )

        if not refs:
            refs.append(
                self._make_ref(
                    "thread", "Archived conversation span", source_span_id, 0, 0.5
                )
            )

        for ref in refs:
            self._db.insert_context_ref(
                self._conversation_id,
                ref_id=ref["ref_id"],
                ref_type=ref["ref_type"],
                label=ref["label"],
                payload=ref["payload"],
                source_span_id=source_span_id,
                salience=ref["salience"],
                open_state=ref["open_state"],
            )
        return refs

    def _make_ref(
        self,
        ref_type: str,
        label: str,
        source_span_id: int,
        ordinal: int,
        salience: float,
    ) -> Dict[str, Any]:
        digest = sha256(
            f"{ref_type}:{label}:{source_span_id}:{ordinal}".encode("utf-8")
        ).hexdigest()[:10]
        return {
            "ref_id": f"ref_{digest}",
            "ref_type": ref_type,
            "label": label,
            "payload": {"label": label, "source_span_id": source_span_id},
            "salience": salience,
            "open_state": "open"
            if ref_type in {"task", "error", "focus", "thread"}
            else "closed",
        }

    def _checkpoint_id(
        self, start_seq: int, end_seq: int, middle: List[Dict[str, Any]]
    ) -> str:
        digest = sha256(
            json.dumps(middle, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return f"ckpt_{start_seq}_{end_seq}_{digest}"

    def _build_checkpoint_text(
        self,
        checkpoint_id: str,
        start_seq: int,
        end_seq: int,
        refs: List[Dict[str, Any]],
        *,
        focus_topic: Optional[str] = None,
    ) -> str:
        open_refs = [ref for ref in refs if ref["open_state"] == "open"]
        decisions = [ref for ref in refs if ref["ref_type"] == "decision"]
        files = [ref for ref in refs if ref["ref_type"] == "file"]
        hot_refs = [ref["ref_id"] for ref in refs[:6]]

        lines = [
            CHECKPOINT_PREFIX,
            f"checkpoint_id: {checkpoint_id}",
            f"conversation_id: {self._conversation_id}",
            f"replaced_span: seq {start_seq}..{end_seq}",
            "available_tools: kortex_recall, kortex_expand",
            "",
            "## Open Threads",
        ]
        lines.extend(f"- {ref['label']} ({ref['ref_id']})" for ref in open_refs[:5])
        if not open_refs:
            lines.append("- none")

        lines.append("")
        lines.append("## Key Decisions")
        lines.extend(f"- {ref['label']}" for ref in decisions[:5])
        if not decisions:
            lines.append("- none")

        lines.append("")
        lines.append("## Critical Artifacts")
        lines.extend(f"- {ref['label']}" for ref in files[:5])
        if not files:
            lines.append("- none")

        if focus_topic:
            lines.extend(["", "## Focus Topic", f"- {focus_topic}"])

        lines.extend(["", "## Expandable References"])
        lines.extend(f"- {ref_id}" for ref_id in hot_refs)
        return "\n".join(lines)

    def _handle_recall(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        limit = max(1, int(args.get("limit", 5)))
        if not query:
            return json.dumps({"error": "query is required"})

        refs = self._db.search_context_refs(self._conversation_id, query, limit=limit)
        msgs = self._db.search_context_messages(
            self._conversation_id, query, limit=limit
        )
        return json.dumps(
            {
                "query": query,
                "conversation_id": self._conversation_id,
                "refs": [
                    {
                        "ref_id": ref["ref_id"],
                        "ref_type": ref["ref_type"],
                        "label": ref["label"],
                        "salience": ref["salience"],
                        "open_state": ref["open_state"],
                    }
                    for ref in refs
                ],
                "message_hits": [
                    {
                        "seq": row["seq"],
                        "role": row["role"],
                        "snippet": row["text_content"][:300],
                    }
                    for row in msgs
                ],
            }
        )

    def _handle_expand(self, args: Dict[str, Any]) -> str:
        limit = max(1, int(args.get("limit", 8)))
        ref_id = str(args.get("ref_id") or "").strip()
        start_seq = args.get("start_seq")
        end_seq = args.get("end_seq")

        if ref_id:
            ref_row = (
                self._db._get_conn()
                .execute(
                    "SELECT * FROM context_refs WHERE ref_id=? AND conversation_id=?",
                    (ref_id, self._conversation_id),
                )
                .fetchone()
            )
            if not ref_row:
                return json.dumps({"error": f"Unknown ref_id: {ref_id}"})
            payload = json.loads(ref_row["payload_json"] or "{}")
            span_id = payload.get("source_span_id")
            if span_id is None:
                return json.dumps({"error": f"ref_id {ref_id} has no source span"})
            span_rows = (
                self._db._get_conn()
                .execute(
                    "SELECT * FROM context_spans WHERE id=? AND conversation_id=?",
                    (span_id, self._conversation_id),
                )
                .fetchall()
            )
            if not span_rows:
                return json.dumps({"error": f"No span found for ref_id: {ref_id}"})
            start_seq = span_rows[0]["start_seq"]
            end_seq = span_rows[0]["end_seq"]

        if start_seq is None or end_seq is None:
            checkpoint = self._db.get_active_context_checkpoint(self._conversation_id)
            if not checkpoint:
                return json.dumps({"error": "No active checkpoint available"})
            start_seq = checkpoint["replaced_start_seq"]
            end_seq = checkpoint["replaced_end_seq"]

        rows = self._db.get_context_messages_by_seq_range(
            self._conversation_id, int(start_seq), int(end_seq)
        )[:limit]
        return json.dumps(
            {
                "conversation_id": self._conversation_id,
                "range": {"start_seq": int(start_seq), "end_seq": int(end_seq)},
                "messages": [json.loads(row["raw_json"]) for row in rows],
            }
        )

    def expand_ref(self, ref_id: str, limit: int = 8) -> Dict[str, Any]:
        return json.loads(self._handle_expand({"ref_id": ref_id, "limit": limit}))

    @staticmethod
    def _message_text_content(message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str) and text:
                        chunks.append(text)
            return "\n".join(chunks)
        return str(content or "")
