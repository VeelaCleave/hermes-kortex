"""Promotion workflows for applying identity deltas to SOUL.md."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .db import KortexDB
from .models import IdentityDelta

_MAX_TRAIT_LENGTH = 500
_LEARNED_TRAITS_HEADER = "## Learned Traits"


class Promoter:
    def __init__(self, db: KortexDB, soul_path: Optional[str] = None):
        """
        soul_path: explicit path to SOUL.md, or None to use default (~/.hermes/SOUL.md)
        """
        self._db = db
        self._soul_path = soul_path
        self._lock = threading.Lock()

    def list_pending(self, limit: int = 20) -> List[IdentityDelta]:
        """Return unapplied identity deltas sorted by confidence desc."""
        deltas = self._db.get_identity_deltas(
            applied=False, limit=max(limit * 5, limit, 1)
        )
        return sorted(
            deltas, key=lambda delta: (-delta.confidence, -int(delta.id or 0))
        )[:limit]

    def preview_delta(self, delta_id: int) -> dict:
        """Return preview info with optional source episode context."""
        delta = self._db.get_identity_delta_by_id(delta_id)
        if not delta:
            return {"error": f"Identity delta {delta_id} not found"}

        preview = {
            "id": delta.id,
            "text": self._truncate_text(delta.text),
            "confidence": delta.confidence,
            "created_at": delta.created_at.isoformat(),
            "applied": delta.applied,
            "source_episode_id": delta.source_episode_id,
        }

        if delta.source_episode_id:
            episode = self._db.get_episode(delta.source_episode_id)
            if episode:
                preview["source_episode"] = {
                    "id": episode.id,
                    "summary": episode.summary,
                    "timestamp": episode.timestamp_iso,
                    "user_text": episode.user_text[:500],
                    "assistant_text": episode.assistant_text[:500],
                }

        return preview

    def approve_and_apply(self, delta_id: int) -> dict:
        """Apply a delta to SOUL.md and mark it as applied in DB."""
        with self._lock:
            delta = self._db.get_identity_delta_by_id(delta_id)
            if not delta:
                return {"error": f"Identity delta {delta_id} not found"}
            if delta.applied:
                return {"error": f"Identity delta {delta_id} already applied"}

            soul_path = self._resolve_soul_path()
            soul_content = self.get_soul_content()
            applied_text = self._format_trait(delta.text, delta.confidence)
            updated_content = self._append_trait(soul_content, applied_text)

            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text(updated_content, encoding="utf-8")
            self._db.mark_identity_delta_applied(delta_id)

            return {
                "success": True,
                "delta_id": delta_id,
                "applied_text": applied_text,
                "soul_path": str(soul_path),
            }

    def reject_delta(self, delta_id: int) -> dict:
        """Delete a delta from the DB."""
        if not self._db.reject_identity_delta(delta_id):
            return {"error": f"Identity delta {delta_id} not found"}
        return {"success": True, "rejected_id": delta_id}

    def approve_multiple(self, delta_ids: List[int]) -> dict:
        """Apply multiple deltas at once. Returns summary of results."""
        if not delta_ids:
            return {
                "success": True,
                "applied": [],
                "failed": [],
                "requested": 0,
                "applied_count": 0,
            }

        applied = []
        failed = []
        for delta_id in delta_ids:
            result = self.approve_and_apply(delta_id)
            if result.get("success"):
                applied.append(result)
            else:
                failed.append(
                    {
                        "delta_id": delta_id,
                        "error": result.get("error", "unknown error"),
                    }
                )

        return {
            "success": len(failed) == 0,
            "applied": applied,
            "failed": failed,
            "requested": len(delta_ids),
            "applied_count": len(applied),
        }

    def get_soul_content(self) -> str:
        """Read current SOUL.md content. Returns empty string if file doesn't exist."""
        soul_path = self._resolve_soul_path()
        if not soul_path.exists():
            return ""
        return soul_path.read_text(encoding="utf-8")

    def _resolve_soul_path(self) -> Path:
        """Resolve SOUL.md path: config override > HERMES_HOME/SOUL.md > ~/.hermes/SOUL.md."""
        if self._soul_path:
            return Path(self._soul_path).expanduser()

        hermes_home = os.environ.get("HERMES_HOME")
        if hermes_home:
            return Path(hermes_home).expanduser() / "SOUL.md"

        return Path.home() / ".hermes" / "SOUL.md"

    @staticmethod
    def _truncate_text(text: str) -> str:
        text = text.strip()
        return text[:_MAX_TRAIT_LENGTH]

    @classmethod
    def _format_trait(cls, text: str, confidence: float) -> str:
        return f"- {cls._truncate_text(text)} [confidence: {confidence:.2f}]"

    @classmethod
    def _append_trait(cls, content: str, trait_line: str) -> str:
        content = content or ""
        if _LEARNED_TRAITS_HEADER not in content:
            base = content.rstrip("\n")
            if base:
                return f"{base}\n\n{_LEARNED_TRAITS_HEADER}\n{trait_line}\n"
            return f"{_LEARNED_TRAITS_HEADER}\n{trait_line}\n"

        lines = content.splitlines()
        header_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip() == _LEARNED_TRAITS_HEADER
            ),
            None,
        )
        if header_index is None:
            base = content.rstrip("\n")
            return f"{base}\n\n{_LEARNED_TRAITS_HEADER}\n{trait_line}\n"

        insert_index = len(lines)
        for index in range(header_index + 1, len(lines)):
            if lines[index].startswith("## "):
                insert_index = index
                break

        lines.insert(insert_index, trait_line)
        return "\n".join(lines).rstrip("\n") + "\n"
