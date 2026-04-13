"""Ranked fusion retrieval and budget-aware context packing for KORTEX.

Single query in, one merged context block out. Ranks by:
  relevance x salience x recency_decay x emotional_weight

Packs into a hard token budget with quota-based allocation per section.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone
from typing import List, Optional

from .config import KortexConfig
from .db import KortexDB
from .models import AffectSignal, Episode, Fact, OpenLoop, Reflection, RelationshipState

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


class Recall:
    """Retrieves and packs memory context for injection."""

    def __init__(self, db: KortexDB, config: KortexConfig):
        self._db = db
        self._config = config

    def build_context(self, query: str, session_id: str = "") -> str:
        sections = []
        budget_used = 0

        relationship = self._db.get_relationship()
        rel_text = relationship.to_compact_text()
        emotional_trajectory = self._build_emotional_trajectory(session_id)
        if emotional_trajectory:
            rel_text = f"{rel_text}\n{emotional_trajectory}"
        rel_budget = self._config.budget.get("relationship_state", 200)
        rel_text = self._trim_to_budget(rel_text, rel_budget)
        if relationship.total_turns > 0:
            sections.append(rel_text)
            budget_used += self._estimate_tokens(rel_text)

        facts_budget = self._config.budget.get("stable_facts", 400)
        facts_text = self._build_facts_section(query, facts_budget)
        if facts_text:
            sections.append(facts_text)
            budget_used += self._estimate_tokens(facts_text)

        episodes_budget = self._config.budget.get("episodic_memories", 800)
        episodes_text = self._build_episodes_section(query, session_id, episodes_budget)
        if episodes_text:
            sections.append(episodes_text)
            budget_used += self._estimate_tokens(episodes_text)

        loops_budget = self._config.budget.get("open_loops", 200)
        loops_text = self._build_loops_section(loops_budget)
        if loops_text:
            sections.append(loops_text)
            budget_used += self._estimate_tokens(loops_text)

        reflections_budget = self._config.budget.get("reflections", 200)
        reflections_text = self._build_reflections_section(reflections_budget)
        if reflections_text:
            sections.append(reflections_text)
            budget_used += self._estimate_tokens(reflections_text)

        if not sections:
            return ""

        header = "[KORTEX Memory]"
        body = "\n".join(sections)
        full = f"{header}\n{body}"

        if self._estimate_tokens(full) > self._config.total_budget:
            full = self._trim_to_budget(full, self._config.total_budget)

        return full

    def _build_facts_section(self, query: str, budget: int) -> str:
        facts: List[Fact] = []

        if query:
            facts = self._db.search_facts(
                query, limit=self._config.max_facts_per_recall
            )

        if len(facts) < self._config.max_facts_per_recall:
            top_facts = self._db.get_active_facts(
                subject_type="user",
                limit=self._config.max_facts_per_recall - len(facts),
            )
            seen_ids = {f.id for f in facts}
            facts.extend(f for f in top_facts if f.id not in seen_ids)

        if not facts:
            return ""

        lines = ["Known facts:"]
        for f in facts[: self._config.max_facts_per_recall]:
            line = f"- {f.object_text}"
            if f.predicate:
                line = f"- [{f.predicate}] {f.object_text}"
            lines.append(line)

        text = "\n".join(lines)
        return self._trim_to_budget(text, budget)

    def _build_episodes_section(self, query: str, session_id: str, budget: int) -> str:
        now = datetime.now(timezone.utc)
        candidates: List[Episode] = []

        if query:
            search_results = self._db.search_episodes(query, limit=10)
            candidates.extend(search_results)

        salient = self._db.get_salient_episodes(
            min_salience=self._config.salience_threshold,
            limit=10,
        )
        seen_ids = {e.id for e in candidates}
        candidates.extend(e for e in salient if e.id not in seen_ids)

        recent = self._db.get_recent_episodes(limit=5)
        seen_ids = {e.id for e in candidates}
        candidates.extend(e for e in recent if e.id not in seen_ids)

        scored = []
        for ep in candidates:
            score = self._rank_episode(ep, query, now)
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._config.max_episodes_per_recall]

        if not top:
            return ""

        lines = ["Recalled memories:"]
        for _score, ep in top:
            lines.append(f"- {ep.to_recall_text(now)}")

        text = "\n".join(lines)
        return self._trim_to_budget(text, budget)

    def _build_loops_section(self, budget: int) -> str:
        loops = self._db.get_open_loops(limit=self._config.max_loops_per_recall)
        if not loops:
            return ""

        lines = ["Open threads:"]
        for loop in loops:
            kind_label = loop.kind.replace("_", " ")
            lines.append(f"- [{kind_label}] {loop.text}")

        text = "\n".join(lines)
        return self._trim_to_budget(text, budget)

    def _build_reflections_section(self, budget: int) -> str:
        reflections = self._db.get_high_confidence_reflections(
            min_confidence=self._config.reflection_confidence_threshold,
            limit=self._config.max_reflections_per_recall,
        )
        if not reflections:
            return ""

        _KIND_LABELS = {
            "mistake": "Avoid",
            "pattern": "Works well",
            "preference": "User prefers",
            "style": "Style note",
        }

        lines = ["Learned behaviors:"]
        for ref in reflections:
            label = _KIND_LABELS.get(ref.kind, ref.kind.title())
            reinforced = (
                f" (x{ref.reinforcement_count})" if ref.reinforcement_count > 1 else ""
            )
            lines.append(f"- [{label}]{reinforced} {ref.text}")

        text = "\n".join(lines)
        return self._trim_to_budget(text, budget)

    def _build_emotional_trajectory(self, session_id: str = "") -> str:
        trajectory = self._db.get_emotional_trajectory(
            limit=5, session_id=session_id or None
        )
        if not trajectory:
            return ""

        recent_emotions = [
            entry["emotion"] for entry in trajectory if entry["emotion"] != "neutral"
        ]
        if not recent_emotions:
            return ""

        recent_valences = [entry["valence"] for entry in trajectory]
        avg_valence = sum(recent_valences) / len(recent_valences)

        trend = "neutral"
        if len(recent_valences) >= 2:
            first_half = recent_valences[len(recent_valences) // 2 :]
            second_half = recent_valences[: len(recent_valences) // 2]
            if first_half and second_half:
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                if second_avg - first_avg > 0.15:
                    trend = "improving"
                elif first_avg - second_avg > 0.15:
                    trend = "declining"

        unique_emotions = []
        seen = set()
        for e in recent_emotions[:3]:
            if e not in seen:
                unique_emotions.append(e)
                seen.add(e)

        parts = [f"Recent mood: {', '.join(unique_emotions)}"]
        if trend != "neutral":
            parts.append(f"(trend: {trend})")

        return " ".join(parts)

    def _rank_episode(self, ep: Episode, query: str, now: datetime) -> float:
        # Recency: exponential decay with configurable half-life
        age_days = max((now - ep.timestamp).total_seconds() / 86400, 0.01)
        half_life = self._config.recency_decay_days
        recency = math.exp(-0.693 * age_days / half_life)  # ln(2) ≈ 0.693

        salience = ep.salience
        emotional = ep.emotional_weight

        # Base relevance (FTS already ranked, so search results get a bonus)
        relevance = 0.5
        if query and ep.summary:
            query_words = set(query.lower().split())
            summary_words = set(ep.summary.lower().split())
            overlap = len(query_words & summary_words)
            if overlap > 0:
                relevance = min(0.5 + overlap * 0.15, 1.0)

        return relevance * 0.3 + salience * 0.25 + recency * 0.25 + emotional * 0.2

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)

    @staticmethod
    def _trim_to_budget(text: str, budget_tokens: int) -> str:
        max_chars = budget_tokens * CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."
