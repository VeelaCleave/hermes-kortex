"""KORTEX plugin configuration.

Reads from Hermes config.yaml under plugins.kortex, with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# Budget allocation for context injection (~1200 tokens total)
# Reduced from ~2000 to avoid bloating context for MOE models.
# Each section is leaner to prevent redundant parsing across experts.
DEFAULT_BUDGET = {
    "relationship_state": 150,  # 100-150 tokens
    "conversation_summaries": 80,
    "stable_facts": 200,  # 150-250 tokens
    "episodic_memories": 300,  # reserve space for graph-expanded recall
    "graph_memories": 100,
    "open_loops": 150,  # 100-200 tokens
    "reflections": 150,  # 100-200 tokens
    "reserve": 100,  # 100-150 tokens
}


@dataclass
class KortexConfig:
    """Configuration for the KORTEX memory plugin."""

    # Database
    db_path: Optional[str] = None  # None = auto (hermes_home / "kortex.db")

    # Episodic memory
    max_episodes_per_recall: int = 3
    max_conversation_summaries_per_recall: int = 2
    salience_threshold: float = 0.2
    recency_decay_days: float = 30.0  # half-life in days for recency scoring
    consolidation_threshold: int = 200
    consolidation_batch_size: int = 100
    same_session_recency_boost: float = 2.0
    temporal_query_boost: float = 1.4
    episode_decay_rate: float = 0.10
    fact_decay_rate: float = 0.05
    reflection_decay_rate: float = 0.08
    cold_memory_threshold: float = 0.1
    warm_memory_threshold: float = 0.3
    graph_max_hops: int = 2
    graph_decay_factor: float = 0.5
    graph_expansion_limit: int = 6

    # Facts
    max_facts_per_recall: int = 4
    fact_confidence_threshold: float = 0.3

    # Open loops
    max_loops_per_recall: int = 2

    # Reflections
    max_reflections_per_recall: int = 2
    reflection_confidence_threshold: float = 0.4

    # Context budget (tokens per section)
    budget: Dict[str, int] = field(default_factory=lambda: DEFAULT_BUDGET.copy())

    # Total hard cap
    total_budget: int = 1230

    # Passive recall / context-engine integration
    passive_recall: bool = True
    prefer_passive_recall: bool = True
    context_engine_enabled: bool = True
    passive_context_hint: bool = True

    # Extraction
    auto_extract: bool = True  # extract facts/loops from turns automatically
    affect_calibration_min_samples: int = 20
    extraction_mode: str = "heuristic"
    search_format: str = "narrative"

    # Stale detection for context status system
    stale_detection_enabled: bool = True
    stale_loop_days: float = 14.0  # Days before an open loop is considered "stale"
    stale_fact_days: float = 30.0  # Days before a fact is considered "stale"
    
    # Context status display
    show_context_status: bool = True
    show_completion_markers: bool = True
    recent_resolution_window_days: float = 7.0  # How long to show resolved loops in context
    
    # Lightweight context mode (skip graph traversal, enrichment, etc.)
    # Use this to speed up context injection during compaction/normal turns
    lightweight_context: bool = True
    
    # Identity
    soul_path: Optional[str] = None  # custom SOUL.md path (None = hermes default)

    # Focus topic (optional, used for targeted context injection)
    focus_topic: Optional[str] = None

    # LLM extraction — use a local vLLM/Ollama endpoint for fact/loop extraction
    extraction_llm_base_url: Optional[str] = None  # e.g. "http://127.0.0.1:8000/v1"
    extraction_llm_model: Optional[str] = None  # e.g. "aeon-backend"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KortexConfig":
        """Create config from a dict (e.g. from YAML)."""
        budget = data.get("budget", DEFAULT_BUDGET.copy())
        # Merge with defaults so missing keys don't break
        merged_budget = DEFAULT_BUDGET.copy()
        merged_budget.update(budget)

        return cls(
            db_path=data.get("db_path"),
            max_episodes_per_recall=data.get("max_episodes_per_recall", 3),
            max_conversation_summaries_per_recall=data.get(
                "max_conversation_summaries_per_recall", 2
            ),
            salience_threshold=data.get("salience_threshold", 0.2),
            recency_decay_days=data.get("recency_decay_days", 30.0),
            consolidation_threshold=data.get("consolidation_threshold", 200),
            consolidation_batch_size=data.get("consolidation_batch_size", 100),
            same_session_recency_boost=data.get("same_session_recency_boost", 2.0),
            temporal_query_boost=data.get("temporal_query_boost", 1.4),
            episode_decay_rate=data.get("episode_decay_rate", 0.10),
            fact_decay_rate=data.get("fact_decay_rate", 0.05),
            reflection_decay_rate=data.get("reflection_decay_rate", 0.08),
            cold_memory_threshold=data.get("cold_memory_threshold", 0.1),
            warm_memory_threshold=data.get("warm_memory_threshold", 0.3),
            graph_max_hops=data.get("graph_max_hops", 2),
            graph_decay_factor=data.get("graph_decay_factor", 0.5),
            graph_expansion_limit=data.get("graph_expansion_limit", 6),
            max_facts_per_recall=data.get("max_facts_per_recall", 4),
            fact_confidence_threshold=data.get("fact_confidence_threshold", 0.3),
            max_loops_per_recall=data.get("max_loops_per_recall", 2),
            max_reflections_per_recall=data.get("max_reflections_per_recall", 2),
            reflection_confidence_threshold=data.get(
                "reflection_confidence_threshold", 0.4
            ),
            budget=merged_budget,
            total_budget=data.get("total_budget", 1230),
            passive_recall=data.get("passive_recall", True),
            prefer_passive_recall=data.get("prefer_passive_recall", True),
            context_engine_enabled=data.get("context_engine_enabled", True),
            passive_context_hint=data.get("passive_context_hint", True),
            auto_extract=data.get("auto_extract", True),
            affect_calibration_min_samples=data.get(
                "affect_calibration_min_samples", 20
            ),
            extraction_mode=data.get("extraction_mode", "heuristic"),
            search_format=data.get("search_format", "narrative"),
            soul_path=data.get("soul_path"),
            stale_detection_enabled=data.get("stale_detection_enabled", True),
            stale_loop_days=data.get("stale_loop_days", 14.0),
            stale_fact_days=data.get("stale_fact_days", 30.0),
            show_context_status=data.get("show_context_status", True),
            show_completion_markers=data.get("show_completion_markers", True),
            recent_resolution_window_days=data.get("recent_resolution_window_days", 7.0),
            lightweight_context=data.get("lightweight_context", True),
            focus_topic=data.get("focus_topic"),
            extraction_llm_base_url=data.get("extraction_llm_base_url"),
            extraction_llm_model=data.get("extraction_llm_model"),
        )


def load_kortex_config(hermes_home: Optional[str] = None) -> KortexConfig:
    """Load KORTEX config from Hermes config.yaml."""
    if hermes_home is None:
        import os

        hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))

    config_path = Path(hermes_home) / "config.yaml"
    if not config_path.exists():
        return KortexConfig()

    try:
        import yaml

        with open(config_path) as f:
            full_config = yaml.safe_load(f) or {}
        plugin_config = full_config.get("plugins", {}).get("kortex", {})
        return KortexConfig.from_dict(plugin_config)
    except Exception:
        return KortexConfig()
