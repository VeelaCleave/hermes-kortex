"""KORTEX plugin configuration.

Reads from Hermes config.yaml under plugins.kortex, with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# Budget allocation for context injection (~2000 tokens total)
DEFAULT_BUDGET = {
    "relationship_state": 200,  # 150-250 tokens
    "conversation_summaries": 100,
    "stable_facts": 350,  # 300-400 tokens
    "episodic_memories": 450,  # reserve space for graph-expanded recall
    "graph_memories": 150,
    "open_loops": 200,  # 150-250 tokens
    "reflections": 200,  # 150-250 tokens
    "reserve": 150,  # 100-200 tokens
}


@dataclass
class KortexConfig:
    """Configuration for the KORTEX memory plugin."""

    # Database
    db_path: Optional[str] = None  # None = auto (hermes_home / "kortex.db")

    # Episodic memory
    max_episodes_per_recall: int = 4
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
    max_facts_per_recall: int = 6
    fact_confidence_threshold: float = 0.3

    # Open loops
    max_loops_per_recall: int = 3

    # Reflections
    max_reflections_per_recall: int = 3
    reflection_confidence_threshold: float = 0.4

    # Context budget (tokens per section)
    budget: Dict[str, int] = field(default_factory=lambda: DEFAULT_BUDGET.copy())

    # Total hard cap
    total_budget: int = 1800

    # Extraction
    auto_extract: bool = True  # extract facts/loops from turns automatically
    affect_calibration_min_samples: int = 20
    extraction_mode: str = "heuristic"

    # Identity
    soul_path: Optional[str] = None  # custom SOUL.md path (None = hermes default)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KortexConfig":
        """Create config from a dict (e.g. from YAML)."""
        budget = data.get("budget", DEFAULT_BUDGET.copy())
        # Merge with defaults so missing keys don't break
        merged_budget = DEFAULT_BUDGET.copy()
        merged_budget.update(budget)

        return cls(
            db_path=data.get("db_path"),
            max_episodes_per_recall=data.get("max_episodes_per_recall", 4),
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
            max_facts_per_recall=data.get("max_facts_per_recall", 6),
            fact_confidence_threshold=data.get("fact_confidence_threshold", 0.3),
            max_loops_per_recall=data.get("max_loops_per_recall", 3),
            max_reflections_per_recall=data.get("max_reflections_per_recall", 3),
            reflection_confidence_threshold=data.get(
                "reflection_confidence_threshold", 0.4
            ),
            budget=merged_budget,
            total_budget=data.get("total_budget", 1800),
            auto_extract=data.get("auto_extract", True),
            affect_calibration_min_samples=data.get(
                "affect_calibration_min_samples", 20
            ),
            extraction_mode=data.get("extraction_mode", "heuristic"),
            soul_path=data.get("soul_path"),
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
