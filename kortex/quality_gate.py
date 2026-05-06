"""Quality gate for KORTEX facts.

Filters out ephemeral facts and boosts confidence for durable knowledge.
The core problem: the LLM extractor produces garbage like "I spent 2 days 
trying to fix" as a fact. Those are episode details, not facts.

Rules:
1. Time-bound facts ("yesterday", "today") get lower confidence
2. Narrative facts ("I did X") get filtered unless they encode a pattern
3. Emotional snapshots ("was frustrated") go to episodes, not facts
4. Action records ("reverted 37 commits") are episodes, not facts
5. Durable knowledge ("uses Python", "prefers dark mode") passes through
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# Predicates that typically indicate DURABLE facts
_DURABLE_PREDICATES = {
    "uses", "prefers", "likes", "dislikes", "owns", "has",
    "works_on", "works_at", "lives_in", "named", "decided",
    "project_location", "project_uses", "configures", "deploys",
}

# Predicates that typically indicate EPHEMERAL facts (should be episodes)
_EPHEMERAL_PREDICATES = {
    "spent time", "action taken", "current status", "user emotional state",
    "attempted solution", "identified problem", "tried", "did",
    "started", "finished", "found", "noticed", "observed",
}

# Patterns that indicate time-bound (ephemeral) facts
_TIME_BOUND_PATTERNS = [
    r'\byesterday\b', r'\btoday\b', r'\bthis (morning|afternoon|evening|week|month)\b',
    r'\blast\s+(week|month|year)\b', r'\b\d+\s+(days?|hours?|minutes?)\s+ago\b',
    r'\bon\s+\d{1,2}/\d{1,2}', r'\bat\s+\d{1,2}:\d{2}',
    r'\bjust\b', r'\bnow\b', r'\bcurrently\b', r'\bpresently\b',
]

# Patterns that indicate emotional snapshots (not durable facts)
_EMOTIONAL_SNAPSHOT_PATTERNS = [
    r'\b(feel|felt|feeling)\s+(happy|sad|tired|frustrated|excited|lost|overwhelmed)',
    r'\b(is|are|was|were)\s+(happy|sad|tired|frustrated|excited|lost|overwhelmed)',
    r'\b(mood|emotion|feeling)\s+(is|was|became)',
]

# Patterns that indicate narrative/action (not facts)
_NARRATIVE_PATTERNS = [
    r'^\d+\s+(days?|hours?|minutes?)\s+(trying|spent|worked)',
    r'\b(reverted|deleted|added|changed|updated|fixed|broke)\s+\d+',
    r'\b(I|we)\s+(tried|attempted|started|finished|completed)',
    r'\b(action|step|task)\s+(taken|completed|done)',
    r'\bstatus\s+(is|was|became)',
    r'\b(current|latest|most recent)',
]


def evaluate_fact_quality(predicate: str, object_text: str) -> Dict[str, float]:
    """Evaluate the quality of a fact. Returns a quality score dict.
    
    Scores:
    - durability: 0.0 (ephemeral) to 1.0 (rock solid)
    - confidence: 0.0 to 1.0 (adjusted confidence)
    - is_ephemeral: bool (should this even be a fact?)
    - is_narrative: bool (is this a story, not a fact?)
    - is_emotional_snapshot: bool (is this a mood, not a fact?)
    - is_time_bound: bool (does it expire soon?)
    """
    pred_lower = predicate.lower()
    obj_lower = object_text.lower()
    combined = f"{pred_lower} {obj_lower}"
    
    is_time_bound = bool(re.search('|'.join(_TIME_BOUND_PATTERNS), combined))
    is_emotional_snapshot = bool(re.search('|'.join(_EMOTIONAL_SNAPSHOT_PATTERNS), combined))
    is_narrative = bool(re.search('|'.join(_NARRATIVE_PATTERNS), combined))
    is_ephemeral_pred = pred_lower in _EPHEMERAL_PREDICATES
    
    # Durability score
    durability = 1.0
    
    if is_ephemeral_pred:
        durability -= 0.7
    if is_time_bound:
        durability -= 0.5
    if is_emotional_snapshot:
        durability -= 0.6
    if is_narrative:
        durability -= 0.5
        
    durability = max(0.0, min(1.0, durability))
    
    # Confidence adjustment
    confidence = 0.5  # Base
    if pred_lower in _DURABLE_PREDICATES:
        confidence = 0.7
    if durability < 0.3:
        confidence *= durability
    
    return {
        "durability": round(durability, 2),
        "confidence": round(confidence, 2),
        "is_ephemeral": is_ephemeral_pred or durability < 0.3,
        "is_narrative": is_narrative,
        "is_emotional_snapshot": is_emotional_snapshot,
        "is_time_bound": is_time_bound,
    }


def should_keep_fact(predicate: str, object_text: str) -> bool:
    """Decide if a fact should be kept in the facts table.
    
    Returns False for clearly ephemeral facts that belong in episodes.
    """
    quality = evaluate_fact_quality(predicate, object_text)
    return not quality["is_ephemeral"]


def adjust_fact_confidence(predicate: str, object_text: str, base_confidence: float) -> float:
    """Adjust the confidence score based on quality heuristics."""
    quality = evaluate_fact_quality(predicate, object_text)
    return quality["confidence"]


def suggest_predicate(predicate: str, object_text: str) -> Optional[str]:
    """Suggest a better predicate if the current one is weak.
    
    Examples:
    - "spent time" + "fixing X" → "works_on"
    - "identified problem" + "memory org" → "project_issue"
    - "action taken" + "reverted" → (might be filtered entirely)
    """
    # Don't suggest changes for already-good predicates
    if predicate.lower() in _DURABLE_PREDICATES:
        return None
    
    # If it's ephemeral, the fact might get filtered anyway
    if not should_keep_fact(predicate, object_text):
        return None
    
    # Suggest improvements
    obj_lower = object_text.lower()
    
    if "memory" in obj_lower or "kortex" in obj_lower or "mempalace" in obj_lower:
        return "project_works_on"
    
    if any(w in obj_lower for w in ["python", "docker", "git", "node", "react"]):
        return "uses"
    
    if any(w in obj_lower for w in ["prefer", "like", "love", "favor"]):
        return "prefers"
    
    return None
