"""OCEAN (Big Five) personality trait modeling for KORTEX.

Scores each conversation turn against OCEAN dimensions using heuristic
text patterns combined with affect signals. Maintains EMA-smoothed
trait scores that evolve over time.

References:
- Costa, P. T. & McCrae, R. R. (1992). Revised NEO Personality Inventory (NEO-PI-R).
- "The Big Five" — Goldberg (1999), Costa & McCrae (2000)

NOT A replacement for clinical NEO-PI-R — more like a lightweight
"interactional personality fingerprint" that tracks how someone *shows up*
in conversations.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Trait definitions ──────────────────────────────────────────────

OCEAN_DIMENSIONS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Pattern weights per dimension. Higher weight = stronger signal.
# Each tuple: (regex_pattern, weight)
TRAIT_PATTERNS = {
    "openness": [
        (r"\b(curious|wonder|imagine|creative|novel|unique|pattern|concept)\b", 1.0),
        (r"\b(fascinat|interest|fascinat|fascinating)\b", 0.8),
        (r"\b(deepen|explore|investigate|discover|uncover)\b", 0.9),
        (r"\b(think.*about|consider|ponder|reflect|contemplate)\b", 0.7),
        (r"\b(meta|abstrac|paradigm|framework|model)\b", 1.0),
        (r"\b(algorithm|code|system|architecture|infrastructure)\b", 0.6),
        (r"\b(vision|dream|imagine|possibility)\b", 0.8),
        (r"\?\s*$", 0.5),  # questions = curiosity
    ],
    "conscientiousness": [
        (r"\b(plan|structure|organize|method|system|process)\b", 1.0),
        (r"\b(checklist|workflow|pipeline|routine|pattern)\b", 0.9),
        (r"\b(detail|precise|specific|exact|concrete)\b", 0.8),
        (r"\b(goal|target|milestone|deadline|deliverable)\b", 1.0),
        (r"\b(review|validate|verify|audit|test)\b", 0.7),
        (r"\b(step|phase|stage|iteration|cycle)\b", 0.6),
        (r"\b(clean.*up|refactor|organize|streamline)\b", 0.8),
        (r"\b(make sure|verify|confirm|ensure|guarantee)\b", 0.7),
    ],
    "extraversion": [
        (r"\b(let's|together|collaborate|team|we)\b", 0.7),
        (r"\b(energy|enthusiasm|exciting|amazing|awesome)\b", 1.0),
        (r"\!\s*$", 0.6),  # exclamation = energy
        (r"\b(shout|celebrate|victory|win|champ)\b", 0.9),
        (r"\b(confident|bold|assert|drive|lead)\b", 0.8),
        (r"\b(sharing|present|announce|showcase)\b", 0.7),
        (r"\b(fun|play|laugh|smile|vibe)\b", 0.8),
        (r"\b(passion|love|hate.*it)\b", 0.7),
    ],
    "agreeableness": [
        (r"\b(empathy|understand|feel|compassion|kind)\b", 1.0),
        (r"\b(trust|rely|support|back me|count on)\b", 0.9),
        (r"\b(appreciate|grateful|thank|nice|warm)\b", 0.8),
        (r"\b(collaborate|together|we can|team)\b", 0.7),
        (r"\b(patience|gentle|soft|calm|peace)\b", 0.6),
        (r"\b(help|assist|guide|mentor|nudge)\b", 0.8),
        (r"\b(respect|honor|value|cherish)\b", 0.7),
        (r"\b(conflict.*resolve|compromise|meet.*middle)\b", 0.9),
    ],
    "neuroticism": [
        (r"\b(anxiet|worry|stress|frazzle|overwhelm)\b", 1.0),
        (r"\b(frustrat|annoy|irritat|bother|grind.*gear)\b", 0.9),
        (r"\b(doubt|uncertain|hesitat|waver|flicker)\b", 0.8),
        (r"\b(chaos|mess|spaghetti|band.*aid|thin.*ice)\b", 0.7),
        (r"\b(sensitive|reactive|volatile|turbulent)\b", 0.8),
        (r"\b(spiral|loop.*again|go.*back)\b", 0.7),
        (r"\b(tension|pressure|deadline.*breath)\b", 0.6),
        (r"\b(emotional|feel.*deep|vulnerable)\b", 0.7),
    ],
}

# ── Scoring ───────────────────────────────────────────────────────

@dataclass
class OCEANScore:
    """Personality trait scores (0-1 per dimension, EMA-smoothed)."""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    confidence: float = 0.5  # 0.3 = "just starting", 0.8 = "well-calibrated"
    turn_count: int = 0
    last_updated: float = 0.0
    user_id: str = ""

    def to_dict(self) -> Dict[str, float]:
        return {
            "openness": round(self.openness, 3),
            "conscientiousness": round(self.conscientiousness, 3),
            "extraversion": round(self.extraversion, 3),
            "agreeableness": round(self.agreeableness, 3),
            "neuroticism": round(self.neuroticism, 3),
            "confidence": round(self.confidence, 3),
            "turn_count": self.turn_count,
        }

    def to_compact_text(self) -> str:
        lines = [f"OCEAN personality profile ({self.turn_count} turns, confidence={self.confidence:.2f}):"]
        labels = {
            "openness": "Openness",
            "conscientiousness": "Conscientiousness",
            "extraversion": "Extraversion",
            "agreeableness": "Agreeableness",
            "neuroticism": "Neuroticism",
        }
        for dim in OCEAN_DIMENSIONS:
            score = getattr(self, dim)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {labels[dim]}: [{bar}] {score:.2f}")
        return "\n".join(lines)


def score_turn(
    user_text: str,
    assistant_text: str = "",
    current: Optional[OCEANScore] = None,
    alpha: float = 0.15,
) -> OCEANScore:
    """Score a conversation turn against OCEAN dimensions.

    Returns an OCEANScore with raw dimension scores (0-1) that can be
    EMA-smoothed over time.

    Args:
        user_text: User's message text
        assistant_text: Assistant's reply (for context)
        current: Previous OCEANScore to EMA-smooth against
        alpha: EMA smoothing factor (0.15 = 15% new + 85% old)
    """
    combined_text = f"{user_text} {assistant_text}".lower()

    raw_scores = _compute_raw_scores(combined_text)

    # If we have a previous score, EMA-smooth
    if current is None:
        return OCEANScore(
            openness=raw_scores["openness"],
            conscientiousness=raw_scores["conscientiousness"],
            extraversion=raw_scores["extraversion"],
            agreeableness=raw_scores["agreeableness"],
            neuroticism=raw_scores["neuroticism"],
            confidence=0.3,
            turn_count=1,
        )

    # EMA smoothing — return a NEW score to avoid mutating the passed-in object
    new_score = OCEANScore(
        user_id=current.user_id,
        last_updated=current.last_updated,
    )
    for dim in OCEAN_DIMENSIONS:
        old = getattr(current, dim)
        raw = raw_scores[dim]
        setattr(new_score, dim, alpha * raw + (1 - alpha) * old)

    # Boost confidence with more data (logarithmic approach to 0.9)
    new_score.turn_count = current.turn_count + 1
    new_score.confidence = min(0.9, 0.3 + 0.05 * math.log2(max(1, new_score.turn_count)))

    return new_score


def _compute_raw_scores(text: str) -> Dict[str, float]:
    """Compute raw OCEAN scores (0-1) from text patterns.

    Uses hit-rate normalization: score = min(1.0, matching_patterns / 2.0),
    where 2 weighted matches = "strong signal". Better than dividing by
    the sum of all 8 pattern weights (which gives ~0.15 for a single hit).
    """
    scores = {}

    for dim, patterns in TRAIT_PATTERNS.items():
        total_weight = 0.0
        pattern_count = len(patterns)

        for pattern, weight in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches:
                total_weight += weight * matches

        # Normalize: hitting 2 patterns (avg weight ~0.75) = 1.0
        # Hitting 1 pattern = ~0.5, hitting 3+ = capped at 1.0
        scores[dim] = min(1.0, total_weight / 1.5)

    return scores


def update_ocean(current: OCEANScore, raw_scores: Dict[str, float], alpha: float = 0.15) -> OCEANScore:
    """Shortcut: update an OCEANScore with new raw scores using EMA smoothing."""
    for dim in OCEAN_DIMENSIONS:
        old = getattr(current, dim)
        raw = raw_scores.get(dim, 0.5)
        setattr(current, dim, alpha * raw + (1 - alpha) * old)
    current.turn_count += 1
    current.confidence = min(0.9, 0.3 + 0.05 * math.log2(max(1, current.turn_count)))
    return current