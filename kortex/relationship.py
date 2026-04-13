"""Relationship dynamics tracker for KORTEX.

Updates the relationship state dimensions after each turn based on
the detected affect signal. Uses momentum (exponential moving average)
so dimensions don't swing wildly, and natural regression toward baseline
over time.

The update model:
  new_value = (1 - alpha) * current_value + alpha * target_value
where alpha is the learning rate (momentum), and target_value is derived
from the turn's affect signal.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from .models import AffectSignal, RelationshipState


# How quickly each dimension responds to new signals (0 = inert, 1 = instant)
_LEARNING_RATES = {
    "warmth": 0.15,
    "trust": 0.08,  # trust moves slowly — hard to build, slow to lose
    "tension": 0.25,  # tension reacts fast
    "familiarity": 0.03,  # familiarity only grows, very slowly
    "humor": 0.20,
    "formality": 0.10,
    "volatility": 0.15,
}

# Natural regression targets — where dimensions drift when nothing happens
_BASELINES = {
    "warmth": 0.5,
    "trust": 0.5,
    "tension": 0.0,
    "familiarity": None,  # familiarity never regresses
    "humor": 0.0,
    "formality": 0.5,
    "volatility": 0.0,
}

# Regression rate per day of inactivity (how fast we drift toward baseline)
_REGRESSION_RATE = 0.02


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ema(current: float, target: float, alpha: float) -> float:
    """Exponential moving average step."""
    return (1 - alpha) * current + alpha * target


def compute_relationship_delta(
    affect: AffectSignal,
    current: RelationshipState,
) -> dict:
    """Compute target deltas for each relationship dimension from an affect signal.

    Returns a dict of dimension -> target_value that the EMA should move toward.
    """
    targets = {}

    # --- Warmth ---
    # Warmth rises from gratitude, warmth signals, humor
    # Drops from hostility, frustration
    warmth_push = max(affect.warmth, affect.gratitude) * 0.8 + affect.humor * 0.2
    warmth_pull = max(affect.hostility, affect.frustration * 0.6)
    if warmth_push > warmth_pull:
        targets["warmth"] = _clamp(current.warmth + (warmth_push - warmth_pull) * 0.4)
    elif warmth_pull > warmth_push:
        targets["warmth"] = _clamp(current.warmth - (warmth_pull - warmth_push) * 0.5)

    # --- Trust ---
    # Trust grows from trust signals, consistency (low volatility)
    # Drops sharply from hostility, sarcasm
    if affect.trust_signal > 0.3:
        targets["trust"] = _clamp(current.trust + affect.trust_signal * 0.2)
    if affect.hostility > 0.5:
        targets["trust"] = _clamp(current.trust - affect.hostility * 0.3)
    if affect.is_sarcastic and affect.hostility > 0.3:
        targets["trust"] = _clamp(current.trust - 0.1)

    # --- Tension ---
    # Rises from frustration, hostility, anxiety
    # Drops from warmth, gratitude, humor
    if affect.frustration > 0.2 or affect.hostility > 0.2 or affect.anxiety > 0.3:
        tension_target = max(affect.frustration, affect.hostility, affect.anxiety * 0.7)
        targets["tension"] = _clamp(tension_target)
    elif affect.warmth > 0.3 or affect.gratitude > 0.3 or affect.humor > 0.3:
        # Active de-escalation
        targets["tension"] = _clamp(current.tension - 0.2)

    # --- Familiarity ---
    # Always increases slightly with each turn (you get more familiar over time)
    # Bigger bump for personal sharing (trust signals)
    fam_increment = 0.01
    if affect.trust_signal > 0.3:
        fam_increment += affect.trust_signal * 0.05
    targets["familiarity"] = _clamp(current.familiarity + fam_increment)

    # --- Humor ---
    # Responds to humor signals, dampened by hostility
    if affect.humor > 0.2:
        targets["humor"] = _clamp(affect.humor)
    elif affect.hostility > 0.3 or affect.frustration > 0.5:
        targets["humor"] = 0.0

    # --- Formality ---
    # Drops when humor/casual language detected, rises with professional tone
    if affect.humor > 0.3 or current.familiarity > 0.5:
        targets["formality"] = _clamp(current.formality - 0.1)
    elif current.familiarity < 0.2 and affect.humor == 0:
        targets["formality"] = _clamp(current.formality + 0.05)

    # --- Volatility ---
    # High when emotions swing a lot between turns
    # We measure the absolute magnitude of emotional change
    emotional_intensity = max(
        affect.frustration,
        affect.hostility,
        affect.excitement,
        affect.warmth * 0.7,
        affect.anxiety,
    )
    if emotional_intensity > 0.5:
        targets["volatility"] = _clamp(emotional_intensity * 0.6)
    else:
        targets["volatility"] = 0.0

    return targets


def apply_regression(
    state: RelationshipState,
    days_since_last: float,
) -> RelationshipState:
    """Apply natural regression toward baselines for time elapsed since last interaction."""
    if days_since_last <= 0:
        return state

    regression_amount = min(days_since_last * _REGRESSION_RATE, 0.3)

    for dim, baseline in _BASELINES.items():
        if baseline is None:
            continue
        current = getattr(state, dim)
        diff = baseline - current
        if abs(diff) < 0.01:
            continue
        new_value = current + diff * regression_amount
        setattr(state, dim, round(_clamp(new_value), 4))

    return state


def update_relationship(
    affect: AffectSignal,
    current: RelationshipState,
    days_since_last: float = 0.0,
) -> RelationshipState:
    """Update relationship state from an affect signal.

    1. Apply time-based regression toward baselines
    2. Compute target deltas from affect
    3. Apply EMA to smooth transitions
    4. Increment turn counter
    """
    # Step 1: Time-based regression
    if days_since_last > 0.5:
        current = apply_regression(current, days_since_last)

    # Step 2: Compute targets
    targets = compute_relationship_delta(affect, current)

    # Step 3: Apply EMA for each dimension with a target
    for dim, target in targets.items():
        alpha = _LEARNING_RATES.get(dim, 0.1)
        current_val = getattr(current, dim)
        new_val = _ema(current_val, target, alpha)
        setattr(current, dim, round(_clamp(new_val), 4))

    # Step 4: Bookkeeping
    current.total_turns += 1
    current.last_updated = datetime.now(timezone.utc)

    return current
