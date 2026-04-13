"""Per-turn affect scoring for KORTEX.

Detects emotional signals from text and produces a multi-dimensional
AffectSignal per turn. Works purely on heuristics — no LLM needed.

Dimensions scored:
  - frustration: annoyance, anger, hostility
  - warmth: friendliness, gratitude, affection
  - humor: playfulness, jokes, teasing, banter
  - hostility: direct attacks, insults, contempt
  - gratitude: thanks, appreciation
  - anxiety: worry, urgency, stress
  - excitement: enthusiasm, eager energy
  - trust_signal: positive trust indicators (vulnerability, personal sharing)

Each dimension is 0.0–1.0 intensity. The aggregate valence and arousal
are derived from these sub-dimensions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from .models import AffectSignal


# --------------------------------------------------------------------------- #
# Pattern definitions: (compiled_regex, dimension, weight)
# --------------------------------------------------------------------------- #

_FRUSTRATION_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\b(?:frustrated|annoyed|irritated|aggravated)\b", re.I), 0.6),
    (re.compile(r"\b(?:ugh|argh|ffs|smh)\b", re.I), 0.4),
    (re.compile(r"\b(?:still (?:not|doesn'?t|won'?t|can'?t))\b", re.I), 0.5),
    (
        re.compile(r"\b(?:again|another|yet again|same (?:issue|problem|bug))\b", re.I),
        0.4,
    ),
    (re.compile(r"\b(?:why (?:is|does|won'?t|can'?t|doesn'?t))\b.*\?", re.I), 0.3),
    (
        re.compile(r"\b(?:this is (?:ridiculous|absurd|insane|unacceptable))\b", re.I),
        0.7,
    ),
    (re.compile(r"\b(?:sick of|tired of|fed up|had enough)\b", re.I), 0.6),
    (re.compile(r"\b(?:waste of time|pointless|useless)\b", re.I), 0.5),
    (re.compile(r"\b(?:terrible|horrible|awful|dreadful|atrocious)\b", re.I), 0.4),
    (re.compile(r"\b(?:broken|bugged|borked|b0rked)\b", re.I), 0.3),
]

_WARMTH_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\b(?:thank(?:s| you)|appreciate|grateful)\b", re.I), 0.5),
    (
        re.compile(
            r"\b(?:love it|love this|that'?s (?:great|awesome|perfect|amazing))\b", re.I
        ),
        0.6,
    ),
    (
        re.compile(
            r"\b(?:you'?re (?:the best|amazing|awesome|great|brilliant))\b", re.I
        ),
        0.7,
    ),
    (re.compile(r"\b(?:nice work|well done|good job|nailed it)\b", re.I), 0.6),
    (re.compile(r"\b(?:glad|happy|pleased|delighted)\b", re.I), 0.4),
    (re.compile(r"\b(?:helpful|exactly what I (?:need|want)ed?)\b", re.I), 0.5),
    (re.compile(r"(?::\)|<3|❤|😊|🙏|👍|🎉)", re.I), 0.3),
]

_HUMOR_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\b(?:lol|lmao|rofl|haha|hehe|😂|🤣|😆)\b", re.I), 0.5),
    (re.compile(r"\b(?:just kidding|jk|joking|banter)\b", re.I), 0.4),
    (re.compile(r"\b(?:funny|hilarious|cracked me up)\b", re.I), 0.5),
    (re.compile(r"(?:;\)|😏|😜|🤪|😈)", re.I), 0.3),
    (re.compile(r"\b(?:no but seriously|in all seriousness)\b", re.I), 0.3),
]

_HOSTILITY_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (
        re.compile(
            r"\b(?:you'?re (?:useless|terrible|stupid|an? idiot|garbage|trash|pathetic))\b",
            re.I,
        ),
        0.8,
    ),
    (re.compile(r"\b(?:fuck (?:you|off|this)|go to hell|screw you)\b", re.I), 0.9),
    (re.compile(r"\b(?:idiot|moron|imbecile|incompetent)\b", re.I), 0.7),
    (re.compile(r"\b(?:worst (?:ever|possible|I'?ve seen))\b", re.I), 0.6),
    (
        re.compile(r"\b(?:I hate (?:you|it|this)|you suck|you'?re the worst)\b", re.I),
        0.8,
    ),
    (re.compile(r"\b(?:shut up|shut the fuck up|stfu)\b", re.I), 0.7),
]

_GRATITUDE_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\bthank(?:s| you)\b(?! for nothing)", re.I), 0.5),
    (re.compile(r"\b(?:really appreciate|much appreciated|so grateful)\b", re.I), 0.7),
    (
        re.compile(
            r"\b(?:couldn'?t (?:have )?do(?:ne)? (?:it |this )?without)\b", re.I
        ),
        0.8,
    ),
    (re.compile(r"\b(?:lifesaver|godsend|you saved)\b", re.I), 0.8),
    (re.compile(r"🙏", re.I), 0.4),
]

_ANXIETY_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\b(?:worried|anxious|nervous|scared|afraid)\b", re.I), 0.6),
    (re.compile(r"\b(?:urgent|asap|emergency|critical|deadline)\b", re.I), 0.5),
    (
        re.compile(r"\b(?:running out of|not enough time|pressure|stressed)\b", re.I),
        0.5,
    ),
    (
        re.compile(
            r"\b(?:what if|I hope|fingers crossed|please (?:work|help))\b", re.I
        ),
        0.3,
    ),
    (re.compile(r"\b(?:panic|freaking out|losing (?:it|my mind))\b", re.I), 0.7),
]

_EXCITEMENT_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\b(?:excited|thrilled|pumped|stoked|can'?t wait)\b", re.I), 0.6),
    (re.compile(r"\b(?:amazing|incredible|unbelievable|mind.?blown)\b", re.I), 0.5),
    (
        re.compile(r"\b(?:YES|FINALLY|WOOO|AWESOME|LET'?S GO)\b"),
        0.6,
    ),  # case-sensitive for caps
    (re.compile(r"(?:🎉|🚀|🔥|✨|💪|🎊|😍)", re.I), 0.3),
    (re.compile(r"!{2,}", re.I), 0.3),
]

_TRUST_SIGNAL_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (
        re.compile(
            r"\b(?:honestly|to be honest|tbh|between (?:us|you and me))\b", re.I
        ),
        0.4,
    ),
    (re.compile(r"\b(?:I (?:trust|rely on|count on|depend on) you)\b", re.I), 0.7),
    (re.compile(r"\b(?:personal(?:ly)?|private|confession|admit)\b", re.I), 0.3),
    (re.compile(r"\b(?:I'?ve never told|just between us|don'?t tell)\b", re.I), 0.6),
    (
        re.compile(r"\b(?:vulnerable|opening up|sharing (?:something|this))\b", re.I),
        0.5,
    ),
]

# Negation and sarcasm detectors — reduce detected warmth/gratitude
_SARCASM_PATTERNS = [
    re.compile(
        r"\b(?:oh (?:great|wonderful|fantastic|brilliant|perfect)|yeah right|sure thing)\b",
        re.I,
    ),
    re.compile(r"\b(?:thanks? for nothing|real helpful|as if)\b", re.I),
    re.compile(r"/s\b", re.I),  # explicit sarcasm tag
]


def _score_dimension(text: str, patterns: List[Tuple[re.Pattern, float]]) -> float:
    """Score a single emotional dimension from 0.0 to 1.0."""
    max_weight = 0.0
    hit_count = 0
    for pat, weight in patterns:
        if pat.search(text):
            max_weight = max(max_weight, weight)
            hit_count += 1
    # Multiple hits boost the score slightly (capped at 1.0)
    if hit_count > 1:
        max_weight = min(1.0, max_weight + hit_count * 0.05)
    return max_weight


def _detect_sarcasm(text: str) -> bool:
    """Check if text contains sarcasm markers."""
    return any(pat.search(text) for pat in _SARCASM_PATTERNS)


def score_affect(user_text: str, assistant_text: str = "") -> AffectSignal:
    """Score the emotional affect of a turn.

    Primarily scores the *user's* text since that's what reflects their
    emotional state. The assistant text is used as secondary context only.
    """
    # Score from user text (primary signal)
    frustration = _score_dimension(user_text, _FRUSTRATION_PATTERNS)
    warmth = _score_dimension(user_text, _WARMTH_PATTERNS)
    humor = _score_dimension(user_text, _HUMOR_PATTERNS)
    hostility = _score_dimension(user_text, _HOSTILITY_PATTERNS)
    gratitude = _score_dimension(user_text, _GRATITUDE_PATTERNS)
    anxiety = _score_dimension(user_text, _ANXIETY_PATTERNS)
    excitement = _score_dimension(user_text, _EXCITEMENT_PATTERNS)
    trust_signal = _score_dimension(user_text, _TRUST_SIGNAL_PATTERNS)

    # Sarcasm check: dampen warmth and gratitude if sarcasm detected
    is_sarcastic = _detect_sarcasm(user_text)
    if is_sarcastic:
        warmth *= 0.2
        gratitude *= 0.2
        frustration = max(frustration, 0.3)  # sarcasm implies mild frustration

    # Hostility boosts frustration (can't be hostile without being frustrated)
    if hostility > 0 and frustration < hostility * 0.5:
        frustration = max(frustration, hostility * 0.5)

    # Derive aggregate valence: positive - negative
    positive = max(warmth, gratitude, excitement * 0.7)
    negative = max(frustration, hostility, anxiety * 0.5)
    if positive > negative:
        valence = min(positive - negative, 1.0)
    elif negative > positive:
        valence = -min(negative - positive, 1.0)
    else:
        valence = 0.0

    # Derive aggregate arousal: how intense is the emotion (regardless of polarity)
    arousal = min(
        1.0,
        max(
            frustration,
            hostility,
            excitement,
            anxiety,
            warmth * 0.6,
            gratitude * 0.5,
            humor * 0.3,
        ),
    )

    # Determine dominant emotion label
    dimensions = {
        "frustration": frustration,
        "warmth": warmth,
        "humor": humor,
        "hostility": hostility,
        "gratitude": gratitude,
        "anxiety": anxiety,
        "excitement": excitement,
        "trust": trust_signal,
    }
    dominant = (
        max(dimensions, key=dimensions.get)
        if any(v > 0 for v in dimensions.values())
        else "neutral"
    )
    if all(v == 0.0 for v in dimensions.values()):
        dominant = "neutral"

    return AffectSignal(
        frustration=frustration,
        warmth=warmth,
        humor=humor,
        hostility=hostility,
        gratitude=gratitude,
        anxiety=anxiety,
        excitement=excitement,
        trust_signal=trust_signal,
        valence=round(valence, 3),
        arousal=round(arousal, 3),
        dominant_emotion=dominant,
        is_sarcastic=is_sarcastic,
    )
