"""Reflection extraction for KORTEX — detects mistakes, successes,
style preferences, and identity directives from turns."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from .db import KortexDB
from .models import AffectSignal, IdentityDelta, Reflection

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Mistake detection patterns
# --------------------------------------------------------------------------- #

# User correcting the agent
_CORRECTION_PATTERNS = [
    re.compile(
        r"\b(?:no,?\s+(?:I (?:said|meant|asked)|that'?s (?:not|wrong)))\b", re.I
    ),
    re.compile(r"\b(?:that'?s (?:incorrect|not right|not what I))\b", re.I),
    re.compile(r"\b(?:you (?:got|have) (?:it |that )?wrong)\b", re.I),
    re.compile(r"\b(?:I (?:already|just) (?:told|said|mentioned|explained))\b", re.I),
    re.compile(r"\b(?:wrong|incorrect|nope|not quite)\b(?:\s*[.,!])", re.I),
    re.compile(r"\b(?:try again|redo|undo that|revert|roll ?back)\b", re.I),
    re.compile(r"\b(?:that'?s not how|you misunderstood|you missed the point)\b", re.I),
    re.compile(r"\b(?:I didn'?t (?:ask|want|mean) (?:for )?(?:that|this))\b", re.I),
]

# User having to repeat themselves
_REPETITION_PATTERNS = [
    re.compile(r"\b(?:I (?:already|just) (?:said|told|asked|mentioned))\b", re.I),
    re.compile(r"\b(?:like I said|as I said|as I mentioned|I repeat)\b", re.I),
    re.compile(r"\b(?:for the (?:second|third|last) time)\b", re.I),
    re.compile(r"\b(?:again|once more|one more time)\b", re.I),
    re.compile(r"\b(?:listen|pay attention|read (?:what I|my))\b", re.I),
]

# Agent produced wrong output
_WRONG_OUTPUT_PATTERNS = [
    re.compile(r"\b(?:doesn'?t (?:work|compile|run|build|pass))\b", re.I),
    re.compile(r"\b(?:still (?:broken|failing|not working|erroring))\b", re.I),
    re.compile(r"\b(?:you (?:broke|messed up|introduced|caused))\b", re.I),
    re.compile(
        r"\b(?:that (?:broke|introduced|caused)\s+(?:a |an |the )?(?:bug|error|issue|problem))\b",
        re.I,
    ),
]


# --------------------------------------------------------------------------- #
# Success/praise detection patterns
# --------------------------------------------------------------------------- #

_PRAISE_PATTERNS = [
    re.compile(r"\b(?:perfect|exactly|spot on|nailed it|that'?s (?:it|right))\b", re.I),
    re.compile(
        r"\b(?:great job|nice work|well done|brilliant|love (?:it|this|that))\b", re.I
    ),
    re.compile(
        r"\b(?:this is (?:exactly|just) what I (?:wanted|needed|meant))\b", re.I
    ),
    re.compile(r"\b(?:yes[!.]*\s*(?:that|this|exactly|perfect))\b", re.I),
    re.compile(
        r"\b(?:you'?re (?:the best|amazing|a lifesaver|awesome|genius))\b", re.I
    ),
    re.compile(r"\b(?:much better|way better|so much better|huge improvement)\b", re.I),
]


# --------------------------------------------------------------------------- #
# Style preference patterns
# --------------------------------------------------------------------------- #

_STYLE_PREFERENCE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Verbosity preferences
    (
        re.compile(
            r"\b(?:too (?:verbose|wordy|long|detailed)|keep it (?:short|brief|concise))\b",
            re.I,
        ),
        "User prefers concise responses",
    ),
    (
        re.compile(
            r"\b(?:more detail|elaborate|explain (?:more|further)|be more (?:specific|detailed))\b",
            re.I,
        ),
        "User prefers detailed explanations",
    ),
    # Format preferences
    (
        re.compile(
            r"\b(?:use (?:bullet|numbered) (?:points|lists?)|list (?:them|it|the))\b",
            re.I,
        ),
        "User prefers structured/listed output",
    ),
    (
        re.compile(
            r"\b(?:don'?t (?:need|want) (?:the |a )?(?:explanation|commentary|preamble))\b",
            re.I,
        ),
        "User prefers direct output without explanation",
    ),
    (
        re.compile(
            r"\b(?:just (?:the|give me (?:the )?)?code|code only|no (?:explanation|comments))\b",
            re.I,
        ),
        "User prefers code-only responses without narration",
    ),
    (
        re.compile(
            r"\b(?:show (?:me )?(?:the |your )?(?:work|reasoning|thinking))\b", re.I
        ),
        "User prefers to see reasoning/thinking process",
    ),
    # Tone preferences
    (
        re.compile(r"\b(?:too formal|loosen up|be (?:more )?casual|chill)\b", re.I),
        "User prefers casual tone",
    ),
    (
        re.compile(r"\b(?:be (?:more )?(?:professional|formal|serious))\b", re.I),
        "User prefers professional tone",
    ),
    (
        re.compile(
            r"\b(?:don'?t (?:be |sound )?(?:so )?(?:robotic|stiff|clinical))\b", re.I
        ),
        "User dislikes robotic/clinical tone",
    ),
    # Approach preferences
    (
        re.compile(r"\b(?:don'?t (?:ask|check|confirm)|just (?:do|go ahead))\b", re.I),
        "User prefers autonomous action without confirmation",
    ),
    (
        re.compile(
            r"\b(?:ask (?:me )?(?:first|before)|check with me|don'?t assume)\b", re.I
        ),
        "User prefers to be consulted before changes",
    ),
    (
        re.compile(
            r"\b(?:one (?:step|thing) at a time|slow(?:er)?|smaller (?:steps|changes))\b",
            re.I,
        ),
        "User prefers incremental/step-by-step approach",
    ),
]


# --------------------------------------------------------------------------- #
# Identity delta patterns
# --------------------------------------------------------------------------- #

_IDENTITY_DIRECTIVE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\byou (?:should|need to) (?:be |act |behave )(?:more |less )?(\w+)", re.I
        ),
        "identity",
    ),
    (
        re.compile(
            r"\b(?:from now on|going forward),?\s+(.{10,80}?)(?:\.(?:\s|$)|!|$)", re.I
        ),
        "directive",
    ),
    (
        re.compile(
            r"\b(?:I want you to|you should always|never (?:do|say|use|give))\s+(.{10,80}?)(?:\.(?:\s|$)|!|$)",
            re.I,
        ),
        "directive",
    ),
    (
        re.compile(
            r"\byour (?:name|personality|role) (?:is|should be)\s+(.{3,50}?)(?:\.(?:\s|$)|!|$)",
            re.I,
        ),
        "identity",
    ),
    (
        re.compile(
            r"\b(?:act|behave|respond) (?:like|as if|as)\s+(.{5,60}?)(?:\.(?:\s|$)|!|$)",
            re.I,
        ),
        "identity",
    ),
    (
        re.compile(r"\bdon'?t (?:ever|always)\s+(.{5,80}?)(?:\.(?:\s|$)|!|$)", re.I),
        "constraint",
    ),
    (
        re.compile(
            r"\bstop\s+(?:being|doing|saying)\s+(.{5,60}?)(?:\.(?:\s|$)|!|$)", re.I
        ),
        "constraint",
    ),
]

# Minimum text length for a reflection to be stored
_MIN_REFLECTION_LENGTH = 10
# Maximum
_MAX_REFLECTION_LENGTH = 300
# Jaccard similarity threshold for "same reflection"
_REFLECTION_SIMILARITY_THRESHOLD = 0.55


def extract_mistakes(
    user_text: str, assistant_text: str, affect: AffectSignal
) -> List[str]:
    """Detect mistake signals from user text + emotional context.

    Returns a list of mistake description strings.
    """
    mistakes = []

    for patterns, prefix in [
        (_CORRECTION_PATTERNS, "Correction"),
        (_REPETITION_PATTERNS, "Had to repeat"),
        (_WRONG_OUTPUT_PATTERNS, "Produced error"),
    ]:
        pat = next((pat for pat in patterns if pat.search(user_text)), None)
        if pat:
            desc = _extract_context(user_text, pat, prefix=prefix)
            if desc:
                mistakes.append(desc)

    return mistakes


def extract_successes(
    user_text: str, assistant_text: str, affect: AffectSignal
) -> List[str]:
    """Detect success signals from user text + emotional context.

    Returns a list of success description strings.
    """
    successes = []

    for pat in _PRAISE_PATTERNS:
        if pat.search(user_text):
            desc = _extract_context(user_text, pat, prefix="Approach worked well")
            if desc:
                successes.append(desc)
            break

    return successes


def extract_style_preferences(user_text: str) -> List[str]:
    """Detect explicit style preferences from user text.

    Returns a list of preference description strings.
    """
    prefs = []
    for pat, description in _STYLE_PREFERENCE_PATTERNS:
        if pat.search(user_text):
            prefs.append(description)

    return prefs


def extract_identity_directives(user_text: str) -> List[Tuple[str, str]]:
    """Detect identity directives from user text.

    Returns list of (kind, text) tuples for IdentityDelta creation.
    """
    directives = []
    for pat, kind in _IDENTITY_DIRECTIVE_PATTERNS:
        match = pat.search(user_text)
        if match:
            text = match.group(1).strip().rstrip(".,!?;:")
            if len(text) >= 5:
                directives.append((kind, text))

    return directives


def process_reflections(
    db: KortexDB,
    user_text: str,
    assistant_text: str,
    affect: AffectSignal,
    episode_id: int,
    user_id: str = "__default__",
) -> List[Reflection]:
    """Main entry point: extract and store/reinforce reflections from a turn.

    This is called from provider.sync_turn() after ingestion and affect scoring.
    """
    created = []

    # 1. Mistakes
    for mistake_text in extract_mistakes(user_text, assistant_text, affect):
        ref = _store_or_reinforce(
            db,
            kind="mistake",
            text=mistake_text,
            episode_id=episode_id,
            base_confidence=0.4 if affect.frustration > 0.3 else 0.3,
            user_id=user_id,
        )
        if ref:
            created.append(ref)

    # 2. Successes
    for success_text in extract_successes(user_text, assistant_text, affect):
        ref = _store_or_reinforce(
            db,
            kind="pattern",
            text=success_text,
            episode_id=episode_id,
            base_confidence=0.4 if affect.warmth > 0.3 else 0.3,
            user_id=user_id,
        )
        if ref:
            created.append(ref)

    # 3. Style preferences
    for pref_text in extract_style_preferences(user_text):
        ref = _store_or_reinforce(
            db,
            kind="preference",
            text=pref_text,
            episode_id=episode_id,
            base_confidence=0.5,  # explicit preferences start higher
            user_id=user_id,
        )
        if ref:
            created.append(ref)

    # 4. Identity directives → stored as IdentityDelta, not Reflection
    for kind, directive_text in extract_identity_directives(user_text):
        delta = IdentityDelta(
            user_id=user_id,
            text=f"[{kind}] {directive_text}",
            confidence=0.5,
            source_episode_id=episode_id,
        )
        db.insert_identity_delta(delta)
        logger.debug("KORTEX identity delta: [%s] %s", kind, directive_text)

    return created


def _store_or_reinforce(
    db: KortexDB,
    kind: str,
    text: str,
    episode_id: int,
    base_confidence: float = 0.3,
    user_id: str = "__default__",
) -> Optional[Reflection]:
    """Store a new reflection or reinforce an existing similar one."""
    text = text.strip()
    if len(text) < _MIN_REFLECTION_LENGTH:
        return None
    if len(text) > _MAX_REFLECTION_LENGTH:
        text = text[:_MAX_REFLECTION_LENGTH]

    # Check for existing similar reflection
    existing = _find_similar_reflection(db, kind, text, user_id=user_id)
    if existing:
        # Reinforce: boost confidence and update timestamp
        db.reinforce_reflection(existing.id, confidence_boost=0.1)
        logger.debug(
            "KORTEX reinforced reflection #%d (count=%d): %s",
            existing.id,
            existing.reinforcement_count + 1,
            existing.text[:60],
        )
        return existing

    # New reflection
    ref = Reflection(
        user_id=user_id,
        kind=kind,
        text=text,
        confidence=base_confidence,
        source_episode_id=episode_id,
    )
    ref.id = db.insert_reflection(ref)
    logger.debug("KORTEX new reflection [%s]: %s", kind, text[:60])
    return ref


def _find_similar_reflection(
    db: KortexDB, kind: str, text: str, user_id: str = "__default__"
) -> Optional[Reflection]:
    """Find an existing reflection that's similar enough to reinforce."""
    existing = db.get_reflections(kind=kind, limit=50, user_id=user_id)

    for ref in existing:
        if _reflections_similar(ref.text, text):
            return ref

    # Also try FTS search for broader matching
    try:
        fts_matches = db.search_reflections(text, limit=5, user_id=user_id)
        for ref in fts_matches:
            if ref.kind == kind and _reflections_similar(ref.text, text):
                return ref
    except Exception:
        pass

    return None


def _reflections_similar(existing: str, new: str) -> bool:
    """Check if two reflection texts are similar enough to be the same insight."""
    # Normalize
    existing_words = set(existing.lower().split())
    new_words = set(new.lower().split())

    if not existing_words or not new_words:
        return False

    # Remove common prefix words that don't carry meaning
    _noise = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "are",
        "were",
        "to",
        "of",
        "in",
        "for",
        "and",
        "or",
        "but",
        "that",
        "this",
        "it",
        "with",
        "on",
        "at",
        "by",
    }
    existing_signal = existing_words - _noise
    new_signal = new_words - _noise

    if not existing_signal or not new_signal:
        # Fall back to full word sets if all signal words removed
        existing_signal = existing_words
        new_signal = new_words

    intersection = existing_signal & new_signal
    union = existing_signal | new_signal

    jaccard = len(intersection) / len(union) if union else 0.0
    return jaccard >= _REFLECTION_SIMILARITY_THRESHOLD


def _extract_context(text: str, pattern: re.Pattern, prefix: str = "") -> str:
    """Extract meaningful context around a pattern match."""
    match = pattern.search(text)
    if not match:
        return ""

    # Get the sentence containing the match
    start = match.start()
    end = match.end()

    # Expand to sentence boundaries
    sent_start = max(0, text.rfind(".", 0, start) + 1)
    sent_end = text.find(".", end)
    if sent_end == -1:
        sent_end = min(len(text), end + 100)
    else:
        sent_end = min(sent_end + 1, len(text))

    sentence = text[sent_start:sent_end].strip()

    if prefix:
        sentence = f"{prefix}: {sentence}"

    return sentence[:_MAX_REFLECTION_LENGTH]
