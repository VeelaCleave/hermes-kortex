"""Turn ingestion pipeline for KORTEX.

Writes raw episodes on every sync_turn(), then extracts metadata
(summary, salience, valence, entities, topics) either inline via
heuristics or asynchronously via LLM extraction.

Stage 1: Heuristic extraction (no LLM dependency)
Stage 2: Fact extraction, deduplication, open loop lifecycle
Stage 3+: LLM-powered extraction will be added as an optional enhancer.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .db import KortexDB
from .models import Episode, Fact, OpenLoop

logger = logging.getLogger(__name__)

_SALIENCE_KEYWORDS = {
    "never": 0.3,
    "always": 0.3,
    "promise": 0.4,
    "important": 0.3,
    "hate": 0.4,
    "love": 0.3,
    "angry": 0.5,
    "frustrated": 0.5,
    "sorry": 0.3,
    "deadline": 0.3,
    "urgent": 0.4,
    "idiot": 0.6,
    "stupid": 0.5,
    "amazing": 0.3,
    "terrible": 0.4,
    "fuck": 0.5,
    "shit": 0.4,
    "brilliant": 0.3,
    "awful": 0.4,
    "thank": 0.2,
    "please": 0.1,
    "remember": 0.3,
    "forget": 0.3,
    "wrong": 0.3,
    "mistake": 0.4,
    "perfect": 0.3,
    "disappoint": 0.5,
}

_NEGATIVE_PATTERNS = [
    re.compile(r"\b(?:hate|angry|frustrated|annoyed|upset|furious|pissed)\b", re.I),
    re.compile(r"\b(?:idiot|stupid|useless|terrible|awful|worst)\b", re.I),
    re.compile(r"\b(?:wrong|mistake|failed|broken|bug|crash)\b", re.I),
    re.compile(r"\b(?:fuck|shit|damn|hell|crap)\b", re.I),
]

_POSITIVE_PATTERNS = [
    re.compile(r"\b(?:love|thank|great|awesome|amazing|perfect|excellent)\b", re.I),
    re.compile(r"\b(?:brilliant|wonderful|fantastic|impressive|beautiful)\b", re.I),
    re.compile(r"\b(?:happy|glad|pleased|grateful|appreciate)\b", re.I),
]

_COMMITMENT_PATTERNS = [
    re.compile(r"\b(?:I will|I'll|I promise|I'?m going to|let me|I should)\b.*", re.I),
    re.compile(r"\b(?:we agreed|we decided|the plan is|next step)\b.*", re.I),
    re.compile(r"\b(?:remind me|don'?t forget|make sure)\b.*", re.I),
]

_QUESTION_PATTERNS = [
    re.compile(r"\b(?:can you|could you|would you|will you)\b.*\?", re.I),
    re.compile(r"\b(?:what|why|how|when|where|who)\b.*\?", re.I),
]

# --- Stage 2: Fact extraction patterns ---

# "I prefer X", "I like X", "I use X", "I'm a X", "I work with/on/at X"
_PREFERENCE_PATTERNS = [
    (
        re.compile(
            r"\bI (?:prefer|like|enjoy|love)\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)", re.I
        ),
        "prefers",
    ),
    (
        re.compile(
            r"\bI (?:hate|dislike|can'?t stand|despise)\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)",
            re.I,
        ),
        "dislikes",
    ),
    (
        re.compile(
            r"\bI (?:always|usually|typically)\s+(?:use|work with)\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)",
            re.I,
        ),
        "uses",
    ),
    (re.compile(r"\bI use\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)", re.I), "uses"),
    (
        re.compile(
            r"\bmy (?:favorite|favourite)\s+\w+\s+is\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)",
            re.I,
        ),
        "prefers",
    ),
]

# "I'm a developer", "I work at Google", "I live in London"
_IDENTITY_PATTERNS = [
    (
        re.compile(
            r"\bI(?:'m| am) (?:a |an )?(\w[\w\s]{2,30}?)(?:\.(?:\s|$)|,|!|\?|$)", re.I
        ),
        "is",
    ),
    (
        re.compile(r"\bI work (?:at|for)\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)", re.I),
        "works_at",
    ),
    (re.compile(r"\bI live in\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)", re.I), "lives_in"),
    (re.compile(r"\bmy name is\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)", re.I), "named"),
    (re.compile(r"\bcall me\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)", re.I), "named"),
    (
        re.compile(
            r"\bI (?:work on|'m working on|am working on)\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)",
            re.I,
        ),
        "works_on",
    ),
]

# "We decided to use X", "The project uses X", "We're going with X"
_PROJECT_PATTERNS = [
    (
        re.compile(
            r"\bwe (?:decided|agreed|chose) to\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)", re.I
        ),
        "decision",
    ),
    (
        re.compile(
            r"\b(?:the |our )?project (?:uses|is using|will use)\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)",
            re.I,
        ),
        "project_uses",
    ),
    (
        re.compile(
            r"\bwe'?re going (?:with|to use)\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)", re.I
        ),
        "decision",
    ),
    (
        re.compile(
            r"\b(?:the |our )?(?:stack|tech stack) (?:is|includes)\s+(.+?)(?:\.(?:\s|$)|,|!|\?|$)",
            re.I,
        ),
        "stack",
    ),
]

# "I'm a developer", "I work at Google", "I live in London"
_IDENTITY_PATTERNS = [
    (
        re.compile(r"\bI(?:'m| am) (?:a |an )?(\w[\w\s]{2,30}?)(?:\.|,|!|\?|$)", re.I),
        "is",
    ),
    (re.compile(r"\bI work (?:at|for)\s+(.+?)(?:\.|,|!|\?|$)", re.I), "works_at"),
    (re.compile(r"\bI live in\s+(.+?)(?:\.|,|!|\?|$)", re.I), "lives_in"),
    (re.compile(r"\bmy name is\s+(.+?)(?:\.|,|!|\?|$)", re.I), "named"),
    (re.compile(r"\bcall me\s+(.+?)(?:\.|,|!|\?|$)", re.I), "named"),
    (
        re.compile(
            r"\bI (?:work on|'m working on|am working on)\s+(.+?)(?:\.|,|!|\?|$)", re.I
        ),
        "works_on",
    ),
]

# "We decided to use X", "The project uses X", "We're going with X"
_PROJECT_PATTERNS = [
    (
        re.compile(r"\bwe (?:decided|agreed|chose) to\s+(.+?)(?:\.|,|!|\?|$)", re.I),
        "decision",
    ),
    (
        re.compile(
            r"\b(?:the |our )?project (?:uses|is using|will use)\s+(.+?)(?:\.|,|!|\?|$)",
            re.I,
        ),
        "project_uses",
    ),
    (
        re.compile(r"\bwe'?re going (?:with|to use)\s+(.+?)(?:\.|,|!|\?|$)", re.I),
        "decision",
    ),
    (
        re.compile(
            r"\b(?:the |our )?(?:stack|tech stack) (?:is|includes)\s+(.+?)(?:\.|,|!|\?|$)",
            re.I,
        ),
        "stack",
    ),
]

_FACT_STOPWORDS = frozenset(
    {
        "it",
        "that",
        "this",
        "them",
        "those",
        "these",
        "here",
        "there",
        "something",
        "anything",
        "nothing",
        "everything",
        "someone",
        "going to",
        "trying to",
        "able to",
        "sure",
        "not sure",
        "fine",
        "good",
        "okay",
        "ok",
        "yes",
        "no",
        "maybe",
    }
)

# Similarity threshold — words in common / total words to consider facts "similar"
_SIMILARITY_THRESHOLD = 0.5


class Ingestor:
    """Processes turns into structured episodes with heuristic metadata extraction."""

    def __init__(self, db: KortexDB):
        self._db = db
        self._lock = threading.Lock()
        self._turn_counter: dict[str, int] = {}

    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str,
        session_id: str = "",
        extract: bool = True,
    ) -> Episode:
        with self._lock:
            count = self._turn_counter.get(session_id, 0)
            self._turn_counter[session_id] = count + 1

        combined = f"{user_text} {assistant_text}"

        ep = Episode(
            session_id=session_id,
            turn_index=count,
            timestamp=datetime.now(timezone.utc),
            user_text=user_text[:4000],
            assistant_text=assistant_text[:4000],
        )

        if extract:
            ep.summary = self._extract_summary(user_text, assistant_text)
            ep.salience = self._score_salience(combined)
            ep.valence = self._score_valence(combined)
            ep.arousal = self._score_arousal(combined)
            ep.topics = self._extract_topics(combined)
            ep.entities = self._extract_entities(combined)

        ep.id = self._db.insert_episode(ep)
        return ep

    def extract_open_loops(self, user_text: str, episode_id: int) -> List[OpenLoop]:
        loops = []
        for pattern in _COMMITMENT_PATTERNS:
            match = pattern.search(user_text)
            if match:
                text = match.group(0).strip()[:300]
                loop = OpenLoop(
                    kind="commitment",
                    text=text,
                    source_episode_id=episode_id,
                )
                loop.id = self._db.insert_open_loop(loop)
                loops.append(loop)
                break

        for pattern in _QUESTION_PATTERNS:
            match = pattern.search(user_text)
            if match:
                text = match.group(0).strip()[:300]
                loop = OpenLoop(
                    kind="question",
                    text=text,
                    source_episode_id=episode_id,
                )
                loop.id = self._db.insert_open_loop(loop)
                loops.append(loop)
                break

        return loops

    def extract_facts(self, user_text: str, episode_id: int) -> List[Fact]:
        """Extract durable facts from user text and deduplicate against existing facts."""
        candidates = self._extract_fact_candidates(user_text)
        results = []

        for predicate, object_text in candidates:
            object_text = object_text.strip().rstrip(".,!?;:")
            if len(object_text) < 3 or len(object_text) > 200:
                continue
            if object_text.lower() in _FACT_STOPWORDS:
                continue

            existing = self._find_matching_fact(predicate, object_text)

            if existing:
                if self._facts_are_equivalent(existing.object_text, object_text):
                    new_conf = min(1.0, existing.confidence + 0.1)
                    self._db.update_fact_confidence(existing.id, new_conf)
                    self._db.bump_fact_last_seen(existing.id)
                    results.append(existing)
                else:
                    new_fact = Fact(
                        subject_type="user",
                        predicate=predicate,
                        object_text=object_text[:500],
                        confidence=0.6,
                        source_episode_id=episode_id,
                    )
                    new_fact.id = self._db.insert_fact(new_fact)
                    self._db.supersede_fact(existing.id, new_fact.id)
                    results.append(new_fact)
                    logger.debug(
                        "KORTEX fact superseded: [%s] '%s' -> '%s'",
                        predicate,
                        existing.object_text,
                        object_text,
                    )
            else:
                new_fact = Fact(
                    subject_type="user",
                    predicate=predicate,
                    object_text=object_text[:500],
                    confidence=0.5,
                    source_episode_id=episode_id,
                )
                new_fact.id = self._db.insert_fact(new_fact)
                results.append(new_fact)

        return results

    def resolve_answered_loops(self, assistant_text: str) -> List[OpenLoop]:
        """Auto-resolve open question loops if the assistant answered them."""
        resolved = []
        open_loops = self._db.get_open_loops(limit=20)

        for loop in open_loops:
            if loop.kind != "question":
                continue

            loop_keywords = set(
                w.strip(".,!?;:'\"()[]")
                for w in loop.text.lower().split()
                if len(w.strip(".,!?;:'\"()[]")) > 3
                and w.strip(".,!?;:'\"()[]")
                not in {
                    "what",
                    "when",
                    "where",
                    "which",
                    "would",
                    "could",
                    "should",
                    "will",
                    "that",
                    "this",
                    "have",
                    "does",
                    "with",
                }
            )

            if not loop_keywords:
                continue

            assistant_lower = assistant_text.lower()
            matches = sum(1 for kw in loop_keywords if kw in assistant_lower)
            if matches >= max(1, len(loop_keywords) * 0.4):
                self._db.resolve_loop(loop.id)
                resolved.append(loop)

        return resolved

    def resolve_completed_commitments(self, assistant_text: str) -> List[OpenLoop]:
        """Resolve commitment loops when assistant confirms completion."""
        _done_signals = re.compile(
            r"\b(?:done|completed|finished|fixed|deployed|resolved|implemented|shipped)\b",
            re.I,
        )
        if not _done_signals.search(assistant_text):
            return []

        resolved = []
        open_loops = self._db.get_open_loops(limit=20)

        for loop in open_loops:
            if loop.kind != "commitment":
                continue

            loop_keywords = set(
                w.strip(".,!?;:'\"()[]")
                for w in loop.text.lower().split()
                if len(w.strip(".,!?;:'\"()[]")) > 3
                and w.strip(".,!?;:'\"()[]")
                not in {
                    "will",
                    "going",
                    "should",
                    "would",
                    "promise",
                    "make",
                    "sure",
                    "that",
                    "this",
                    "with",
                }
            )

            if not loop_keywords:
                continue

            assistant_lower = assistant_text.lower()
            matches = sum(1 for kw in loop_keywords if kw in assistant_lower)
            if matches >= max(1, len(loop_keywords) * 0.4):
                self._db.resolve_loop(loop.id)
                resolved.append(loop)

        return resolved

    def _extract_fact_candidates(self, text: str) -> List[Tuple[str, str]]:
        candidates = []

        for pattern, predicate in _PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                candidates.append((predicate, match.group(1)))

        for pattern, predicate in _IDENTITY_PATTERNS:
            match = pattern.search(text)
            if match:
                candidates.append((predicate, match.group(1)))

        for pattern, predicate in _PROJECT_PATTERNS:
            match = pattern.search(text)
            if match:
                candidates.append((predicate, match.group(1)))

        return candidates

    def _find_matching_fact(self, predicate: str, object_text: str) -> Optional[Fact]:
        existing = self._db.get_facts_by_predicate(predicate, limit=10)
        for fact in existing:
            if self._facts_are_related(fact.object_text, object_text):
                return fact

        try:
            similar = self._db.find_similar_facts(
                object_text, predicate=predicate, limit=3
            )
            if similar:
                return similar[0]
        except Exception:
            pass

        return None

    @staticmethod
    def _facts_are_equivalent(existing: str, new: str) -> bool:
        """Two facts are equivalent if they say essentially the same thing."""
        existing_words = set(existing.lower().split())
        new_words = set(new.lower().split())
        if not existing_words or not new_words:
            return False
        intersection = existing_words & new_words
        union = existing_words | new_words
        jaccard = len(intersection) / len(union)
        return jaccard >= 0.7

    @staticmethod
    def _facts_are_related(existing: str, new: str) -> bool:
        """Two facts are related (same topic, possibly contradicting) if they share enough words."""
        existing_words = set(existing.lower().split())
        new_words = set(new.lower().split())
        if not existing_words or not new_words:
            return False
        intersection = existing_words & new_words
        smaller = min(len(existing_words), len(new_words))
        overlap = len(intersection) / max(smaller, 1)
        return overlap >= _SIMILARITY_THRESHOLD

    def _extract_summary(self, user_text: str, assistant_text: str) -> str:
        user_part = user_text[:200].strip()
        if len(user_text) > 200:
            user_part += "..."

        assistant_part = assistant_text[:150].strip()
        if len(assistant_text) > 150:
            assistant_part += "..."

        return f"User: {user_part} | Agent: {assistant_part}"

    def _score_salience(self, text: str) -> float:
        text_lower = text.lower()
        score = 0.0
        for keyword, weight in _SALIENCE_KEYWORDS.items():
            if keyword in text_lower:
                score = max(score, weight)

        word_count = len(text.split())
        if word_count > 200:
            score = max(score, 0.2)

        question_count = text.count("?")
        if question_count >= 3:
            score = max(score, 0.2)
        if text.count("!") >= 2:
            score = max(score, 0.25)

        caps_words = len(re.findall(r"\b[A-Z]{2,}\b", text))
        if caps_words >= 2:
            score = max(score, 0.3)

        return min(score, 1.0)

    def _score_valence(self, text: str) -> int:
        neg_count = sum(1 for p in _NEGATIVE_PATTERNS if p.search(text))
        pos_count = sum(1 for p in _POSITIVE_PATTERNS if p.search(text))

        if neg_count >= 3:
            return -2
        if neg_count >= 1 and pos_count == 0:
            return -1
        if pos_count >= 3:
            return 2
        if pos_count >= 1 and neg_count == 0:
            return 1
        return 0

    def _score_arousal(self, text: str) -> float:
        signals = 0.0
        if text.count("!") >= 2:
            signals += 0.3
        if text.count("?") >= 3:
            signals += 0.2
        caps = len(re.findall(r"\b[A-Z]{2,}\b", text))
        if caps >= 2:
            signals += 0.3
        if any(p.search(text) for p in _NEGATIVE_PATTERNS):
            signals += 0.2
        if len(text) > 500:
            signals += 0.1
        return min(signals, 1.0)

    def _extract_topics(self, text: str) -> str:
        topic_patterns = {
            "code": re.compile(
                r"\b(?:code|programming|function|class|variable|bug|debug)\b", re.I
            ),
            "design": re.compile(
                r"\b(?:design|UI|UX|layout|style|CSS|frontend)\b", re.I
            ),
            "infra": re.compile(
                r"\b(?:deploy|server|docker|kubernetes|CI|CD|pipeline)\b", re.I
            ),
            "data": re.compile(
                r"\b(?:database|SQL|query|schema|migration|table)\b", re.I
            ),
            "personal": re.compile(
                r"\b(?:feel|emotion|life|family|friend|relationship)\b", re.I
            ),
            "work": re.compile(
                r"\b(?:project|deadline|meeting|team|manager|sprint)\b", re.I
            ),
            "learning": re.compile(
                r"\b(?:learn|study|course|tutorial|understand|explain)\b", re.I
            ),
        }
        found = [name for name, pat in topic_patterns.items() if pat.search(text)]
        return ",".join(found[:5])

    def _extract_entities(self, text: str) -> str:
        entities = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        singles = re.findall(r"(?<!\.\s)\b([A-Z][a-z]{2,})\b", text)

        seen = set()
        result = []
        for e in entities + singles:
            low = e.lower()
            if low not in seen and low not in {
                "the",
                "this",
                "that",
                "user",
                "agent",
                "here",
            }:
                seen.add(low)
                result.append(e)
                if len(result) >= 5:
                    break

        return ",".join(result)
