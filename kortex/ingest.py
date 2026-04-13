"""Turn ingestion pipeline for KORTEX.

Writes raw episodes on every sync_turn(), then extracts metadata
(summary, salience, valence, entities, topics) either inline via
heuristics or asynchronously via LLM extraction.

Stage 1: Heuristic extraction (no LLM dependency)
Stage 3+: LLM-powered extraction will be added as an optional enhancer.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from typing import List, Optional

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
        # Capitalized multi-word sequences (crude NER)
        entities = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        # Single capitalized words that aren't sentence starters (rough heuristic)
        singles = re.findall(r"(?<!\.\s)\b([A-Z][a-z]{2,})\b", text)

        # Deduplicate, take top 5
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
