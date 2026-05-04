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
import math
import re
import threading
from typing import Any, List, Optional, Tuple

from .db import DEFAULT_USER_ID, KortexDB
from .extract_llm import extract_structured_memory
from .linker import Linker
from .models import Episode, Fact, OpenLoop
from .semantic import SemanticSearch
from .time_utils import now_epoch

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

# Terminator: matches end-of-fact boundaries (period+space, comma, exclaim, question, or end-of-string)
_FACT_END = r'(?:\.(?:\s|$)|,|!|\?|$)'

# "I prefer X", "I like X", "I use X", "I'm a X", "I work with/on/at X"
_PREFERENCE_PATTERNS = [
    (re.compile(r'\bI (?:prefer|like|enjoy|love)\s+(.+?)' + _FACT_END, re.I), "prefers"),
    (re.compile(r'\bI (?:hate|dislike|cannot|despise)\s+(.+?)' + _FACT_END, re.I), "dislikes"),
    (re.compile(r'\bI (?:always|usually|typically)\s+(?:use|work with)\s+(.+?)' + _FACT_END, re.I), "uses"),
    (re.compile(r'\bI use\s+(.+?)' + _FACT_END, re.I), "uses"),
    (re.compile(r'\bmy (?:favorite|favourite)\s+\w+\s+is\s+(.+?)' + _FACT_END, re.I), "prefers"),
]

# "I'm a developer", "I work at Google", "I live in London"
_IDENTITY_PATTERNS = [
    (re.compile(r'\bI(?:\'m| am) (?:a |an )?((?:\w+(?:\s+\w+){1,10})(?=\s*[.,!?;:]\s*$|\s*$))', re.I), "is"),
    (re.compile(r'\bI work (?:at|for)\s+(.+?)' + _FACT_END, re.I), "works_at"),
    (re.compile(r'\bI live in\s+(.+?)' + _FACT_END, re.I), "lives_in"),
    (re.compile(r'\bmy name is\s+(.+?)' + _FACT_END, re.I), "named"),
    (re.compile(r'\bcall me\s+(.+?)' + _FACT_END, re.I), "named"),
    (re.compile(r'\bI (?:work on|\'m working on|am working on)\s+(.+?)' + _FACT_END, re.I), "works_on"),
]

# "We decided to use X", "The project uses X", "We're going with X"
_PROJECT_PATTERNS = [
    (re.compile(r'\bwe (?:decided|agreed|chose) to\s+(.+?)' + _FACT_END, re.I), "decision"),
    (re.compile(r'\b(?:the |our )?project (?:uses|is using|will use)\s+(.+?)' + _FACT_END, re.I), "project_uses"),
    (re.compile(r'\bwe\'re going (?:with|to use)\s+(.+?)' + _FACT_END, re.I), "decision"),
    (re.compile(r'\b(?:the |our )?(?:stack|tech stack) (?:is|includes)\s+(.+?)' + _FACT_END, re.I), "stack"),
]

_FACT_STOPWORDS = frozenset({
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
        # Specific garbage phrases that produced junk facts
        "going bed",
        "going to bed",
        "going to sleep",
        "going to lunch",
        "going to dinner",
        "going to work",
        "coming home",
        "just kidding",
        "just joking",
        "ngl",
        "rn",
        "ikr",
    })

# Similarity threshold — words in common / total words to consider facts "similar"
_SIMILARITY_THRESHOLD = 0.5


def _extract_keywords(text: str, stopwords: frozenset) -> set[str]:
    """Extract significant keywords from text, filtering short words and stopwords."""
    return {
        w.strip(".,!?;:'\"()[]")
        for w in text.lower().split()
        if len(w.strip(".,!?;:'\"()[]")) > 3
        and w.strip(".,!?;:'\"()[]") not in stopwords
    }


_ACTION_VERB_PATTERN = re.compile(
    r"\b(?:fix|deploy|build|create|update|change|add|remove|edit|write|read|search|find|"
    r"check|test|run|install|configure|setup|generate|parse|format|validate|verify|"
    r"confirm|approve|reject|enable|disable|start|stop|pause|resume|continue|finish|"
    r"complete|resolve|solve|work|break|crash|error|issue|problem|solution|result|"
    r"output|input|data|file|directory|folder|path|name|type|value|key|variable|"
    r"function|class|method|property|attribute|parameter|argument|return|yield|import|"
    r"export|module|package|library|dependency)\b"
)

_TOPIC_CATEGORIES = {
    "code": ["code", "programming", "function", "class", "variable", "bug", "debug", "python", "javascript", "typescript", "java", "rust", "go", "ruby", "swift", "kotlin", "react", "vue", "angular", "django", "flask", "spring", "rails", "laravel", "express", "fastapi", "api", "endpoint", "route", "middleware", "controller", "model", "view", "component", "module", "package", "library", "framework", "dependency", "inheritance", "polymorphism", "encapsulation", "abstraction", "oop", "functional", "async", "await", "promise", "callback", "closure", "scope", "context", "state", "reactive", "immutable", "mutable", "typing", "annotation", "decorator", "generator", "iterator", "comprehension"],
    "design": ["design", "ui", "ux", "layout", "style", "css", "frontend", "responsive", "mobile", "desktop", "tablet", "screen", "display", "monitor", "resolution", "pixel", "vector", "raster", "image", "photo", "graphic", "illustration", "icon", "logo", "brand", "typography", "font", "color", "palette", "theme", "dark", "light", "mode", "transition", "animation", "keyframe", "easing", "duration", "delay", "timing", "speed", "pace", "rhythm", "flow", "grid", "flex", "flexbox", "container", "wrapper", "section", "header", "footer", "sidebar", "nav", "navigation", "menu", "dropdown", "accordion", "tab", "panel", "card", "modal", "dialog", "popup", "tooltip", "badge", "tag", "label", "input", "field", "form", "button", "link", "anchor"],
    "infra": ["deploy", "server", "docker", "kubernetes", "ci", "cd", "pipeline", "cloud", "aws", "azure", "gcp", "ec2", "s3", "lambda", "ecs", "eks", "fargate", "container", "registry", "compose", "swarm", "orchestration", "service", "mesh", "gateway", "proxy", "nginx", "apache", "tomcat", "jetty", "pod", "replica", "deployment", "rollout", "rollback", "canary", "blue-green", "feature flag", "config", "environment", "staging", "production", "development", "local", "remote", "ssh", "rsync", "git", "version control", "branch", "merge", "rebase", "commit", "push", "pull", "clone", "fork", "pr", "pull request", "issue", "ticket", "sprint", "epic", "story", "milestone", "release", "tag", "changelog", "readme", "docs", "documentation", "wiki", "blog", "post", "article", "tutorial", "guide", "recipe", "anti-pattern", "best practice", "convention", "standard", "specification", "composition", "delegation", "injection", "di", "ioc", "mvc", "mvvm", "flux", "redux", "state management", "store", "reducer", "selector", "plugin", "extension", "addon"],
    "data": ["database", "sql", "query", "schema", "migration", "table", "row", "column", "field", "record", "entry", "item", "element", "node", "tree", "graph", "list", "queue", "stack", "heap", "set", "map", "hash", "sort", "search", "filter", "reduce", "slice", "splice", "concat", "join", "split", "replace", "match", "regex", "pattern", "template", "literal", "string", "number", "boolean", "null", "undefined", "object", "array", "symbol", "bigint", "promise", "async", "await", "callback", "closure", "scope", "context", "state", "reactive", "immutable", "mutable", "type", "typing", "annotation", "decorator", "generator", "iterator", "comprehension"],
    "personal": ["feel", "emotion", "life", "family", "friend", "relationship", "love", "hate", "happy", "sad", "angry", "excited", "nervous", "calm", "energetic", "tired", "focused", "distracted", "creative", "analytical", "intuitive", "logical", "emotional", "rational"],
    "work": ["project", "deadline", "meeting", "team", "manager", "sprint", "task", "issue", "ticket", "story", "epic", "milestone", "release", "version", "changelog", "readme", "docs", "documentation", "wiki", "blog", "post", "article", "tutorial", "guide", "recipe", "pattern", "anti-pattern", "best practice", "convention", "standard", "specification", "interface", "abstraction", "encapsulation", "inheritance", "polymorphism", "composition", "delegation", "injection", "container", "di", "ioc", "mvc", "mvvm", "flux", "redux", "state management", "store", "reducer", "selector", "middleware", "plugin", "extension", "addon", "module", "package", "library", "framework", "dependency", "import", "export", "require", "use", "inject", "provide", "consume", "produce", "event", "listener", "observer", "subscriber", "publisher", "dispatcher", "emitter", "signal", "broadcast", "notify", "update", "refresh", "reload", "restart", "start", "stop", "pause", "resume", "continue", "break"],
    "learning": ["learn", "study", "course", "tutorial", "understand", "explain", "concept", "principle", "theory", "model", "framework", "paradigm", "pattern", "anti-pattern", "best practice", "convention", "standard", "specification", "interface", "abstraction", "encapsulation", "inheritance", "polymorphism", "composition", "delegation", "injection", "container", "di", "ioc", "mvc", "mvvm", "flux", "redux", "state management", "store", "reducer", "selector", "middleware", "plugin", "extension", "addon", "module", "package", "library", "framework", "dependency", "import", "export", "require", "use", "inject", "provide", "consume", "produce", "event", "listener", "observer", "subscriber", "publisher", "dispatcher", "emitter", "signal", "broadcast", "notify", "update", "refresh", "reload", "restart", "start", "stop", "pause", "resume", "continue", "break"],
    "memory": ["memory", "recall", "remember", "forget", "context", "history", "past", "present", "future", "timeline", "sequence", "order", "step", "stage", "phase", "level", "layer", "tier", "group", "category", "type", "kind", "sort", "classify", "categorize", "organize", "structure", "pattern", "rule", "constraint", "limit", "boundary", "edge", "case", "exception", "special", "unique", "common", "frequent", "rare", "typical", "atypical", "normal", "abnormal", "standard", "non-standard", "convention", "custom", "default", "fallback", "backup", "primary", "secondary", "tertiary", "main", "core", "central", "peripheral", "threshold", "minimum", "maximum", "range", "span", "interval", "duration", "time", "period", "epoch", "timestamp", "date", "day", "week", "month", "year", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond"],
    "security": ["security", "auth", "authentication", "authorization", "permission", "role", "scope", "context", "session", "token", "cookie", "header", "query", "parameter", "argument", "return", "yield", "throw", "catch", "try", "finally", "exception", "error", "warning", "log", "console", "print", "debugger", "breakpoint", "step", "next", "previous", "current", "last", "first", "index", "key", "value", "pair", "entry", "item", "element", "node", "tree", "graph", "list", "queue", "stack", "heap", "set", "map", "hash", "sort", "search", "filter", "reduce", "map", "flat", "slice", "splice", "concat", "join", "split", "replace", "match", "regex", "pattern", "template", "literal", "string", "number", "boolean", "null", "undefined", "object", "array", "symbol", "bigint", "promise", "async", "await", "callback", "closure", "scope", "context", "state", "reactive", "immutable", "mutable", "type", "typing", "annotation", "decorator", "generator", "iterator", "comprehension"],
    "testing": ["test", "unit", "integration", "e2e", "bvt", "smoke", "regression", "performance", "load", "stress", "chaos", "coverage", "mock", "stub", "spy", "fixture", "assertion", "expectation", "result", "output", "input", "data", "file", "directory", "folder", "path", "name", "type", "value", "key", "variable", "function", "class", "method", "property", "attribute", "parameter", "argument", "return", "yield", "throw", "catch", "try", "finally", "exception", "error", "warning", "log", "console", "print", "debugger", "breakpoint", "step", "next", "previous", "current", "last", "first", "index", "key", "value", "pair", "entry", "item", "element", "node", "tree", "graph", "list", "queue", "stack", "heap", "set", "map", "hash", "sort", "search", "filter", "reduce", "map", "flat", "slice", "splice", "concat", "join", "split", "replace", "match", "regex", "pattern", "template", "literal", "string", "number", "boolean", "null", "undefined", "object", "array", "symbol", "bigint", "promise", "async", "await", "callback", "closure", "scope", "context", "state", "reactive", "immutable", "mutable", "type", "typing", "annotation", "decorator", "generator", "iterator", "comprehension"],
    "devops": ["devops", "ci", "cd", "pipeline", "deploy", "server", "docker", "kubernetes", "cloud", "aws", "azure", "gcp", "ec2", "s3", "lambda", "ecs", "eks", "fargate", "container", "image", "registry", "compose", "swarm", "orchestration", "service", "mesh", "gateway", "proxy", "nginx", "apache", "tomcat", "jetty", "pod", "replica", "deployment", "rollout", "rollback", "canary", "blue-green", "feature flag", "config", "environment", "staging", "production", "development", "local", "remote", "ssh", "rsync", "git", "version control", "branch", "merge", "rebase", "commit", "push", "pull", "clone", "fork", "pr", "pull request", "issue", "ticket", "sprint", "epic", "story", "task", "milestone", "release", "tag", "version", "changelog", "readme", "docs", "documentation", "wiki", "blog", "post", "article", "tutorial", "guide", "recipe", "pattern", "anti-pattern", "best practice", "convention", "standard", "specification", "interface", "abstraction", "encapsulation", "inheritance", "polymorphism", "composition", "delegation", "injection", "container", "di", "ioc", "mvc", "mvvm", "flux", "redux", "state management", "store", "reducer", "selector", "middleware", "plugin", "extension", "addon", "module", "package", "library", "framework", "dependency", "import", "export", "require", "use", "inject", "provide", "consume", "produce", "event", "listener", "observer", "subscriber", "publisher", "dispatcher", "emitter", "signal", "broadcast", "notify", "update", "refresh", "reload", "restart", "start", "stop", "pause", "resume", "continue", "break"],
    "ai": ["ai", "ml", "dl", "nn", "transformer", "attention", "embedding", "token", "context", "window", "batch", "epoch", "gradient", "loss", "metric", "accuracy", "precision", "recall", "f1", "auc", "roc", "confusion", "matrix", "feature", "label", "target", "input", "output", "data", "file", "directory", "folder", "path", "name", "type", "value", "key", "variable", "function", "class", "method", "property", "attribute", "parameter", "argument", "return", "yield", "throw", "catch", "try", "finally", "exception", "error", "warning", "log", "console", "print", "debugger", "breakpoint", "step", "next", "previous", "current", "last", "first", "index", "key", "value", "pair", "entry", "item", "element", "node", "tree", "graph", "list", "queue", "stack", "heap", "set", "map", "hash", "sort", "search", "filter", "reduce", "map", "flat", "slice", "splice", "concat", "join", "split", "replace", "match", "regex", "pattern", "template", "literal", "string", "number", "boolean", "null", "undefined", "object", "array", "symbol", "bigint", "promise", "async", "await", "callback", "closure", "scope", "context", "state", "reactive", "immutable", "mutable", "type", "typing", "annotation", "decorator", "generator", "iterator", "comprehension"],
}


class Ingestor:
    """Processes turns into structured episodes with heuristic metadata extraction."""

    def __init__(self, db: KortexDB):
        self._db = db
        self._linker = Linker(db)
        self._lock = threading.Lock()
        self._turn_counter: dict[str, int] = {}
        self._auxiliary_client: Any = None
        self._extraction_mode: str = "heuristic"
        self._semantic: Optional[SemanticSearch] = None

    def configure_extraction(
        self, mode: str = "heuristic", auxiliary_client: Any = None
    ) -> None:
        self._extraction_mode = mode or "heuristic"
        self._auxiliary_client = auxiliary_client

    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str,
        session_id: str = "",
        user_id: str = "__default__",
        extract: bool = True,
    ) -> Episode:
        with self._lock:
            count = self._turn_counter.get(session_id, 0)
            self._turn_counter[session_id] = count + 1

        combined = f"{user_text} {assistant_text}"

        ep = Episode(
            user_id=user_id,
            session_id=session_id,
            turn_index=count,
            timestamp=now_epoch(),
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
            structured = self._extract_structured_memory(user_text, assistant_text)
            if structured:
                if structured.get("summary"):
                    ep.summary = structured["summary"]
                if structured.get("topics"):
                    ep.topics = ",".join(structured["topics"][:5])
                if structured.get("entities"):
                    ep.entities = ",".join(structured["entities"][:5])

        ep.id = self._db.insert_episode(ep)

        # Auto-embed episode for vector similarity search
        self._embed_episode(ep)

        logger.info(
            "[episode] id=%d session=%s user=%s salience=%.2f topics=%s",
            ep.id, session_id, user_id, ep.salience, ep.topics,
        )
        return ep

    def _get_semantic(self) -> SemanticSearch:
        """Lazily create SemanticSearch instance."""
        if self._semantic is None:
            self._semantic = SemanticSearch(self._db)
        return self._semantic

    def _embed_episode(self, ep: Episode) -> None:
        """Generate and store TF-IDF embedding for an episode."""
        try:
            search = self._get_semantic()
            search.embed_episode(
                ep.id, ep.user_text, ep.assistant_text, user_id=ep.user_id
            )
        except Exception:
            logger.debug("Embedding failed for episode %d", ep.id, exc_info=True)

    def _embed_fact(self, fact: Fact) -> None:
        """Generate and store TF-IDF embedding for a fact."""
        try:
            search = self._get_semantic()
            search.embed_fact(fact.id, fact.object_text, user_id=fact.user_id)
        except Exception:
            logger.debug("Embedding failed for fact %d", fact.id, exc_info=True)

    def extract_open_loops(
        self, user_text: str, episode_id: int, user_id: str = "__default__"
    ) -> List[OpenLoop]:
        structured = self._extract_structured_memory(user_text, "")
        if structured and structured.get("open_loops"):
            loops = []
            for item in structured["open_loops"]:
                loop = OpenLoop(
                    user_id=user_id,
                    kind=item.get("kind", "question"),
                    text=item.get("text", "")[:300],
                    source_episode_id=episode_id,
                )
                loop.id = self._db.insert_open_loop(loop)
                loops.append(loop)
            if loops:
                return loops

        loops = []
        for pattern in _COMMITMENT_PATTERNS:
            match = pattern.search(user_text)
            if match:
                text = match.group(0).strip()[:300]
                loop = OpenLoop(
                    user_id=user_id,
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
                    user_id=user_id,
                    kind="question",
                    text=text,
                    source_episode_id=episode_id,
                )
                loop.id = self._db.insert_open_loop(loop)
                loops.append(loop)
                logger.info(
                    "[loop] episode=%d kind=%s text=%s",
                    episode_id, loop.kind, loop.text[:60],
                )
                break

        if loops:
            logger.info("[loops] episode=%d extracted=%d", episode_id, len(loops))
        return loops

    def extract_facts(
        self, user_text: str, episode_id: int, user_id: str = "__default__"
    ) -> List[Fact]:
        """Extract durable facts from user text and deduplicate against existing facts."""
        structured = self._extract_structured_memory(user_text, "")
        candidates = self._extract_fact_candidates(user_text)
        if structured and structured.get("facts"):
            merged = candidates + structured["facts"]
            candidates = []
            seen = set()
            for predicate, object_text in merged:
                key = (predicate.lower(), object_text.lower())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((predicate, object_text))
        results = []

        for predicate, object_text in candidates:
            object_text = object_text.strip().rstrip(".,!?;:")
            if len(object_text) < 1 or len(object_text) > 200:
                continue
            if object_text.lower() in _FACT_STOPWORDS:
                continue
            # Filter garbage identity phrases like "going to bed", "trying to fix"
            lower = object_text.lower()
            if any(
                lower.startswith(prefix)
                for prefix in (
                    "going to ",
                    "trying to ",
                    "about to ",
                    "ready to ",
                    "going ",
                    "trying ",
                    "about ",
                )
            ):
                continue

            existing = self._find_matching_fact(predicate, object_text, user_id=user_id)

            if existing:
                if self._facts_are_equivalent(existing.object_text, object_text):
                    new_conf = min(1.0, existing.confidence + 0.1)
                    self._db.update_fact_confidence(existing.id, new_conf)
                    self._db.bump_fact_last_seen(existing.id)
                    results.append(existing)
                else:
                    new_fact = Fact(
                        user_id=user_id,
                        subject_type="user",
                        subject_id=user_id,
                        predicate=predicate,
                        object_text=object_text[:500],
                        confidence=0.6,
                        source_episode_id=episode_id,
                    )
                    new_fact.id = self._db.insert_fact(new_fact)
                    self._apply_fact_conflict(existing, new_fact)
                    results.append(new_fact)
                    self._embed_fact(new_fact)
                    logger.debug(
                        "KORTEX fact superseded: [%s] '%s' -> '%s'",
                        predicate,
                        existing.object_text,
                        object_text,
                    )
            else:
                new_fact = Fact(
                    user_id=user_id,
                    subject_type="user",
                    subject_id=user_id,
                    predicate=predicate,
                    object_text=object_text[:500],
                    confidence=0.5,
                    source_episode_id=episode_id,
                )
                new_fact.id = self._db.insert_fact(new_fact)
                results.append(new_fact)
                self._embed_fact(new_fact)
                logger.info(
                    "[fact] episode=%d predicate=%s object=%s confidence=%.2f",
                    episode_id, predicate, object_text[:80], new_fact.confidence,
                )

        if results:
            logger.info("[facts] episode=%d extracted=%d", episode_id, len(results))
        return results

    def resolve_answered_loops(
        self,
        assistant_text: str,
        resolving_episode_id: Optional[int] = None,
        user_id: str = "__default__",
    ) -> List[OpenLoop]:
        """Auto-resolve open question loops if the assistant answered them."""
        resolved = []
        open_loops = self._db.get_open_loops(limit=20, user_id=user_id)

        for loop in open_loops:
            if loop.kind != "question":
                continue

            loop_keywords = _extract_keywords(
                loop.text,
                frozenset({"what", "when", "where", "which", "would", "could",
                           "should", "will", "that", "this", "have", "does", "with"}),
            )

            if not loop_keywords:
                continue

            assistant_lower = assistant_text.lower()
            matches = sum(1 for kw in loop_keywords if kw in assistant_lower)
            if matches >= max(1, len(loop_keywords) * 0.4):
                resolution = self._build_loop_resolution(assistant_text, loop_keywords)
                self._db.resolve_loop(
                    loop.id,
                    resolution=resolution,
                    resolved_by_episode_id=resolving_episode_id,
                )
                resolved.append(loop)

        return resolved

    def resolve_completed_commitments(
        self,
        assistant_text: str,
        resolving_episode_id: Optional[int] = None,
        user_id: str = "__default__",
    ) -> List[OpenLoop]:
        """Resolve commitment loops when assistant confirms completion."""
        _done_signals = re.compile(
            r"\b(?:done|completed|finished|fixed|deployed|resolved|implemented|shipped)\b",
            re.I,
        )
        if not _done_signals.search(assistant_text):
            return []

        resolved = []
        open_loops = self._db.get_open_loops(limit=20, user_id=user_id)

        for loop in open_loops:
            if loop.kind != "commitment":
                continue

            loop_keywords = _extract_keywords(
                loop.text,
                frozenset({"will", "going", "should", "would", "promise",
                           "make", "sure", "that", "this", "with"}),
            )

            if not loop_keywords:
                continue

            assistant_lower = assistant_text.lower()
            matches = sum(1 for kw in loop_keywords if kw in assistant_lower)
            if matches >= max(1, len(loop_keywords) * 0.4):
                resolution = self._build_loop_resolution(assistant_text, loop_keywords)
                self._db.resolve_loop(
                    loop.id,
                    resolution=resolution,
                    resolved_by_episode_id=resolving_episode_id,
                )
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

    def _extract_structured_memory(
        self, user_text: str, assistant_text: str
    ) -> Optional[dict]:
        if self._auxiliary_client is None:
            return None
        # Use LLM extraction whenever auxiliary_client is available,
        # regardless of extraction mode (augments heuristic patterns)
        try:
            structured = extract_structured_memory(
                user_text,
                assistant_text,
                auxiliary_client=self._auxiliary_client,
            )
        except Exception as e:
            logger.warning("[llm] extraction failed: %s", e)
            return None
        if structured:
            has_facts = bool(structured.get("facts"))
            has_loops = bool(structured.get("open_loops"))
            has_reflections = bool(structured.get("reflections"))
            logger.info(
                "[llm] extracted facts=%d loops=%d reflections=%d",
                len(structured.get("facts", [])),
                len(structured.get("open_loops", [])),
                len(structured.get("reflections", [])),
            )
            return structured
        return None

    @staticmethod
    def _build_loop_resolution(assistant_text: str, loop_keywords: set[str]) -> str:
        clean_text = assistant_text.strip()
        if not clean_text:
            return "resolved from assistant response"

        fragments = [kw for kw in sorted(loop_keywords) if kw in clean_text.lower()][:3]
        if fragments:
            return f"Resolved via assistant response about: {', '.join(fragments)}"
        return f"Resolved via assistant response: {clean_text[:160]}"

    def _find_matching_fact(
        self, predicate: str, object_text: str, user_id: str = "__default__"
    ) -> Optional[Fact]:
        # Check by predicate with improved matching
        existing = self._db.get_facts_by_predicate(predicate, limit=10, user_id=user_id)
        for fact in existing:
            if self._facts_are_equivalent(fact.object_text, object_text):
                return fact
            if self._facts_are_related(fact.object_text, object_text):
                return fact

        # FTS search for similar content across all predicates
        try:
            similar = self._db.find_similar_facts(
                object_text, predicate=predicate, limit=3, user_id=user_id
            )
            if similar:
                for fact in similar:
                    if self._facts_are_equivalent(fact.object_text, object_text):
                        return fact
        except Exception:
            pass

        return None

    def _apply_fact_conflict(self, existing: Fact, new_fact: Fact) -> None:
        if not existing.id or not new_fact.id:
            return

        contradictory = self._facts_contradict(
            existing.predicate, existing.object_text, new_fact.object_text
        )
        self._db.supersede_fact(existing.id, new_fact.id)
        self._linker.link_superseded_facts(existing.id, new_fact.id)
        if contradictory:
            self._db.mark_fact_contradiction(existing.id, new_fact.id)
            self._linker.link_contradicting_facts(existing.id, new_fact.id)

    @classmethod
    def _facts_contradict(cls, predicate: str, existing: str, new: str) -> bool:
        existing_lower = existing.lower()
        new_lower = new.lower()

        negation_markers = (
            "no longer",
            "not anymore",
            "stopped",
            "quit",
            "instead",
            "switched",
            "now",
        )
        if any(marker in new_lower for marker in negation_markers):
            return True

        if predicate in {
            "uses",
            "decision",
            "project_uses",
            "works_at",
            "lives_in",
            "named",
        }:
            if existing_lower != new_lower and cls._facts_are_related(existing, new):
                return True

        if predicate == "prefers":
            existing_terms = set(existing_lower.split())
            new_terms = set(new_lower.split())
            if existing_terms and new_terms and existing_terms.isdisjoint(new_terms):
                return True

        existing_version = cls._extract_version(existing_lower)
        new_version = cls._extract_version(new_lower)
        if existing_version and new_version and existing_version != new_version:
            return True

        return False

    @staticmethod
    def _extract_version(text: str) -> Optional[Tuple[int, ...]]:
        match = re.search(r"\b(\d+(?:\.\d+)+)\b", text)
        if not match:
            return None
        return tuple(int(part) for part in match.group(1).split("."))

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison: lowercase, strip ASCII punctuation, preserve emoji, collapse spaces."""
        import re
        import string
        text = text.lower().strip()
        # Strip ASCII punctuation only (preserves Unicode/emoji characters)
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _facts_are_equivalent(existing: str, new: str) -> bool:
        """Two facts are equivalent if they say essentially the same thing.

        Uses a three-pass approach:
        1. Length ratio quick filter (skip obviously different facts)
        2. Word-level Jaccard (catches near-identical facts)
        3. Bigram Jaccard (catches reworded facts)
        4. Trigram Jaccard (catches slightly shifted phrasing)

        Thresholds were tuned empirically against real fact pairs:
        - Word Jaccard ≥ 0.65 (was 0.7, too strict for short facts)
        - Bigram Jaccard ≥ 0.45 (was 0.5, too strict for 3-word facts)
        - Trigram Jaccard ≥ 0.4 (new, catches "uses python daily" vs "uses python every day")
        """
        existing_clean = Ingestor._normalize_text(existing)
        new_clean = Ingestor._normalize_text(new)

        # Quick check: if lengths differ by more than 50%, they're probably different
        if len(existing_clean) > 0 and len(new_clean) > 0:
            length_ratio = min(len(existing_clean), len(new_clean)) / max(len(existing_clean), len(new_clean))
            if length_ratio < 0.4:
                return False

        # Word-level Jaccard
        existing_words = set(existing_clean.split())
        new_words = set(new_clean.split())
        if not existing_words or not new_words:
            return False
        intersection = existing_words & new_words
        union = existing_words | new_words
        word_jaccard = len(intersection) / len(union)

        # If word-level is strong enough, call it a match
        if word_jaccard >= 0.65:
            return True

        # Bigram-level Jaccard (catches reworded facts)
        existing_bigrams = set()
        new_bigrams = set()
        words_e = existing_clean.split()
        words_n = new_clean.split()
        for i in range(len(words_e) - 1):
            existing_bigrams.add((words_e[i], words_e[i+1]))
        for i in range(len(words_n) - 1):
            new_bigrams.add((words_n[i], words_n[i+1]))

        if existing_bigrams and new_bigrams:
            bigram_intersection = existing_bigrams & new_bigrams
            bigram_union = existing_bigrams | new_bigrams
            bigram_jaccard = len(bigram_intersection) / len(bigram_union)
            if bigram_jaccard >= 0.45:
                return True

        # Trigram-level Jaccard (catches slightly shifted phrasing)
        existing_trigrams = set()
        new_trigrams = set()
        for i in range(len(words_e) - 2):
            existing_trigrams.add((words_e[i], words_e[i+1], words_e[i+2]))
        for i in range(len(words_n) - 2):
            new_trigrams.add((words_n[i], words_n[i+1], words_n[i+2]))

        if existing_trigrams and new_trigrams:
            trigram_intersection = existing_trigrams & new_trigrams
            trigram_union = existing_trigrams | new_trigrams
            trigram_jaccard = len(trigram_intersection) / len(trigram_union)
            if trigram_jaccard >= 0.4:
                return True

        return False

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

    @staticmethod
    def _score_salience(text: str) -> float:
        """Score salience using a composite heuristic:
        - Emotional intensity
        - Action density (verbs)
        - Structural signals (length, punctuation, caps)
        - Information density
        """
        text_lower = text.lower()
        score = 0.0

        # 1. Emotional keywords (existing approach)
        for keyword, weight in _SALIENCE_KEYWORDS.items():
            if keyword in text_lower:
                score = max(score, weight)

        # 2. Action density - count action verbs
        action_verbs = _ACTION_VERB_PATTERN.findall(text_lower)
        if len(action_verbs) >= 3:
            score = max(score, 0.3)
        elif len(action_verbs) >= 5:
            score = max(score, 0.4)
        elif len(action_verbs) >= 8:
            score = max(score, 0.5)

        # 3. Structural signals
        word_count = len(text.split())
        if word_count > 200:
            score = max(score, 0.2)
        if text.count('?') >= 3:
            score = max(score, 0.2)
        if text.count('!') >= 2:
            score = max(score, 0.25)
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        if caps_words >= 2:
            score = max(score, 0.3)

        # 4. Information density (unique words / total)
        words = text_lower.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.7:
                score = max(score, 0.3)
            elif unique_ratio > 0.5:
                score = max(score, 0.2)

        # 5. Commitment signals
        if re.search(r'\b(?:I will|I\'ll|I promise|I\'m going to|let me|I should)\b', text_lower):
            score = max(score, 0.35)
        if re.search(r'\b(?:we agreed|we decided|the plan is|next step)\b', text_lower):
            score = max(score, 0.35)
        if re.search(r'\b(?:remind me|don\'t forget|make sure)\b', text_lower):
            score = max(score, 0.35)

            score = max(score, 0.5)
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
        """Extract topics using a comprehensive set of category patterns.
        Returns comma-separated topics, prioritized by relevance."""
        text_lower = text.lower()
        topics = []

        topic_categories = _TOPIC_CATEGORIES

        # Check each category
        for category, keywords in topic_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics.append(category)
                    break

        # Deduplicate and return top 5
        seen = set()
        result = []
        for t in topics:
            if t not in seen:
                seen.add(t)
                result.append(t)
                if len(result) >= 5:
                    break

        return ",".join(result)

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
