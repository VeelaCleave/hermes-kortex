"""Topic classification for KORTEX facts.

Classifies facts into hierarchical topics:
- Domain (broad): Personal, Technical, Project, Relationship
- Category (mid-level): e.g. "Tech Stack", "Emotional State", "Goals"
- Topic (specific): e.g. "Python", "Docker", "Kortex Memory"

Uses TF-IDF + keyword matching for lightweight classification.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ── Hierarchy definitions ──────────────────────────────────────────────────

DOMAINS = {
    "personal": "Personal life, identity, emotions, habits",
    "technical": "Tech stack, tools, programming, infrastructure",
    "project": "Active projects, repos, milestones",
    "relationship": "Dynamic between user and agent",
    "goal": "Goals, aspirations, commitments",
    "preference": "Likes, dislikes, style choices",
}

CATEGORIES = {
    "identity": ("personal", ["name", "role", "identity", "called", "known as"]),
    "emotions": ("personal", ["feeling", "mood", "emotion", "state", "tired", "frustrated", "happy", "overwhelmed", "lost"]),
    "habits": ("personal", ["habit", "routine", "daily", "usually", "often", "sometimes"]),
    "health": ("personal", ["sleep", "energy", "focus", "attention", "ADHD", "burnout"]),
    "tech_stack": ("technical", ["python", "docker", "git", "linux", "node", "typescript", "react", "postgres", "redis", "nginx", "aws", "vps", "server"]),
    "tools": ("technical", ["tool", "cli", "IDE", "editor", "vscode", "neovim", "tmux", "zsh", "bash"]),
    "infrastructure": ("technical", ["deploy", "pipeline", "CI/CD", "kubernetes", "terraform", "ansible", "infra"]),
    "security": ("technical", ["auth", "token", "secret", "key", "password", "encryption"]),
    "project_status": ("project", ["progress", "milestone", "sprint", "deadline", "release", "version", "commit"]),
    "project_scope": ("project", ["scope", "feature", "module", "component", "endpoint", "API"]),
    "relationship_dynamic": ("relationship", ["trust", "rapport", "communication", "feedback", "dynamic"]),
    "goals": ("goal", ["goal", "want", "plan", "aim", "aspire", "target", "objective"]),
    "preferences": ("preference", ["prefer", "like", "love", "hate", "favor", "style", "format"]),
}

# Keyword → category mapping for quick classification
_CATEGORY_KEYWORD_TO_CAT: Dict[str, str] = {}

def _build_keyword_map():
    for category, (domain, keywords) in CATEGORIES.items():
        for kw in keywords:
            _CATEGORY_KEYWORD_TO_CAT[kw] = category

_build_keyword_map()


def classify_fact(object_text: str, predicate: str = "") -> Dict[str, str]:
    """Classify a fact into domain/category/topic.
    
    Returns: {"domain": ..., "category": ..., "topic": ...}
    """
    text = (predicate + " " + object_text).lower()
    
    # Score each category
    scores = {}
    for category, (domain, keywords) in CATEGORIES.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        if score > 0:
            scores[category] = score
    
    if scores:
        best_category = max(scores, key=scores.get)
        domain = CATEGORIES[best_category][0]
    else:
        # Default classification
        domain = _guess_domain(text)
        best_category = _guess_category(text, domain)
    
    # Extract specific topic
    topic = _extract_topic(object_text, best_category)
    
    return {
        "domain": domain,
        "category": best_category,
        "topic": topic or best_category,
    }


def _guess_domain(text: str) -> str:
    """Guess the broad domain from text content."""
    project_signals = ["project", "repo", "branch", "commit", "PR", "issue", "ticket", "sprint"]
    if any(s in text for s in project_signals):
        return "project"
    
    tech_signals = ["code", "API", "database", "server", "deploy", "config", "endpoint"]
    if any(s in text for s in tech_signals):
        return "technical"
    
    goal_signals = ["want", "plan", "goal", "aim", "target", "objective", "try to", "need to"]
    if any(s in text for s in goal_signals):
        return "goal"
    
    emotion_signals = ["feel", "mood", "tired", "happy", "sad", "frustrated", "excited"]
    if any(s in text for s in emotion_signals):
        return "personal"
    
    return "personal"  # Default


def _guess_category(text: str, domain: str) -> str:
    """Guess the category given a domain."""
    category_map = {
        "personal": "identity",
        "technical": "tech_stack",
        "project": "project_status",
        "relationship": "relationship_dynamic",
        "goal": "goals",
        "preference": "preferences",
    }
    return category_map.get(domain, "identity")


def _extract_topic(text: str, category: str) -> Optional[str]:
    """Extract a specific topic name from the fact text."""
    # Look for proper nouns / project names
    entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    if entities:
        return entities[0]
    
    # Look for project/repo names (camelCase, kebab-case)
    repos = re.findall(r'\b([\w-]+/[A-Za-z][\w-]+)\b', text)
    if repos:
        return repos[0]
    
    # Look for quoted strings
    quoted = re.findall(r'["\x27]([^"\x27]+)["\x27]', text)
    if quoted:
        return quoted[0]
    
    return None


def get_domain_categories(domain: str) -> List[str]:
    """Get all categories belonging to a domain."""
    return [cat for cat, (d, _) in CATEGORIES.items() if d == domain]


def get_all_domains() -> List[str]:
    return list(DOMAINS.keys())


def get_all_categories() -> List[str]:
    return list(CATEGORIES.keys())


def get_category_domain(category: str) -> str:
    return CATEGORIES.get(category, ("personal",))[0]
