"""LLM-assisted structured extraction helpers for KORTEX."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def extract_structured_memory(
    user_text: str,
    assistant_text: str,
    auxiliary_client: Any = None,
) -> Optional[Dict[str, Any]]:
    if auxiliary_client is None:
        return None

    prompt = {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "instruction": (
            "Extract JSON with keys: summary, topics, entities, facts, open_loops, "
            "reflections, affect_hints. Facts should be objects with predicate and object_text. "
            "Open loops should be objects with kind and text."
        ),
    }

    result = None
    if hasattr(auxiliary_client, "extract_structured"):
        result = auxiliary_client.extract_structured(prompt)
    elif hasattr(auxiliary_client, "generate_json"):
        result = auxiliary_client.generate_json(prompt)
    elif hasattr(auxiliary_client, "complete"):
        result = auxiliary_client.complete(prompt)

    if not isinstance(result, dict):
        return None

    return {
        "summary": str(result.get("summary", "") or "").strip(),
        "topics": _normalize_str_list(result.get("topics", [])),
        "entities": _normalize_str_list(result.get("entities", [])),
        "facts": _normalize_facts(result.get("facts", [])),
        "open_loops": _normalize_loops(result.get("open_loops", [])),
        "reflections": _normalize_str_list(result.get("reflections", [])),
        "affect_hints": result.get("affect_hints", {}) or {},
    }


def _normalize_str_list(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    seen = set()
    result: List[str] = []
    for item in value:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result


def _normalize_facts(value: Any) -> List[Tuple[str, str]]:
    if not isinstance(value, list):
        return []
    result: List[Tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        predicate = str(item.get("predicate", "") or "").strip()
        object_text = str(item.get("object_text", "") or "").strip()
        if predicate and object_text:
            result.append((predicate, object_text))
    return result


def _normalize_loops(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "") or "").strip() or "question"
        text = str(item.get("text", "") or "").strip()
        if text:
            result.append({"kind": kind, "text": text})
    return result
