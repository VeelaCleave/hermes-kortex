"""LLM-assisted structured extraction helpers for KORTEX."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExtractionLLMClient:
    """Wrapper around an OpenAI-compatible client that exposes complete().

    Used when auxiliary_client is not provided by Hermes — lets kortex create
    its own LLM client from plugins.kortex.extraction_llm_base_url / model.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Lazy import to avoid hard dependency when not using extraction LLM
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning(
                "OpenAI package not available. Install with: pip install openai"
            )
            self._client = None
            return
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=1,
        )

    def complete(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Call chat completion with a structured extraction prompt.

        Args:
            prompt: dict with user_text, assistant_text, and instruction keys.
        """
        if self._client is None:
            return {}

        user_text = prompt.get("user_text", "")
        assistant_text = prompt.get("assistant_text", "")
        instruction = prompt.get(
            "instruction",
            "Extract structured memory from this conversation. Return JSON with "
            "keys: summary, topics, entities, facts, open_loops, reflections, affect_hints.",
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory extraction assistant. Your task is to extract "
                    "structured information from user conversations.\n\n"
                    "Return ONLY valid JSON — no markdown, no explanation, no text outside "
                    "the JSON object. The JSON may contain these keys:\n"
                    "- summary: a 1-2 sentence summary of the conversation\n"
                    "- topics: list of topic keywords\n"
                    "- entities: list of named entities (people, places, projects)\n"
                    "- facts: list of {predicate, object_text} pairs for durable facts\n"
                    "- open_loops: list of {kind, text} for commitments or questions\n"
                    "- reflections: list of insight strings\n"
                    "- affect_hints: dict of emotional signals\n\n"
                    f"Instruction: {instruction}"
                ),
            },
            {
                "role": "user",
                "content": f"User: {user_text}\n\nAssistant: {assistant_text}",
            },
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            import json

            return json.loads(content)
        except Exception as e:
            logger.warning("[llm] client.complete failed: %s", e)
            return {}


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
