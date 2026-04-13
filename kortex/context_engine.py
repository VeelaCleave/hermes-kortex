"""Optional KORTEX context-engine adapter for Hermes.

This is intentionally a thin wrapper around Hermes' built-in context
compressor. KORTEX remains primarily a MemoryProvider; this adapter exists so
KORTEX can participate in Hermes' new context-engine slot without taking over
compression semantics.

When activated via ``context.engine: kortex``, this engine delegates actual
compression behavior to Hermes' built-in ``ContextCompressor`` when available,
while preserving a stable plugin identity and future extension point for
compression-aware KORTEX features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


try:  # pragma: no cover - exercised only inside Hermes runtime
    from agent.context_engine import ContextEngine as HermesContextEngine
except ImportError:  # pragma: no cover - local repo tests

    class HermesContextEngine:
        """Minimal fallback used when Hermes isn't importable in unit tests."""

        last_prompt_tokens: int = 0
        last_completion_tokens: int = 0
        last_total_tokens: int = 0
        threshold_tokens: int = 0
        context_length: int = 0
        compression_count: int = 0
        threshold_percent: float = 0.75
        protect_first_n: int = 3
        protect_last_n: int = 6

        def on_session_reset(self) -> None:
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            self.last_total_tokens = 0
            self.compression_count = 0

        def get_tool_schemas(self) -> List[Dict[str, Any]]:
            return []

        def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
            import json

            return json.dumps({"error": f"Unknown context engine tool: {name}"})

        def get_status(self) -> Dict[str, Any]:
            usage_percent = (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length
                else 0
            )
            return {
                "last_prompt_tokens": self.last_prompt_tokens,
                "threshold_tokens": self.threshold_tokens,
                "context_length": self.context_length,
                "usage_percent": usage_percent,
                "compression_count": self.compression_count,
            }

        def update_model(
            self,
            model: str,
            context_length: int,
            base_url: str = "",
            api_key: str = "",
            provider: str = "",
        ) -> None:
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self.threshold_percent)


try:  # pragma: no cover - exercised only inside Hermes runtime
    from agent.context_compressor import ContextCompressor
except ImportError:  # pragma: no cover - local repo tests
    ContextCompressor = None  # type: ignore[assignment]


class KortexContextEngine(HermesContextEngine):
    """Delegate compression to Hermes while reserving a KORTEX engine slot."""

    threshold_percent: float = 0.50
    protect_first_n: int = 3
    protect_last_n: int = 20

    def __init__(self) -> None:
        self._delegate: Optional[Any] = None
        self._session_id: str = ""
        self._model: str = "unknown"
        self._kwargs: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "kortex"

    def is_available(self) -> bool:
        return True

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._kwargs = dict(kwargs)
        self._model = str(kwargs.get("model") or self._model or "unknown")
        self._ensure_delegate()
        if self._delegate and hasattr(self._delegate, "on_session_start"):
            self._delegate.on_session_start(session_id, **kwargs)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if self._delegate and hasattr(self._delegate, "on_session_end"):
            self._delegate.on_session_end(session_id, messages)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        if self._delegate and hasattr(self._delegate, "on_session_reset"):
            self._delegate.on_session_reset()

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if self._delegate:
            self._delegate.update_from_response(usage)
            self._sync_from_delegate()
            return
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        self._ensure_delegate()
        if self._delegate:
            result = self._delegate.should_compress(prompt_tokens)
            self._sync_from_delegate()
            return result
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens if self.threshold_tokens else False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_delegate()
        if self._delegate:
            result = self._delegate.compress(messages, current_tokens=current_tokens)
            self._sync_from_delegate()
            return result

        self.compression_count += 1
        return messages

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        self._ensure_delegate()
        if self._delegate and hasattr(self._delegate, "should_compress_preflight"):
            return bool(self._delegate.should_compress_preflight(messages))
        return False

    def get_status(self) -> Dict[str, Any]:
        base = super().get_status()
        base.update(
            {
                "engine": self.name,
                "delegate": "compressor" if self._delegate else "passthrough",
                "session_id": self._session_id,
            }
        )
        return base

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        self._model = model
        if self._delegate:
            self._delegate.update_model(
                model,
                context_length,
                base_url=base_url,
                api_key=api_key,
                provider=provider,
            )
            self._sync_from_delegate()
            return

        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

    def _ensure_delegate(self) -> None:
        if self._delegate is not None or ContextCompressor is None:
            return

        config_context_length = self._kwargs.get("context_length")
        base_url = str(self._kwargs.get("base_url") or "")
        api_key = str(self._kwargs.get("api_key") or "")
        provider = str(self._kwargs.get("provider") or "")
        api_mode = str(self._kwargs.get("api_mode") or "")

        self._delegate = ContextCompressor(
            model=self._model,
            quiet_mode=True,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            api_mode=api_mode,
            config_context_length=config_context_length,
        )
        self._sync_from_delegate()

    def _sync_from_delegate(self) -> None:
        if not self._delegate:
            return
        self.last_prompt_tokens = getattr(self._delegate, "last_prompt_tokens", 0)
        self.last_completion_tokens = getattr(
            self._delegate, "last_completion_tokens", 0
        )
        self.last_total_tokens = getattr(self._delegate, "last_total_tokens", 0)
        self.threshold_tokens = getattr(self._delegate, "threshold_tokens", 0)
        self.context_length = getattr(self._delegate, "context_length", 0)
        self.compression_count = getattr(self._delegate, "compression_count", 0)
