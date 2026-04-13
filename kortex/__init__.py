"""Project KORTEX — Experiential memory plugin for Hermes Agent.

Provides genuine continuity of experience through episodic memory,
emotional tagging, temporal awareness, and relationship modeling.

Install: pip install hermes-kortex
   - or - drop into ~/.hermes/plugins/kortex/
"""

from __future__ import annotations

from .config import load_kortex_config
from .context_engine import KortexContextEngine
from .provider import KortexProvider


def register(ctx) -> None:
    config = load_kortex_config()
    provider = KortexProvider(config=config)
    if getattr(config, "context_engine_enabled", True) and hasattr(
        ctx, "register_context_engine"
    ):
        ctx.register_context_engine(KortexContextEngine())
    if hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(provider)
    else:
        # Fallback: register tools directly if running as a general plugin
        # (PluginContext doesn't expose register_memory_provider)
        for schema in provider.get_tool_schemas():
            ctx.register_tool(
                name=schema["name"],
                toolset="kortex",
                schema=schema,
                handler=lambda args, _prov=provider, _name=schema["name"], **kw: (
                    _prov.handle_tool_call(_name, args, **kw)
                ),
            )
        ctx.register_hook(
            "on_session_start",
            lambda **kw: provider.initialize(kw.pop("session_id", ""), **kw),
        )
        ctx.register_hook(
            "on_session_end",
            lambda **kw: (
                provider.on_session_end(kw.get("messages", [])),
                provider.shutdown(),
            ),
        )
