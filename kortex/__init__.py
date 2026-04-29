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

    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(
            KortexContextEngine(db_path=config.db_path)
        )
    if hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(provider)
    elif hasattr(ctx, "register_tool"):
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
            lambda **kw: provider.initialize(
                kw.pop("session_id", ""), **kw
            ),
        )
        ctx.register_hook(
            "on_session_end",
            lambda **kw: (
                provider.on_session_end(kw.get("messages", [])),
                provider.shutdown(),
            ),
        )

    if hasattr(ctx, "register_command"):
        ctx.register_command(
            "memory",
            _handle_memory_command,
            description="Search and manage KORTEX experiential memory",
            args_hint="[search <query> | status | facts | loops | consolidate]",
        )


def _handle_memory_command(args: str) -> str:
    import json

    config = load_kortex_config()
    provider = KortexProvider(config=config)
    parts = args.strip().split(maxsplit=1)
    action = parts[0] if parts else "status"
    query = parts[1] if len(parts) > 1 else ""

    action_map = {
        "status": lambda: provider._handle_status(),
        "facts": lambda: provider._handle_list_facts(10),
        "loops": lambda: provider._handle_list_loops(10),
        "search": lambda: provider._handle_search(query, 5)
        if query
        else json.dumps({"error": "query required for search"}),
        "consolidate": lambda: provider._handle_consolidate(None),
    }

    handler = action_map.get(action, action_map["status"])
    return handler()
