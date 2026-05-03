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
        # Single unified query tool — replaces kortex_search, kortex_identity, kortex_export
        ctx.register_tool(
            name="kortex_query",
            toolset="kortex",
            schema={
                "name": "kortex_query",
                "description": (
                    "Query KORTEX experiential memory. Searches episodes, facts, open loops, "
                    "and identity deltas. Returns ranked results. "
                    "Use for: recalling past conversations, finding facts, checking open threads.\n\n"
                    "Actions:\n"
                    "- search: Search memory by query string\n"
                    "- recent: Get recent episodes\n"
                    "- facts: List known facts\n"
                    "- loops: List open threads\n"
                    "- status: Memory statistics\n"
                    "- consolidate: Run consolidation\n"
                    "- identity: Manage identity deltas\n"
                    "- export/import: Backup/restore memory"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "search", "recent", "facts", "loops",
                                "status", "consolidate", "identity",
                                "export", "import",
                            ],
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query or identity action",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 5)",
                        },
                        "params": {
                            "type": "object",
                            "description": "Additional params (e.g. for identity actions)",
                        },
                    },
                    "required": ["action"],
                },
            },
            handler=lambda args, _prov=provider, **kw: _prov.handle_tool_call("kortex_query", args, **kw),
        )
        # ── Session lifecycle hooks ──────────────────────────────────
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
        # ── Passive memory injection via pre_llm_call hook ─────────
        ctx.register_hook("pre_llm_call", lambda **kw: _kortex_passive_recall(provider, kw))

    if hasattr(ctx, "register_command"):
        ctx.register_command(
            "memory",
            lambda args: provider.handle_tool_call("kortex_query", {"action": args}),
            description="Query KORTEX memory",
            args_hint="[search <query> | status | facts | loops]",
        )


def _kortex_passive_recall(provider: KortexProvider, kwargs: dict) -> dict:
    """Passive memory injection — fires before every LLM call.

    Uses delta-only injection: tracks last injection time per session and only
    returns memories newer than that cutoff, avoiding repeated context across calls.
    """
    try:
        user_message = kwargs.get("user_message", "")
        session_id = kwargs.get("session_id", "")
        if not provider._recall:
            provider.initialize(session_id)
        if not provider._recall:
            return {}
        # Delta cutoff: only include memories created after the last LLM call
        since = provider._last_injection.get(session_id, 0.0)
        context = provider._recall.build_context(
            user_message, session_id=session_id, user_id=provider._user_id,
            lightweight=True, delta_since=since
        )
        # Record injection time for next call's delta calculation
        from .time_utils import now_epoch
        provider._last_injection[session_id] = now_epoch()
        return {"context": context}
    except Exception:
        return {}