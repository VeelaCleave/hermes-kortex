"""Project KORTEX — Experiential memory plugin for Hermes Agent.

Provides genuine continuity of experience through episodic memory,
emotional tagging, temporal awareness, and relationship modeling.

Install: pip install hermes-kortex
   - or - drop into ~/.hermes/plugins/kortex/
"""

from __future__ import annotations

from .config import load_kortex_config
from .provider import KortexProvider


def register(ctx) -> None:
    config = load_kortex_config()
    provider = KortexProvider(config=config)
    ctx.register_memory_provider(provider)
