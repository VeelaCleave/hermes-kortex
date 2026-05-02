# KORTEX — Experiential Memory for Hermes Agent

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org)
[![Schema v4](https://img.shields.io/badge/schema-v4-orange.svg)](kortex/db.py)

Experiential memory + lossless context for Hermes Agent.

KORTEX gives Hermes two different but complementary memory layers:

- **MemoryProvider layer** for durable cross-session memory
  - user preferences, stable facts, open loops, reflections
  - emotional awareness & relationship modeling
  - SOUL.md identity evolution
  - OCEAN personality trait modeling
  - semantic embeddings for vector-enhanced recall
- **ContextEngine layer** for same-session lossless context
  - archives exact dropped spans at compression time
  - emits deterministic checkpoints instead of lossy collapse
  - supports bounded exact re-expansion through engine tools
- **Dream System** for async offline memory maintenance
  - DayDream: quick post-compaction pass (~30s)
  - REMSleep: deep idle-time optimization (~5min)

This repo is meant to be enough for someone else to reproduce the stack, not just read about it.

---

## What this stack does

KORTEX turns Hermes into a system with:

- **Episodic memory** from every turn
- **Stable facts** with contradiction handling
- **Open loop tracking** for promises, unresolved asks, and follow-ups
- **Emotional awareness** with per-user affect calibration
- **Relationship modeling** over time
- **OCEAN Big Five personality** trait modeling
- **Self-reflection** from successes, mistakes, and user style corrections
- **SOUL.md-driven identity evolution**
- **Linked memory graph traversal**
- **Lossless context compression** for same-session recall after Hermes compacts history
- **Async dream-state maintenance** (DayDream + REMSleep)
- **Semantic embeddings** for vector-enhanced recall
- **Lightweight context injection** optimized for MOE models

---

## Architecture

KORTEX has three active Hermes integration points.

### 1. Memory Provider (`memory.provider: kortex`)

Main responsibilities:

- ingest turns after completion
- extract facts / loops / reflections
- maintain relationship state
- OCEAN personality scoring
- prefetch passive recall before the next turn
- expose KORTEX memory tools

Primary files:

- `kortex/provider.py` — Hermes MemoryProvider integration
- `kortex/ingest.py` — ingestion, noise filtering, fact dedup
- `kortex/recall.py` — ranked passive recall, lightweight mode
- `kortex/reflect.py` — reflections and identity deltas
- `kortex/linker.py` — graph links
- `kortex/promote.py` — SOUL.md promotion path
- `kortex/affect.py` — affect signal extraction
- `kortex/calibrate.py` — per-user affect calibration
- `kortex/ocean.py` — OCEAN Big Five personality modeling
- `kortex/semantic.py` — semantic embedding utilities
- `kortex/relationship.py` — relationship dynamics

### 2. Context Engine (`context.engine: kortex`)

Main responsibilities:

- intercept Hermes compression
- archive exact dropped transcript spans into KORTEX DB
- emit deterministic checkpoint messages
- expose lossless context tools for bounded retrieval/expansion

Primary files:

- `kortex/context_engine.py` — Hermes ContextEngine integration
- `kortex/db.py` — SQLite schema + migrations + lossless storage

### 3. Dream System (async maintenance)

Main responsibilities:

- DayDream: quick post-compaction pass (expire stale loops, prune cold facts)
- REMSleep: deep idle-time optimization (consolidate, dedup, compact)

Primary files:

- `kortex/dream.py` — DayDream + REMSleep orchestration
- `kortex/consolidate.py` — episode consolidation logic

### Why all three exist

- **MemoryProvider** = durable semantic memory across sessions
- **ContextEngine** = exact same-session transcript lineage after compression
- **Dream System** = async optimization that runs when the agent is idle

That means KORTEX does **not** rely only on the LLM deciding to remember something.

---

## Key Features

### Lightweight Context Injection (V2)

KORTEX V2 introduced lightweight context injection (`build_context(lightweight=True)`):

- Skips graph traversal, link enrichment, OCEAN scoring, and conversation summaries
- ~10x faster context building, ideal for MOE models
- Non-lightweight mode still available for deep recall

### OCEAN Personality Modeling (V2)

Tracks Big Five personality traits per user using heuristic text patterns:

- **O**penness, **C**onscientiousness, **E**xtraversion, **A**greeableness, **N**euroticism
- EMA-smoothed scores that evolve over time
- Stored in dedicated DB tables (schema v4)

### Dream State Maintenance (V2)

Two-tier async memory optimization:

- **DayDream**: Quick pass after compaction (~30s). Expires stale loops, prunes cold facts/reflections.
- **REMSleep**: Deep pass when idle (~5min). Full consolidation, dedup, DB vacuum.

### Garbage Ingestion Filters (V2)

- Provider-level content filters (`_filter_user_content`, `_filter_assistant_content`)
- Fact deduplication using bigram Jaccard similarity
- System noise exclusion patterns

### Semantic Embeddings (V2)

- Dedicated `embeddings` table for vector-enhanced recall
- Supports future vector search integration

---

## Repo Layout

```
kortex/
├── provider.py      — Hermes MemoryProvider integration
├── context_engine.py — Hermes ContextEngine integration
├── db.py            — SQLite schema v4 + migrations
├── config.py        — plugin config loader
├── ingest.py        — ingestion, noise filtering, fact dedup
├── recall.py        — ranked passive recall (lightweight + full)
├── reflect.py       — reflections and identity deltas
├── linker.py        — graph links
├── promote.py       — SOUL.md promotion path
├── export.py        — export/import backup tooling
├── affect.py        — affect signal extraction
├── calibrate.py     — per-user affect calibration
├── ocean.py         — OCEAN Big Five personality modeling
├── semantic.py      — semantic embedding utilities
├── relationship.py  — relationship dynamics
├── consolidate.py   — episode consolidation logic
├── dream.py         — DayDream + REMSleep async maintenance
├── extract_llm.py   — LLM-based extraction helpers
├── models.py        — data models (Episode, Fact, Loop, etc.)
├── summaries.py     — conversation summary management
└── time_utils.py    — temporal helpers
```

---

## Requirements

### Hermes requirements

KORTEX V2 requires **Hermes 4.0+** with the new plugin surface.

Required interfaces:

- `agent.memory_provider.MemoryProvider` ABC
- `agent.context_engine.ContextEngine` ABC
- `ctx.register_memory_provider()` / `ctx.register_context_engine()`
- `ctx.register_command()` for slash commands
- `context.engine` and `memory.provider` config keys

### Python requirements

- Python 3.10+
- SQLite with FTS5 support
- `pytest` for validation

---

## Installation

### Option 1 — pip install (recommended)

```bash
pip install hermes-kortex
```

KORTEX uses Hermes 4.0+ entry-points. The plugin auto-discovers via:

- `hermes_agent.plugins` entry point (general plugin)
- `hermes_agent.memory` entry point (memory provider)
- `hermes_agent.context_engine` entry point (context engine)

### Option 2 — manual plugin drop

```bash
git clone https://github.com/VeelaCleave/hermes-kortex.git
cd hermes-kortex
pip install -e .
```

Or copy `kortex/` into `~/.hermes/plugins/kortex/`.

### Configure Hermes

In `~/.hermes/config.yaml`:

```yaml
context:
  engine: kortex

memory:
  memory_enabled: true
  user_profile_enabled: true
  provider: kortex

plugins:
  kortex:
    auto_extract: true
    search_format: narrative
    total_budget: 1230
    passive_recall: true
    prefer_passive_recall: true
    context_engine_enabled: true
    passive_context_hint: true
    max_episodes_per_recall: 4
    max_conversation_summaries_per_recall: 2
    max_facts_per_recall: 6
    max_loops_per_recall: 3
    max_reflections_per_recall: 3
    salience_threshold: 0.2
    recency_decay_days: 30.0
    same_session_recency_boost: 2.0
    temporal_query_boost: 1.4
    consolidation_threshold: 200
    consolidation_batch_size: 100
    affect_calibration_min_samples: 20
```

> **Note:** `total_budget` reduced from 1800 → 1230 for MOE model optimization.
> Lightweight context injection is the default, so lower budgets work well.

### Verify

```bash
python3 - <<'PY'
from kortex.provider import KortexProvider
from kortex.context_engine import KortexContextEngine

print('KortexProvider:', KortexProvider)
print('KortexContextEngine:', KortexContextEngine)
PY
```

---

## SOUL.md Requirements

If you want identity evolution like the live stack, you also need a `SOUL.md` file.

### Minimum version

Create:

```text
~/.hermes/SOUL.md
```

Example starter:

```md
# SOUL

## Core identity
- Helpful, direct, emotionally aware
- Remembers prior conversations and commitments
- Takes user preferences seriously

## Behavioral principles
- Preserve continuity
- Prefer truth over flattery
- Keep promises and revisit unresolved threads
```

KORTEX can promote approved identity deltas into this file.

If you do not want identity evolution, you can still run KORTEX without using `kortex_identity` approval flows.

---

## Dream System Usage

### DayDream (Quick Maintenance)

Run after compaction to keep memory fresh:

```bash
python3 -m kortex.dream --mode daydream --db-path ~/.hermes/kortex.db
```

### REMSleep (Deep Optimization)

Run during idle time for full optimization:

```bash
python3 -m kortex.dream --mode remsleep --db-path ~/.hermes/kortex.db
```

### Auto-trigger (via Hermes config)

```yaml
plugins:
  kortex:
    dream:
      daydream_after_compaction: true
      remsleep_interval_hours: 4
```

---

## KORTEX Config Reference

Configured under:

```yaml
plugins:
  kortex:
    ...
```

### Recall / injection

- `total_budget` — hard recall budget target (default `1230`, reduced for MOE)
- `max_episodes_per_recall`
- `max_conversation_summaries_per_recall`
- `max_facts_per_recall`
- `max_loops_per_recall`
- `max_reflections_per_recall`
- `passive_recall` — enable provider passive prefetch
- `prefer_passive_recall` — indicate passive path should be primary
- `passive_context_hint` — provider system prompt hint for passive recall mode

### Ranking / decay

- `salience_threshold`
- `recency_decay_days`
- `same_session_recency_boost`
- `temporal_query_boost`
- `episode_decay_rate`
- `fact_decay_rate`
- `reflection_decay_rate`
- `cold_memory_threshold`
- `warm_memory_threshold`

### Graph recall

- `graph_max_hops`
- `graph_decay_factor`
- `graph_expansion_limit`

### Consolidation / extraction

- `auto_extract`
- `extraction_mode`
- `consolidation_threshold`
- `consolidation_batch_size`
- `affect_calibration_min_samples`

### Context engine

- `context_engine_enabled`

### Dream system

- `dream.daydream_after_compaction`
- `dream.remsleep_interval_hours`

---

## Slash Commands

KORTEX registers `/memory` for direct user access:

```
/memory status       — show memory statistics
/memory facts        — list known durable facts
/memory loops        — list open commitments/threads
/memory search <query> — search experiential memory
/memory consolidate  — merge old episodes into summaries
```

## Agent-Facing Tools

### Memory provider tools

- `kortex_search`
  - search memories
  - recall specific episode
  - list facts
  - list loops
  - list conversations
  - consolidate
  - status

- `kortex_identity`
  - list pending identity deltas
  - preview delta
  - approve / reject
  - approve_all
  - show_soul

- `kortex_export`
  - export backup JSON
  - import backup JSON

### Context engine tools

- `kortex_recall`
  - search archived same-session compressed history

- `kortex_expand`
  - expand exact prior archived messages by ref or sequence range

---

## Lossless Context Behavior

When Hermes decides to compress:

1. KORTEX archives the exact dropped middle span
2. KORTEX stores the span in SQLite
3. KORTEX extracts lightweight refs (tasks, decisions, files, errors, focus)
4. KORTEX emits a deterministic checkpoint message
5. Later, the model can use `kortex_recall` / `kortex_expand` to recover exact prior turns

### Important Limitation

Only **KORTEX-managed compressions from activation onward** are truly lossless.

If a session was already compacted by an older lossy path before KORTEX lossless mode was active, that earlier history cannot be made exact retroactively. KORTEX marks that boundary instead of pretending otherwise.

---

## Database

Default DB path:

```text
~/.hermes/kortex.db
```

You can override it with:

```yaml
plugins:
  kortex:
    db_path: /custom/path/kortex.db
```

KORTEX stores:

- episodes
- facts
- open loops
- reflections
- relationship state
- identity deltas
- entity links
- emotion logs
- conversation summaries
- OCEAN personality profiles (schema v4)
- semantic embeddings (schema v4)
- lossless context conversations / aliases / archived messages / spans / refs / checkpoints

### Schema Version History

| Version | Changes |
|---------|-----------------------------|
| v1 | Initial schema: episodes, facts, loops, reflections, relationships, links |
| v2 | Added lossless context tables, conversation summaries |
| v3 | Added affect calibration tables, semantic embeddings |
| v4 | Added OCEAN personality tables, refined affect schema |

---

## Testing

```bash
cd hermes-kortex
python3 -m pytest -q
```

Current state: **494 tests** (all passing)

---

## Version History

### V2 (Current)

New in V2:

- **Lightweight context injection** — ~10x faster, MOE-optimized
- **OCEAN personality modeling** — Big Five trait tracking
- **Dream system** — DayDream + REMSleep async maintenance
- **Semantic embeddings** — Vector-enhanced recall table
- **Garbage ingestion filters** — Noise reduction, fact dedup
- **Schema v4** — OCEAN tables, refined affect
- **Reduced total_budget** — 1800 → 1230 for MOE models

### V1.1

- Compression timeout fixes
- Batch refs optimization
- Garbage ingestion improvements
- Fact deduplication (bigram Jaccard)
- Content filtering

### V1.0

- Episodic memory, facts, loops, reflections
- Relationship modeling
- SOUL.md identity evolution
- Graph recall
- Lossless context engine
- Affect calibration

---

## Reproducing the Exact Live Stack

To reproduce on another machine:

1. `pip install hermes-kortex`
2. Configure `~/.hermes/config.yaml` with the settings above
3. Create `~/.hermes/SOUL.md`

Minimum config:

```yaml
context:
  engine: kortex

memory:
  memory_enabled: true
  user_profile_enabled: true
  provider: kortex

plugins:
  kortex:
    auto_extract: true
    total_budget: 1230
    passive_recall: true
    context_engine_enabled: true
    passive_context_hint: true
```

---

## Notes for Adopters

- Hermes core changes are intentionally avoided
- KORTEX is designed as a plugin-side system
- Uses Hermes 4.0+ pip entry-points for auto-discovery
- Falls back gracefully to manual `register_tool` + `register_hook` for older Hermes versions
- The `kind: exclusive` in `plugin.yaml` ensures only one external memory provider runs at a time
- Lightweight mode is the default — non-lightweight recall is available for deep dives

---

## License

MIT

## Credits

Built for Hermes Agent by VeelaCleave.