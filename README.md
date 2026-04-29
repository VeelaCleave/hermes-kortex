# Project KORTEX

Experiential memory + lossless context for Hermes Agent.

KORTEX gives Hermes two different but complementary memory layers:

- **MemoryProvider layer** for durable cross-session memory
  - user preferences
  - facts
  - open loops / commitments
  - reflections
  - relationship state
  - SOUL.md identity evolution
- **ContextEngine layer** for same-session lossless context
  - archives exact dropped spans at compression time
  - emits deterministic checkpoints instead of lossy summary-only collapse
  - supports bounded exact re-expansion through engine tools

This repo is meant to be enough for someone else to reproduce the stack, not just read about it.

---

## What this stack does

KORTEX turns Hermes into a system with:

- **episodic memory** from every turn
- **stable facts** with contradiction handling
- **open loop tracking** for promises, unresolved asks, and follow-ups
- **emotional awareness** and affect calibration
- **relationship modeling** over time
- **self-reflection** from successes, mistakes, and user style corrections
- **SOUL.md-driven identity evolution**
- **linked memory graph traversal**
- **lossless context compression** for same-session recall after Hermes compacts history

---

## Current architecture

KORTEX now has two active Hermes integration points.

### 1. Memory provider (`memory.provider: kortex`)

Main responsibilities:

- ingest turns after completion
- extract facts / loops / reflections
- maintain relationship state
- prefetch passive recall before the next turn
- expose KORTEX memory tools

Primary files:

- `kortex/provider.py`
- `kortex/ingest.py`
- `kortex/recall.py`
- `kortex/reflect.py`
- `kortex/linker.py`
- `kortex/promote.py`

### 2. Context engine (`context.engine: kortex`)

Main responsibilities:

- intercept Hermes compression
- archive exact dropped transcript spans into KORTEX DB
- emit deterministic checkpoint messages
- expose lossless context tools for bounded retrieval/expansion

Primary files:

- `kortex/context_engine.py`
- `kortex/db.py`

### Why both exist

The split is intentional:

- **MemoryProvider** = durable semantic memory across sessions
- **ContextEngine** = exact same-session transcript lineage after compression

That means KORTEX does **not** rely only on the LLM deciding to remember something.

---

## Features

### Durable memory features

- **Episodic Memory** — Every turn is stored with salience, valence, arousal, topics, and entities.
- **Stable Facts** — Durable user/project facts with confidence and contradiction/supersession handling.
- **Open Loops** — Commitments, follow-ups, and unresolved threads are tracked and can be auto-resolved.
- **Emotional Awareness** — Multi-dimensional affect scoring and per-user baseline calibration.
- **Relationship Dynamics** — Warmth, trust, tension, familiarity, humor, formality, and volatility evolve over time.
- **Reflections** — Mistakes, preferences, patterns, and behavior-shaping lessons are extracted and reinforced.
- **Identity Evolution** — Identity deltas can be reviewed and promoted into `SOUL.md`.
- **Graph Recall** — Episode ↔ fact ↔ reflection linking for graph-enhanced retrieval.

### Lossless context features

- **Exact dropped-span archival** at compression time
- **Deterministic checkpoint messages** instead of only lossy summary collapse
- **Session aliasing across Hermes compression splits**
- **Archived transcript search** via `kortex_recall`
- **Exact historical re-expansion** via `kortex_expand`

---

## Repo layout

Important files:

- `kortex/provider.py` — Hermes MemoryProvider integration
- `kortex/context_engine.py` — Hermes ContextEngine integration
- `kortex/db.py` — SQLite schema + migrations + lossless storage
- `kortex/config.py` — plugin config loader
- `kortex/ingest.py` — ingestion and extraction
- `kortex/recall.py` — ranked passive recall
- `kortex/reflect.py` — reflections and identity deltas
- `kortex/linker.py` — graph links
- `kortex/promote.py` — SOUL.md promotion path
- `kortex/export.py` — export/import backup tooling
- `tests/` — full test suite

---

## Requirements

### Hermes requirements

KORTEX v0.2.0 requires **Hermes 4.0+** with the new plugin surface.

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
    total_budget: 1800
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

### Verify

```bash
python3 - <<'PY'
from kortex import register
from kortex.provider import KortexProvider
from kortex.context_engine import KortexContextEngine

print('KortexProvider:', KortexProvider)
print('KortexContextEngine:', KortexContextEngine)
PY
```

---

## SOUL.md requirements

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

## Recommended Hermes settings for this stack

These are not strictly required, but they reproduce the intended setup more closely.

### Tool-use enforcement

If using Qwen or similar models, Hermes’ `auto` enforcement may be too weak.

Recommended:

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "qwen"]
```

### Web/browser capability

If you want Hermes to actually use web/browser tools instead of falling back to shell patterns, make sure your normal Hermes web/browser tool credentials are configured too.

### Memory enabled

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  provider: kortex
```

### Context engine enabled

```yaml
context:
  engine: kortex
```

---

## KORTEX config reference

Configured under:

```yaml
plugins:
  kortex:
    ...
```

Important keys:

### Recall / injection

- `total_budget` — hard recall budget target (default `1800`)
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

---

## Slash commands

KORTEX registers `/memory` for direct user access:

```
/memory status       — show memory statistics
/memory facts        — list known durable facts
/memory loops        — list open commitments/threads
/memory search <query> — search experiential memory
/memory consolidate  — merge old episodes into summaries
```

## Agent-facing tools

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

## Lossless context behavior

When Hermes decides to compress:

1. KORTEX archives the exact dropped middle span
2. KORTEX stores the span in SQLite
3. KORTEX extracts lightweight refs (tasks, decisions, files, errors, focus)
4. KORTEX emits a deterministic checkpoint message
5. Later, the model can use `kortex_recall` / `kortex_expand` to recover exact prior turns

### Important limitation

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
- lossless context conversations / aliases / archived messages / spans / refs / checkpoints

---

## Export / backup

KORTEX supports JSON export/import via `kortex_export`.

Use this for:

- migration
- backup
- transferring memories between environments

---

## Testing

Run full verification:

```bash
python3 -m pytest -q
```

Current passing state in this repo:

- `453 passed`

Targeted lossless-engine validation also passes.

---

## Reproducing the exact live stack

To reproduce on another machine:

1. `pip install hermes-kortex`
2. Configure `~/.hermes/config.yaml` with the settings from **Configure Hermes** above
3. Create `~/.hermes/SOUL.md` (starter template below)

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
    total_budget: 1800
    passive_recall: true
    context_engine_enabled: true
    passive_context_hint: true
```

If using Qwen-like models:

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "qwen"]
```

---

## Notes for adopters

- Hermes core changes are intentionally avoided.
- KORTEX is designed as a plugin-side system.
- Uses Hermes 4.0+ pip entry-points for auto-discovery.
- Falls back gracefully to manual `register_tool` + `register_hook` for older Hermes versions.
- The `kind: exclusive` in plugin.yaml ensures only one external memory provider runs at a time.

---

## License

MIT

## Credits

Built for Hermes Agent.
