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

You need a Hermes Agent build that supports **both**:

- external **memory providers**
- external **context engines**

This stack assumes a Hermes checkout that has:

- `plugins/memory/<name>/`
- `plugins/context_engine/<name>/`
- `context.engine` config support

### Python requirements

- Python 3.10+
- SQLite with FTS5 support
- `pytest` for validation

---

## Installation modes

There are two realistic ways to use KORTEX.

## Mode A — source repo + manual live deployment (matches this stack)

This is the most accurate reproduction of the working setup.

### 1. Clone the repo

```bash
git clone https://github.com/VeelaCleave/hermes-kortex.git
cd hermes-kortex
```

### 2. Verify tests

```bash
python3 -m pytest -q
```

### 3. Install into live Hermes

You need to sync KORTEX into **three** places:

#### A. User plugin copy

```text
~/.hermes/plugins/kortex/
```

#### B. Live Hermes memory provider path

```text
~/.hermes/hermes-agent/plugins/memory/kortex/
```

#### C. Live Hermes context engine path

```text
~/.hermes/hermes-agent/plugins/context_engine/kortex/
```

The memory-provider and user-plugin directories should contain the KORTEX Python package files.

The context-engine directory needs:

- `__init__.py`
- `plugin.yaml`

The context-engine entrypoint should load/register `KortexContextEngine`.

### 4. Configure Hermes

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

### 5. Verify Hermes can load both integrations

From the live Hermes checkout:

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, '/path/to/live/hermes-agent')

from plugins.memory import discover_memory_providers, load_memory_provider
from plugins.context_engine import discover_context_engines, load_context_engine

print('memory_providers=', [name for name, _, _ in discover_memory_providers()])
print('context_engines=', [name for name, _, _ in discover_context_engines()])

mp = load_memory_provider('kortex')
ce = load_context_engine('kortex')

print('memory_load=', type(mp).__name__ if mp else None)
print('context_load=', type(ce).__name__ if ce else None)
print('context_tools=', [schema['name'] for schema in ce.get_tool_schemas()] if ce else None)
PY
```

Expected shape:

```text
memory_load= KortexProvider
context_load= KortexContextEngine
context_tools= ['kortex_recall', 'kortex_expand']
```

---

## Mode B — pip install / package usage

```bash
pip install git+https://github.com/VeelaCleave/hermes-kortex.git
```

This is useful for development or packaging, but many Hermes users will still want the explicit live plugin path installation shown above.

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

If you want another person to reproduce the same stack you have now, send them:

1. this repo
2. a Hermes build with memory-provider + context-engine support
3. your intended `config.yaml` template
4. a starter `SOUL.md`
5. the install instructions from **Mode A** above

Minimum required live settings:

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

And if using Qwen-like models:

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "qwen"]
```

---

## Notes for adopters

- Hermes core changes are intentionally avoided.
- KORTEX is designed as a plugin-side system.
- Live deployment currently uses explicit plugin sync into Hermes plugin directories.
- If Hermes changes its plugin loading layout in future updates, deployment paths may need adjustment, but KORTEX’s source architecture should remain valid.

---

## License

MIT

## Credits

Built for Hermes Agent.
