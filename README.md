# Project KORTEX

Experiential memory for Hermes Agent

KORTEX transforms Hermes from a stateless LLM into a self-evolving entity with genuine continuity of experience. It provides episodic memory, emotional awareness, relationship modeling, temporal context, self-reflection, and identity evolution, all within a ~1800 token context budget per turn.

## Features

- **Episodic Memory** — Every conversation turn is stored with extracted topics, entities, emotional valence, and salience scoring. Memories are ranked by recency, relevance, and emotional weight for retrieval.
- **Stable Facts & Open Loops** — Durable facts about users and projects are extracted via heuristic patterns and maintained with confidence scoring. Open commitments and questions are tracked and auto-resolved.
- **Emotional Awareness** — 8-dimensional affect scoring (frustration, warmth, humor, hostility, gratitude, anxiety, excitement, trust) with sarcasm detection and dampening.
- **Relationship Dynamics** — EMA-based relationship state (warmth, trust, tension, familiarity, humor, formality, volatility) that evolves naturally over interactions with per-dimension learning rates and momentum.
- **Self-Reflection** — Automatic detection of mistakes, successful patterns, user style preferences, and identity directives. Reflections are deduplicated (Jaccard similarity) and reinforced over time.
- **Linked Memory Graph** — Entity-to-entity, episode-to-fact, and episode-to-reflection linking with traversal. Related episodes are discovered via Jaccard similarity on shared entities and topics.
- **Identity Evolution** — Identity deltas extracted from conversations can be reviewed and promoted to SOUL.md, enabling the agent's personality to evolve with user approval.

## Architecture

The KORTEX pipeline follows a structured cycle: INGEST -> CONSOLIDATE -> RECALL -> INJECT.

Key files and components:
- `kortex/provider.py` — KortexProvider (MemoryProvider implementation)
- `kortex/db.py` — SQLite + FTS5 storage layer
- `kortex/ingest.py` — Turn ingestion and heuristic extraction
- `kortex/recall.py` — Ranked retrieval with budget packing
- `kortex/affect.py` — Multi-dimensional emotion scoring
- `kortex/relationship.py` — Relationship dynamics engine
- `kortex/reflect.py` — Reflection extraction (mistakes, patterns, preferences)
- `kortex/linker.py` — Memory graph edge creation and traversal
- `kortex/promote.py` — SOUL.md identity evolution

## Context Budget

| Slot | Tokens |
|------|--------|
| Relationship state | 200 |
| Stable facts | 350 |
| Episodic memories (top 2-4) | 700 |
| Open loops | 200 |
| Reflections | 200 |
| Reserve | 150 |
| **Total** | **~1800** |

## Installation

### Method 1: Drop-in plugin (recommended)

```bash
cd ~/.hermes/plugins
mkdir kortex
# Symlink or copy all .py files from the kortex/ package directory
# Plus plugin.yaml from the repo root
```

### Method 2: pip install

```bash
pip install git+https://github.com/VeelaCleave/hermes-kortex.git
```

Set `memory.provider: kortex` in `~/.hermes/config.yaml` or drop the package into the plugins directory.

## Configuration

Example `config.yaml` setup:

```yaml
plugins:
  kortex:
    db_path: null          # auto: ~/.hermes/kortex.db
    auto_extract: true     # extract facts/loops automatically
    max_episodes_per_recall: 4
    max_facts_per_recall: 6
    total_budget: 1800
```

## Agent Tools

The agent gains access to these specialized tools:
- `kortex_search` — Search memories, recall episodes, list facts/loops, and get status.
- `kortex_identity` — Manage identity evolution, review/approve/reject personality deltas, and view SOUL.md.

## Testing

```bash
python3 -m pytest tests/ -q
# 394 tests, ~51s
```

## License

MIT

## Credits

Built for NousResearch/hermes-agent.
