# KORTEX v1.1 — Full Upgrade Plan

## Overview

Complete upgrade of KORTEX memory plugin from v1.0 (heuristic extraction, ISO8601 timestamps, single-user) to v1.1 (12 new features, forward-compatible schema, epoch timestamps, multi-user ready). All schema changes done before any real memories are created.

## Constraints

- Do NOT modify hermes-agent core files
- Keep context injection under ~2000 tokens per turn
- Commit between each phase, all tests must pass before moving on
- `system_prompt_block()` MUST remain static (prompt caching)
- No data retrofitting — schema must be forward-compatible from day one
- INTEGER PRIMARY KEY AUTOINCREMENT kept (NOT UUIDs — FTS5 content tables require integer rowid)
- SQLite WAL mode mandatory

## Repository

- Location: `/home/veela/hermes-kortex/`
- Current schema version: 2
- Current timestamp format: ISO8601 strings
- Target schema version: 3
- Target timestamp format: REAL epoch floats (time.time())

---

## Phase 0: Schema v3 + Timestamp Migration

**Scope**: Every file in KORTEX. Most disruptive phase — do it first, do it once.

### 0A: Timestamp Migration (ISO8601 → REAL epoch floats)

**Rationale**: Consistent with `hermes_state.py`, enables arithmetic temporal comparisons without parsing, avoids timezone ambiguity.

**Files to change**:
- `models.py` — Change all `datetime` fields to `float` (epoch). Update `Episode.to_recall_text()`, `Episode.timestamp_iso`, `RelationshipState.to_compact_text()`, `AffectSignal.to_db_row()`, `Fact.first_seen/last_seen`, `OpenLoop.created_at/resolved_at`, `Reflection.created_at/last_reinforced`, `IdentityDelta.created_at`.
- `db.py` — Change all `TEXT NOT NULL` timestamp columns to `REAL NOT NULL`. Update all `datetime.now(timezone.utc).isoformat()` to `time.time()`. Update all `.isoformat()` calls. Update `_SCHEMA_SQL`. Update migration from v2→v3.
- `ingest.py` — Update timestamp creation/comparison.
- `recall.py` — Update temporal decay calculation (currently parses ISO8601).
- `provider.py` — Update any `.isoformat()` calls in tool handlers.
- `affect.py` — No timestamps (pure scoring). Likely unchanged.
- `relationship.py` — Update `days_since` calculation if it parses timestamps.
- `reflect.py` — Update timestamp creation.
- `linker.py` — No timestamps. Unchanged.
- `promote.py` — Update any timestamp operations.
- `config.py` — No timestamps. Unchanged.
- All 9 test files — Update all datetime fixtures to epoch floats.

**Helper function**: Add `now_epoch() -> float` returning `time.time()` and `epoch_to_display(ts: float) -> str` for human-readable output in tool responses.

### 0B: Add Forward-Compatible Columns

Add to `episodes`:
- `user_id TEXT NOT NULL DEFAULT '__default__'`
- `last_accessed_at REAL` (NULL until accessed)
- `retrieval_count INTEGER NOT NULL DEFAULT 0`
- `consolidated_into INTEGER REFERENCES episodes(id)` (NULL until consolidated)
- `raw_preserved INTEGER NOT NULL DEFAULT 1`

Add to `facts`:
- `user_id TEXT NOT NULL DEFAULT '__default__'`
- `last_accessed_at REAL`
- `retrieval_count INTEGER NOT NULL DEFAULT 0`
- `valid_from REAL` (NULL, backfill = first_seen on existing data)
- `valid_to REAL` (NULL = current)
- `contradiction_status TEXT NOT NULL DEFAULT 'active'`

Add to `open_loops`:
- `user_id TEXT NOT NULL DEFAULT '__default__'`
- `last_accessed_at REAL`
- `resolution TEXT` (how it was resolved)
- `resolved_by_episode_id INTEGER REFERENCES episodes(id)`

Add to `reflections`:
- `user_id TEXT NOT NULL DEFAULT '__default__'`
- `last_accessed_at REAL`
- `retrieval_count INTEGER NOT NULL DEFAULT 0`
- `promotion_status TEXT NOT NULL DEFAULT 'active'`
- `promoted_at REAL`

Add to `relationship_state`:
- (already has `user_id`) — no change needed

Add to `identity_deltas`:
- `user_id TEXT NOT NULL DEFAULT '__default__'`

Add to `entity_links`:
- `user_id TEXT NOT NULL DEFAULT '__default__'`

Add to `emotion_log`:
- `user_id TEXT NOT NULL DEFAULT '__default__'`

### 0C: New Tables

**`conversation_summaries`** (#5):
```sql
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL DEFAULT '__default__',
    session_id      TEXT NOT NULL,
    summary_text    TEXT NOT NULL,
    summary_level   TEXT NOT NULL DEFAULT 'conversation',
    episode_range_start REAL,
    episode_range_end   REAL,
    episode_count   INTEGER NOT NULL DEFAULT 0,
    key_entities    TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_summaries_user ON conversation_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_summaries_session ON conversation_summaries(session_id);
```

**`affect_baselines`** (#6):
```sql
CREATE TABLE IF NOT EXISTS affect_baselines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL DEFAULT '__default__' UNIQUE,
    baseline_frustration    REAL NOT NULL DEFAULT 0.0,
    baseline_warmth         REAL NOT NULL DEFAULT 0.0,
    baseline_humor          REAL NOT NULL DEFAULT 0.0,
    baseline_hostility      REAL NOT NULL DEFAULT 0.0,
    baseline_gratitude      REAL NOT NULL DEFAULT 0.0,
    baseline_anxiety        REAL NOT NULL DEFAULT 0.0,
    baseline_excitement     REAL NOT NULL DEFAULT 0.0,
    baseline_trust_signal   REAL NOT NULL DEFAULT 0.0,
    sample_count    INTEGER NOT NULL DEFAULT 0,
    ema_alpha       REAL NOT NULL DEFAULT 0.1,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
```

**`kortex_schema_version`**:
```sql
CREATE TABLE IF NOT EXISTS kortex_schema_version (
    version     INTEGER NOT NULL,
    migrated_at REAL NOT NULL
);
```

### 0D: New Indexes

- `idx_episodes_user ON episodes(user_id)`
- `idx_facts_user ON facts(user_id)`
- `idx_facts_valid ON facts(valid_from, valid_to)`
- `idx_facts_contradiction ON facts(contradiction_status)`
- `idx_facts_accessed ON facts(last_accessed_at)`
- `idx_loops_user ON open_loops(user_id)`
- `idx_reflections_user ON reflections(user_id)`
- `idx_reflections_promotion ON reflections(promotion_status)`
- `idx_links_user ON entity_links(user_id)`
- `idx_emotion_user ON emotion_log(user_id)`

### 0E: Update `models.py` Dataclasses

Add new fields to match schema changes. All timestamps become `float` with default `time.time()`.

### 0F: Migration Code

In `db.py._migrate()`, add an **explicit SQLite rebuild/copy migration** for populated v2 databases. Timestamp type conversion from `TEXT` → `REAL` is **not** an in-place `ALTER TABLE` operation.

**Required migration strategy (executable):**
1. Start transaction.
2. Create v3 replacement tables with final schema:
   - `episodes_v3`
   - `facts_v3`
   - `open_loops_v3`
   - `reflections_v3`
   - `relationship_state_v3`
   - `identity_deltas_v3`
   - `emotion_log_v3`
3. Copy data from each v2 table into its v3 replacement using conversion expressions:
   - Existing ISO8601 values must be converted to epoch floats during `INSERT INTO ... SELECT ...`
   - Conversion helper in Python: parse ISO8601 robustly, normalize `Z` to `+00:00`, convert to `datetime`, then `.timestamp()`
   - New columns must be backfilled during copy:
     - `user_id = '__default__'`
     - `retrieval_count = 0`
     - `raw_preserved = 1`
     - `valid_from = first_seen` for existing facts
     - `contradiction_status = 'active'`
     - `promotion_status = 'active'`
4. Drop old indexes tied to replaced tables if needed.
5. Drop old v2 tables.
6. Rename `*_v3` tables to canonical names.
7. Recreate indexes against final tables.
8. Create new tables (`conversation_summaries`, `affect_baselines`, `kortex_schema_version`).
9. Insert initial `kortex_schema_version` record.
10. Update `PRAGMA user_version = 3`.
11. Commit transaction.

**Important implementation note**: do not rely on SQLite casting for ISO8601 conversion. Perform row-wise conversion in Python during migration so populated v2 databases migrate deterministically.

### Verification

- All 394 existing tests pass (updated for epoch floats)
- New test: migration from v2→v3 on a populated database
- New test: verify all tables have `user_id` column
- New test: verify all timestamp columns are REAL
- `PRAGMA journal_mode` returns `wal`

---

## Phase 1: Conversation Summaries (#5)

**New file**: `kortex/summaries.py`

Generate whole-conversation summaries (not per-turn — that's `Episode.summary`). Uses heuristic approach: concatenate episode summaries, extract key themes.

**Integration**:
- Hook into `on_session_end(messages)` in `provider.py` — generate conversation summary when session ends
- Store in `conversation_summaries` table
- Include in recall pipeline (`recall.py`) when query matches session-level patterns

**Tool addition**: Add `list_conversations` action to `kortex_search` — returns conversation summaries.

**Tests**: ~25 tests covering summary generation, storage, recall, session-end hook.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_summaries.py tests/test_provider.py tests/test_recall.py -q`
- Pass criteria:
  - summary generation tests pass
  - session-end hook stores `conversation_summaries` rows
  - recall can surface matching conversation summaries

---

## Phase 2: Consolidation/Compaction (#1)

**New file**: `kortex/consolidate.py`

Merge old episodes into summary records. Triggered by:
1. **Auto**: Episode count > configurable threshold (default 200) per user
2. **Manual**: New `consolidate` action on `kortex_search` tool

**Rules**:
- Never delete episodes — mark `is_consolidated = 1`, set `consolidated_into`
- Raw text preserved (`raw_preserved = 1`)
- Graph edges updated to point to both original AND summary
- Conversation summaries (#1) feed consolidation — episodes grouped by session, summarized

**Integration**:
- Check threshold in `sync_turn()` background thread
- New config options: `consolidation_threshold`, `consolidation_batch_size`

**Tests**: ~30 tests covering trigger logic, consolidation output, edge preservation, raw text preservation.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_consolidate.py tests/test_db.py tests/test_provider.py -q`
- Pass criteria:
  - auto/manual consolidation triggers pass
  - consolidated episodes retain raw text and link integrity
  - no episode rows are physically deleted

---

## Phase 3: Fact Contradiction Handling (#2)

**Changes to**: `ingest.py`, `db.py`

Detect when a new fact contradicts an existing one:
1. Extract facts as now
2. For each new fact, search existing active facts by entity overlap
3. If high entity overlap + conflicting content → mark old fact `contradiction_status = 'superseded'`, set `superseded_by`, set `valid_to = now`
4. New fact gets `valid_from = now`

**Detection heuristics** (no LLM needed):
- Negation patterns ("not", "no longer", "stopped")
- Value conflicts ("uses Python" vs "uses Rust" for same subject)
- Temporal supersession ("now uses" implies old value changed)

**Integration**:
- Part of `ingest.py` fact extraction pipeline
- Create `contradiction` graph edges between old and new facts

**Tests**: ~25 tests covering detection, supersession chain, temporal versioning, graph edges.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_ingest.py tests/test_db.py tests/test_linker.py -q`
- Pass criteria:
  - contradictory facts mark prior versions superseded
  - `valid_from`/`valid_to` intervals are correct
  - contradiction edges are created for linked facts

---

## Phase 4: Open Loop Resolution Improvement (#4)

**Changes to**: `ingest.py`

Enhance the existing resolution detection:
1. Entity-based matching — match open loops to assistant responses by shared entities
2. Action verb detection — "done", "completed", "fixed", "resolved", "sorted"
3. Contradiction-informed — if a fact supersession resolves an open question, auto-resolve the loop
4. Store resolution context (`resolution` column) — what resolved it and when
5. Set `resolved_by_episode_id`

**Integration**:
- Runs after contradiction handling in `sync_turn()` pipeline
- Create `resolves` graph edges from resolving episode to resolved loop

**Tests**: ~20 tests covering entity matching, resolution context, cross-referencing with contradictions.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_ingest.py tests/test_linker.py -q`
- Pass criteria:
  - open loops resolve from assistant actions and entity matches
  - `resolution` and `resolved_by_episode_id` are populated when appropriate
  - contradiction-driven loop closure works

---

## Phase 5: Temporal Awareness in Recall (#3)

**Changes to**: `recall.py`

Enhance the ranking formula with stronger temporal signals:
1. **Recency boost** — Configurable half-life decay (current: 30 days). Add session-recency: same-session memories get 2x boost.
2. **Temporal clustering** — When retrieving, prefer memories from the same time period as the query context.
3. **"When" query detection** — If query contains temporal markers ("last week", "yesterday", "in March"), adjust retrieval window.
4. **Timestamp display** — Use `epoch_to_display()` for human-readable time anchors in recall output.

**Tests**: ~20 tests covering temporal decay, session boost, temporal query detection, display formatting.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_recall.py tests/test_provider.py -q`
- Pass criteria:
  - temporal ranking changes retrieval order as expected
  - same-session boost works
  - displayed timestamps are human-readable and stable

---

## Phase 6: Affect Calibration Per User (#6)

**Changes to**: `affect.py`, new file `kortex/calibrate.py`

Replace hardcoded affect thresholds with per-user baselines:
1. **Baseline building** — EMA (exponential moving average) of raw affect dimensions over time
2. **Store in `affect_baselines` table** — One row per user, updated after each scored turn
3. **Calibrated scoring** — Raw score minus baseline = calibrated deviation. "User is MORE frustrated than their normal" instead of "frustration = 0.6".
4. **Minimum sample count** — Use hardcoded thresholds until N samples collected (configurable, default 20)
5. **Raw dimensions always stored separately** — Never calibrate-in-place. `emotion_log` stores raw; calibration is a query-time transform.

**Integration**:
- Update `score_affect()` to optionally accept baseline
- Update `sync_turn()` to update baseline after scoring
- Update `recall.py` to use calibrated scores for emotional weighting

**Tests**: ~25 tests covering baseline building, EMA update, calibrated scoring, minimum sample threshold.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_affect.py tests/test_calibrate.py tests/test_recall.py -q`
- Pass criteria:
  - baseline EMA updates correctly
  - calibration is derived from raw stored scores
  - pre-threshold fallback behavior remains correct

---

## Phase 7: Richer Entity Extraction (#7)

**New file**: `kortex/extract_llm.py`

LLM-assisted extraction via Hermes auxiliary client:
1. **Single batched prompt** — One LLM call extracts ALL types: entities, facts, reflections, open loops, affect hints
2. **Structured output** — Prompt returns JSON with sections for each extraction type
3. **Fallback** — If auxiliary client unavailable, fall back to existing heuristic extraction
4. **Entity enrichment** — Extract entity types, relationships, aliases (beyond regex)

**Integration**:
- New config option: `extraction_mode: "heuristic" | "llm" | "hybrid"` (default: "heuristic", user opts in)
- In hybrid mode: heuristic runs first (fast), LLM enriches in background
- Update `ingest.py` to dispatch to appropriate extractor

**Critical**: This is the single feature with the largest downstream impact on graph quality. Getting the extraction prompt right is essential.

**Tests**: ~30 tests covering LLM extraction, fallback, hybrid mode, structured output parsing, entity enrichment.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_extract_llm.py tests/test_ingest.py tests/test_config.py -q`
- Pass criteria:
  - heuristic fallback works when auxiliary client is unavailable
  - structured extraction parses successfully
  - hybrid mode preserves existing ingest behavior

---

## Phase 8: Memory Decay/Forgetting (#10)

**Changes to**: `recall.py`, `db.py`

Unaccessed memories fade over time:
1. **Decay formula**: `strength = salience * e^(-lambda * days_since_access) * (1 + retrieval_count * 0.2)`
2. **Category-specific rates**: Episodes λ=0.10, Facts λ=0.05 (slower — facts are durable), Reflections λ=0.08
3. **Retrieval refreshes** — Each access updates `last_accessed_at` and increments `retrieval_count`
4. **Tiered storage**: active (strength > 0.3) → warm (0.1-0.3) → cold (< 0.1). Cold memories excluded from default recall but available via explicit search.
5. **No deletion** — Memories never disappear, just become harder to surface.

**Integration**:
- Update `recall.py` scoring to incorporate decay
- Update `db.py` to touch `last_accessed_at`/`retrieval_count` on retrieval
- New config options: `decay_rates`, `decay_tiers`

**Tests**: ~25 tests covering decay calculation, retrieval refresh, tier classification, cold memory exclusion.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_recall.py tests/test_db.py -q`
- Pass criteria:
  - decay strength reflects last access time and retrieval count
  - retrieval updates access metadata
  - cold memories are omitted from default recall but remain searchable

---

## Phase 9: GraphRAG Traversal (#12)

**Changes to**: `linker.py`, `recall.py`

Multi-hop graph walks for richer recall:
1. **BFS traversal** — From seed entities in query, walk `entity_links` up to N hops (default 2)
2. **Score propagation** — Each hop reduces edge weight by configurable factor (default 0.5)
3. **Type-aware traversal** — Follow different edge types with different weights (`extracted_from` > `related_to` > `co_occurs_with`)
4. **Community detection** — Light clustering: entities that frequently co-occur in episodes form implicit communities
5. **Recall integration** — GraphRAG results merged with FTS5 results in `recall.py` via rank fusion

**Integration**:
- New method `Linker.traverse(entity_ids, max_hops=2)` → returns ranked connected nodes
- Update `Recall.build_context()` to include graph-expanded results
- Budget allocation: dedicate portion of context budget to graph-expanded memories

**Tests**: ~25 tests covering BFS traversal, score propagation, hop limiting, rank fusion.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_linker.py tests/test_recall.py -q`
- Pass criteria:
  - traversal respects max hop limits
  - score propagation decays by hop/edge type
  - graph-expanded recall merges with FTS results deterministically

---

## Phase 10: Memory Search Tool UX (#8)

**Changes to**: `provider.py`

Formatted, conversational responses instead of raw JSON:
1. **Narrative formatting** — "I remember 3 weeks ago when we discussed X. You were frustrated about Y..."
2. **Contextual grouping** — Group results by topic/time period
3. **Confidence indicators** — "I'm fairly certain..." vs "I vaguely recall..."
4. **Source attribution** — "From our conversation on [date]" with episode IDs for drill-down

**Integration**:
- New formatter functions in `provider.py` tool handlers
- Config option: `search_format: "json" | "narrative"` (default: "narrative")

**Tests**: ~15 tests covering formatting, grouping, confidence language, source attribution.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_provider.py -q`
- Pass criteria:
  - narrative output formatting is stable
  - grouping and confidence wording match expected snapshots/assertions
  - source attribution includes usable time anchors and IDs

---

## Phase 11: Multi-User Activation (#9)

**Changes to**: `provider.py`, `db.py`, `recall.py`, `ingest.py`

Schema is ready from Phase 0. This phase activates it:
1. **User ID propagation** — `initialize()` receives user context; propagate to all DB operations
2. **Scoped queries** — All SELECT queries add `WHERE user_id = ?`
3. **Scoped writes** — All INSERT/UPDATE set `user_id`
4. **Relationship per user** — Each user gets their own `relationship_state` row (already has `user_id`)
5. **Backward compatible** — `'__default__'` user ID works as before when no user context

**Integration**:
- `initialize(**kwargs)` extracts `user_id` from kwargs
- Pass `user_id` through all DB methods
- Update `recall.py` to scope by user

**Tests**: ~25 tests covering user isolation, scoped queries, backward compatibility, relationship per user.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_provider.py tests/test_db.py tests/test_recall.py tests/test_ingest.py -q`
- Pass criteria:
  - reads/writes are isolated by `user_id`
  - `'__default__'` mode remains backward compatible
  - relationship state is independent per user

---

## Phase 12: Export/Backup Tooling (#11)

**New file**: `kortex/export.py`

Full JSON dump/import for backup and migration:
1. **Export** — Dump all tables to structured JSON. Each table = one key. Timestamps as ISO8601 in export (human-readable), REAL internally.
2. **Import** — Load from JSON export. Handles schema version checking, deduplication.
3. **Selective export** — Export by user_id, date range, or memory type
4. **Tool integration** — New `kortex_export` action on search tool, or separate tool

**Integration**:
- New tool action or standalone tool
- Config: export directory path

**Tests**: ~20 tests covering full export/import roundtrip, selective export, version handling, dedup on import.

**Executable QA scenario**:
- Command: `source venv/bin/activate && python3 -m pytest tests/test_export.py tests/test_provider.py tests/test_db.py -q`
- Pass criteria:
  - full export/import roundtrip preserves rows and relationships
  - selective export filters correctly
  - import rejects incompatible schema versions cleanly

---

## Total Estimated Scope

| Phase | New Files | Modified Files | Est. Tests |
|---|---|---|---|
| 0 | 0 | 11+ all test files | 30 |
| 1 | 1 (summaries.py) | 2 (provider, recall) | 25 |
| 2 | 1 (consolidate.py) | 3 (provider, db, config) | 30 |
| 3 | 0 | 2 (ingest, db) | 25 |
| 4 | 0 | 1 (ingest) | 20 |
| 5 | 0 | 1 (recall) | 20 |
| 6 | 1 (calibrate.py) | 3 (affect, provider, recall) | 25 |
| 7 | 1 (extract_llm.py) | 2 (ingest, config) | 30 |
| 8 | 0 | 2 (recall, db) | 25 |
| 9 | 0 | 2 (linker, recall) | 25 |
| 10 | 0 | 1 (provider) | 15 |
| 11 | 0 | 4 (provider, db, recall, ingest) | 25 |
| 12 | 1 (export.py) | 1 (provider) | 20 |
| **Total** | **5 new files** | **widespread** | **~315 new tests** |

Final test count: ~394 (existing, updated) + ~315 (new) = ~709 total.

---

## Risk Mitigations

1. **Circular memory creation** — Tag KORTEX tool results with `[KORTEX_OUTPUT]` marker. Filter in `on_pre_compress()` and `sync_turn()` — skip ingestion of messages containing the marker.
2. **Auxiliary client bottleneck** (#7) — Single batched extraction prompt. Config option to disable. Default = heuristic only.
3. **Affect baseline contamination** (#6) — `emotion_log` always stores raw scores. Baselines computed from raw dimensions. Calibration is query-time, never write-time.
4. **FTS5 sync drift** — All writes go through single-path methods that fire triggers. Consolidation uses the same methods.
5. **Graph edge orphaning** — Never delete. Mark `is_consolidated`. Edges preserved.

---

## Verification Matrix

### Phase 0 QA (blocking before any later phase)
- Command: `source venv/bin/activate && python3 -m pytest tests/ -q`
- Additional migration-focused command: `source venv/bin/activate && python3 -m pytest tests/test_db.py -q -k "migration or schema or timestamp"`
- Pass criteria:
  - all existing tests pass after epoch migration updates
  - populated v2 fixture migrates successfully to v3
  - migrated timestamp columns are stored as `REAL`
  - required new columns/tables/indexes exist
  - `PRAGMA journal_mode` returns `wal`

### Final Verification Wave (run after every phase is complete)
1. `source venv/bin/activate && python3 -m pytest tests/ -q`
2. If any focused phase command exists above, run it again after the full suite.
3. Spot-check schema on a migrated DB fixture:
   - `sqlite3 <db-path> ".schema"`
   - `sqlite3 <db-path> "PRAGMA table_info(episodes);"`
   - `sqlite3 <db-path> "SELECT version FROM kortex_schema_version;"`
4. Pass criteria:
   - full suite green
   - no schema regression from previous phases
   - migration path from v2 remains valid

## Metis Session

Analysis source: `ses_278e3a31bffeByPkF8zwYYbxAf` (875 lines of dependency/schema/risk analysis)
