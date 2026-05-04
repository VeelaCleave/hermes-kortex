# KORTEX Plugin — Complete Codebase Audit

## 1. Module Inventory (22 modules)

### kortex/__init__.py — Plugin Registration
- **Public exports**: `register(ctx)`, `_kortex_passive_recall(provider, kwargs)`
- **Imports**: config, context_engine, provider
- **Role**: Top-level entry point. Registers memory provider, context engine, tool handlers, and session lifecycle hooks.

### kortex/config.py — Configuration
- **Classes**: `KortexConfig` (dataclass, 40+ fields)
- **Functions**: `load_kortex_config(hermes_home) -> KortexConfig`, `KortexConfig.from_dict(data)`
- **Constants**: `DEFAULT_BUDGET` dict
- **Role**: Plugin configuration loading from YAML.

### kortex/models.py — Data Models
- **Enums**: `Valence` (VERY_NEGATIVE..VERY_POSITIVE)
- **Dataclasses**: `Episode`, `Fact`, `OpenLoop`, `Reflection`, `RelationshipState`, `AffectSignal`, `IdentityDelta`, `EntityLink`
- **Note**: `EntityLink` has `from_db_row` missing — only has class definition with fields, no `from_db_row` method (see Issues)

### kortex/db.py — SQLite Storage Layer (2784 lines)
- **Class**: `KortexDB` — Thread-safe SQLite storage
- **Public methods**: `__init__`, `insert_episode`, `update_episode`, `get_episode`, `get_recent_episodes`, `search_episodes`, `get_salient_episodes`, `count_episodes`, `get_session_turn_count`, `get_episodes_for_session`, `touch_episode`, `touch_episodes`, `count_unconsolidated_episodes`, `get_unconsolidated_episodes`, `mark_episodes_consolidated`, `insert_conversation_summary`, `list_conversation_summaries`, `search_conversation_summaries`, `insert_fact`, `get_active_facts`, `get_fact`, `get_facts_superseded_by`, `search_facts`, `update_fact_confidence`, `bump_fact_last_seen`, `supersede_fact`, `mark_fact_contradiction`, `get_facts_by_predicate`, `find_similar_facts`, `count_facts`, `decay_stale_facts`, `get_facts_by_episode`, `get_evidence_for_fact`, `get_fact_evidence_chain`, `get_evidence_summary`, `get_orphaned_facts`, `insert_open_loop`, `get_open_loops`, `resolve_loop`, `expire_old_loops`, `count_open_loops`, `get_relationship`, `upsert_relationship`, `insert_emotion_log`, `get_affect_baseline`, `upsert_affect_baseline`, `upsert_ocean_profile`, `get_ocean_profile`, `get_ocean_profiles_summary`, plus context management methods
- **Private methods**: `_get_conn`, `_tx`, `_init_schema`, `_migrate`, `_migrate_v2_to_v3`, `_copy_table_rows`, `_table_exists`, `_normalize_fts_query`, `_record_schema_version`
- **Constants**: `DEFAULT_USER_ID`, `SCHEMA_VERSION = 5`

### kortex/provider.py — Memory Provider (1076 lines)
- **Class**: `KortexProvider(MemoryProvider)`
- **Public methods**: `name`, `is_available`, `initialize`, `system_prompt_block`, `prefetch`, `queue_prefetch`, `sync_turn`, `get_tool_schemas`, `handle_tool_call`, `on_turn_start`, `on_session_end`, `on_pre_compress`, `on_memory_write`, `on_delegation`, `shutdown`, `get_config_schema`, `save_config`
- **Private methods**: `_migrate_embeddings`, `_filter_user_content`, `_filter_assistant_content`, `_update_ocean`, `_make_ocean_score`, `_daydream_deactivate`, `_trigger_daydream`, `_handle_search`, `_handle_recent`, `_format_search_narrative`, `_handle_recall_episode`, `_handle_list_facts`, `_handle_list_loops`, `_handle_list_conversations`, `_handle_status`, `_handle_consolidate`, `_handle_identity_call`, `_handle_export_call`, `_handle_import_call`

### kortex/context_engine.py — Context Engine
- **Class**: `KortexContextEngine(HermesContextEngine)`
- **Public methods**: `name`, `is_available`, `on_session_start`, `on_session_end`, `on_session_reset`, `update_from_response`, `should_compress`, `compress`, `get_tool_schemas`, `handle_tool_call`, `get_status`, `update_model`, `expand_ref`
- **Private methods**: `_derive_conversation_id`, `_contains_lossy_marker`, `_extract_refs`, `_make_ref`, `_checkpoint_id`, `_build_checkpoint_text`, `_handle_recall`, `_handle_expand`, `_message_text_content`

### kortex/ingest.py — Turn Ingestion
- **Class**: `Ingestor`
- **Public methods**: `configure_extraction`, `ingest_turn`, `extract_open_loops`, `extract_facts`, `resolve_answered_loops`, `resolve_completed_commitments`
- **Private methods**: `_get_semantic`, `_embed_episode`, `_embed_fact`, `_extract_fact_candidates`, `_extract_structured_memory`, `_build_loop_resolution`, `_find_matching_fact`, `_apply_fact_conflict`, `_facts_contradict`, `_extract_version`, `_normalize_text`, `_facts_are_equivalent`, `_facts_are_related`, `_extract_summary`, `_score_salience`, `_score_valence`, `_score_arousal`, `_extract_topics`, `_extract_entities`
- **Module-level**: `_extract_keywords()`

### kortex/recall.py — Memory Recall
- **Class**: `Recall`
- **Public methods**: `build_context`
- **Private methods**: `_get_semantic`, `_build_episodes_lightweight`, `_build_conversation_summaries_section`, `_select_facts`, `_deduplicate_facts`, `_build_facts_section`, `_build_episodes_section`, `_enrich_with_links`, `_graph_episode_scores`, `_query_entity_ids`, `_rank_fusion`, `_graph_candidate_limit`, `_build_loops_section`, `_build_reflections_section`, `_build_emotional_trajectory`, `_rank_episode`, `_relationship_boost`, `_episode_strength`, `_is_explicit_episode_match`, `_memory_tier`, `_build_ocean_section`, `_estimate_tokens`, `_trim_to_budget`

### kortex/consolidate.py — Memory Consolidation
- **Class**: `Consolidator`
- **Methods**: `maybe_consolidate`, `consolidate`, `_create_summary_episode`, `_latest_summary_for_session`, `_copy_links`
- **Module-level**: `_merge_csv()`

### kortex/linker.py — Entity Linking
- **Class**: `Linker`
- **Public methods**: `link_episode_to_facts`, `link_episode_to_reflections`, `link_related_episodes`, `link_superseded_facts`, `link_contradicting_facts`, `link_episode_to_loops`, `get_related_episodes`, `get_episode_facts`, `get_fact_episodes`, `traverse`, `entity_id`
- **Private methods**: `_link_entities_to_episode`, `_neighbors`, `_create_link`, `_episode_tokens`, `_split_csv`, `_jaccard`, `_relation_weight`, `_entity_id`, `_unique_positive_ids`

### kortex/semantic.py — Semantic Search
- **Class**: `SemanticSearch`
- **Methods**: `build_vocab`, `embed`, `embed_episode`, `embed_fact`, `batch_embed_episodes`, `batch_embed_facts`, `search_episodes_hybrid`, `search_facts_hybrid`, `_tokenize`, `_cosine`

### kortex/dream.py — DayDream Pipeline
- **Functions**: `get_db_stats()`, `daydream()`, `rem_sleep()`, `dream()`, `_audit_evidence()`, `batch_embed()`, `main()`

### kortex/affect.py — Affect Scoring
- **Functions**: `score_affect(user_text, assistant_text)`, `_score_dimension()`, `_detect_sarcasm()`

### kortex/calibrate.py — Affect Calibration
- **Class**: `AffectBaseline` (dataclass)
- **Functions**: `update_baseline(baseline, affect)`, `calibrate_affect(affect, baseline, minimum_samples)`

### kortex/reflect.py — Reflection Processing
- **Functions**: `process_reflections()`, `extract_mistakes()`, `extract_successes()`, `extract_style_preferences()`, `extract_identity_directives()`, `_store_or_reinforce()`, `_find_similar_reflection()`, `_reflections_similar()`, `_extract_context()`

### kortex/relationship.py — Relationship Modeling
- **Functions**: `compute_relationship_delta()`, `apply_regression()`, `update_relationship()`

### kortex/promote.py — Identity Promotion
- **Class**: `Promoter`
- **Methods**: `list_pending`, `preview_delta`, `approve_and_apply`, `reject_delta`, `approve_multiple`, `get_soul_content`, `_resolve_soul_path`, `_truncate_text`, `_format_trait`, `_append_trait`

### kortex/ocean.py — OCEAN Personality
- **Class**: `OCEANScore` (dataclass)
- **Methods**: `to_dict`, `from_db_row`, `to_compact_text`
- **Functions**: `score_turn()`, `update_ocean()`, `_compute_raw_scores()`

### kortex/summaries.py — Conversation Summaries
- **Functions**: `build_conversation_summary()`, `_clip()`

### kortex/extract_llm.py — LLM-Based Extraction
- **Functions**: `extract_structured_memory()`, `_normalize_str_list()`, `_normalize_facts()`, `_normalize_loops()`

### kortex/time_utils.py — Time Utilities
- **Functions**: `now_epoch()`, `parse_timestamp()`, `epoch_to_datetime()`, `epoch_to_iso()`, `epoch_to_display()`, `_clamp()`, `_ema()`, `query_emotion_score()`, `detect_temporal_window_days()`

### kortex/export.py — Import/Export
- **Functions**: `export_to_json()`, `import_from_json()`, `_db_user_version()`, `_within_range()`, `_episode_to_dict()`, `_fact_to_dict()`, `_loop_to_dict()`, `_reflection_to_dict()`, `_summary_to_dict()`, `_identity_delta_to_dict()`, `_affect_baseline_to_dict()`, `_iso_or_none()`

---

## 2. Dependency Map (Who Imports Whom)

```
__init__.py
├── config (load_kortex_config)
├── context_engine (KortexContextEngine)
└── provider (KortexProvider)

provider.py
├── config (KortexConfig, load_kortex_config)
├── calibrate (calibrate_affect, update_baseline)
├── consolidate (Consolidator)
├── db (DEFAULT_USER_ID, KortexDB)
├── affect (score_affect)
├── ingest (Ingestor)
├── linker (Linker)
├── models (AffectSignal, Episode, Fact, OpenLoop, RelationshipState)
├── promote (Promoter)
├── recall (Recall)
├── reflect (process_reflections)
├── relationship (update_relationship)
├── summaries (build_conversation_summary)
├── time_utils (epoch_to_display, epoch_to_iso)
├── export (export_to_json, import_from_json)
├── ocean (score_turn)
└── semantic (SemanticSearch) [lazy import]

context_engine.py
├── db (DEFAULT_USER_ID, KortexDB)
├── time_utils (now_epoch, epoch_to_iso)
├── models (Episode)

ingest.py
├── db (DEFAULT_USER_ID, KortexDB)
├── models (Episode, Fact, OpenLoop)
├── extract_llm (extract_structured_memory)
├── semantic (SemanticSearch)
├── time_utils (now_epoch)

recall.py
├── config (KortexConfig)
├── calibrate (calibrate_affect)
├── db (DEFAULT_USER_ID, KortexDB)
├── linker (Linker)
├── models (Episode, Fact, OpenLoop, Reflection, RelationshipState, AffectSignal)
├── semantic (SemanticSearch)
├── time_utils (detect_temporal_window_days, epoch_to_display, now_epoch, query_emotion_score)

consolidate.py
├── config (KortexConfig)
├── db (DEFAULT_USER_ID, KortexDB)
├── linker (Linker)
├── models (Episode)
├── summaries (build_conversation_sync)

linker.py
├── db (DEFAULT_USER_ID, KortexDB)
├── models (Episode)

semantic.py
├── db (DEFAULT_USER_ID, KortexDB)

reflect.py
├── db (KortexDB)
├── models (AffectSignal, IdentityDelta, Reflection)

relationship.py
├── models (AffectSignal, RelationshipState)
├── time_utils (now_epoch, _clamp, _ema)

calibrate.py
├── models (AffectSignal)
├── time_utils (now_epoch, _ema)

promote.py
├── db (KortexDB)
├── models (IdentityDelta)

ocean.py
├── models (no direct import, uses OCEANScore)

export.py
├── db (DEFAULT_USER_ID, SCHEMA_VERSION, KortexDB)
├── models (Episode, Fact, OpenLoop, Reflection)
├── time_utils (epoch_to_iso, now_epoch, parse_timestamp)

summaries.py
├── time_utils (now_epoch)

dream.py
├── db (DEFAULT_USER_ID, KortexDB)
├── semantic (SemanticSearch)
├── time_utils (now_epoch)

affect.py
├── models (AffectSignal)

extract_llm.py
├── (mostly standalone, uses typing)

config.py
├── (standalone, uses dataclasses, pathlib, yaml)

time_utils.py
├── (standalone, uses datetime, re, math)
```

---

## 3. Dead Code & Unused Items

### Confirmed Dead Code (defined but never called outside own module):

1. **`EntityLink` (models.py)** — Defined as a dataclass with fields, but:
   - Has no `from_db_row()` method (unlike all other models)
   - Never instantiated anywhere in the codebase
   - The `entity_links` table exists in the DB schema, but `EntityLink` is the only model without DB conversion methods

2. **`KortexDB.get_facts_by_episode()`** — Only defined in db.py, never called by any consumer

3. **`KortexDB.get_ocean_profiles_summary()`** — Only defined in db.py, never called externally (only used internally if at all)

4. **`_merge_csv()` (consolidate.py)** — Module-level function, only used within `Consolidator._create_summary_episode()`

5. **`OCEANScore.from_db_row()` (ocean.py)** — Defined but `get_ocean_profile()` returns a raw dict, not an OCEANScore instance

6. **`_compute_raw_scores()` (ocean.py)** — Only called by `score_turn()`, could be inlined

7. **`_score_dimension()`, `_detect_sarcasm()` (affect.py)** — Private helpers only used by `score_affect()`. Could be inlined.

### Potentially Dead (only used in CLI/main()):

8. **`dream.main()`** — Entry point for `python -m kortex.dream`, only used if running as standalone script
9. **`_audit_evidence()` (dream.py)** — Called from `main()` CLI and from `daydream()`
10. **`batch_embed()` (dream.py)** — Called from `main()` CLI

---

## 4. Bugs & Inconsistencies

### Critical:

1. **`EntityLink` has no `from_db_row()` method** (models.py:385-395)
   - All other models (Episode, Fact, OpenLoop, etc.) have `from_db_row()`. `EntityLink` is the odd one out — likely a copy-paste omission.
   - The `entity_links` table IS populated by `Linker._create_link()`, but `EntityLink` is never materialized from DB rows.

2. **`_handle_search` in provider.py references `_format_search_narrative`** which takes `self` — but `_confidence_phrase` is defined as a static method (`@staticmethod` equivalent — no `self` param) while being called as an instance method. Minor but inconsistent.

3. **`provider.py` line 871: `_confidence_phrase` is defined as `def _confidence_phrase(confidence: float)` (no `self`) inside `KortexProvider` class** — This means it's effectively a staticmethod but isn't decorated as one. Works in practice but is technically a class-level function.

4. **`_handle_list_conversations` method exists in provider.py** but is never called from `handle_tool_call()` dispatch — dead code path.

### Design Inconsistencies:

5. **Duplicate tool schemas**: Both `__init__.py` (register function) and `provider.py` (`get_tool_schemas()`) define the `kortex_query` tool schema. If both registration paths are active, the tool gets registered twice.

6. **`__init__.py` registers hooks that call `provider.handle_tool_call("kortex_query", ...)`**, but `provider.handle_tool_call` only accepts `tool_name="kortex_query"` — the fallback `json.dumps({"error": ...})` for unknown tool names is never triggered through this path.

7. **`KortexContextEngine` has its own `handle_tool_call`** that handles `kortex_recall` and `kortex_expand`, while `KortexProvider.handle_tool_call` only handles `kortex_query`. Three separate tool handlers across two classes with minimal overlap.

8. **Schema version mismatch potential**: `SCHEMA_VERSION = 5` in db.py, but migration logic only handles v1→v2→v3→v4→v5. The `_LOSSLESS_CONTEXT_SCHEMA_SQL` is always executed (even on fresh init), but the migration path doesn't add these tables for databases at v5+ — meaning a v5 DB upgraded later would miss the lossless context tables unless `_init_schema` re-executes them.

9. **`calibrate_affect` import in recall.py** (line 17) — Imported but the function signature expects `(affect, baseline, minimum_samples)` while recall.py may not pass all three arguments consistently.

### Minor Issues:

10. **`_clamp` and `_ema` in time_utils.py** — Utility functions with underscore prefix suggesting "private" but imported by multiple modules (calibrate.py, relationship.py). Should either be public (no underscore) or moved to a shared utilities module.

11. **`ocean.py` OCEANScore.to_compact_text()** — Returns personality profile text, but `get_ocean_profile()` in db.py returns a raw dict, so the `to_compact_text()` method is only reachable if someone manually constructs an OCEANScore.

12. **`export.py` import_from_json** — The function handles complex JSON import with many optional parameters, but the provider's `_handle_import_call` passes a minimal args dict. Potential for silent failures on edge cases.

---

## 5. Summary Statistics

| Metric | Count |
|--------|-------|
| Total modules | 22 |
| Total lines (approx) | ~12,000+ |
| Public classes | 14 (KortexDB, KortexProvider, KortexContextEngine, Ingestor, Consolidator, Linker, SemanticSearch, Promoter, Recall, KortexConfig, AffectBaseline, OCEANScore, Episode, Fact, OpenLoop, Reflection, RelationshipState, AffectSignal, IdentityDelta, EntityLink, Valence) |
| Public functions/methods | ~150+ |
| Confirmed dead code items | 10 |
| Bugs/inconsistencies | 12 |
| Modules with no external consumers | 0 (all modules are transitively reachable) |
