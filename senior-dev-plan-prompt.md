# Senior Developer Code Review & Cleanup Plan

## Context

You are a senior Python developer with 15 years of experience. You've been asked to plan a cleanup and modernization pass on **KORTEX**, a Python-based personal memory and context management system for an AI assistant (Hermes).

## Tech Stack

- **Language:** Python 3.11+
- **Database:** SQLite with FTS5 full-text search
- **Architecture:** Single `kortex/` package (~9,834 LOC across 18 modules) + `tests/` (592 tests)
- **Key modules:** `db.py` (2,906 LOC — the monolith), `provider.py` (947 LOC), `recall.py` (827 LOC), `ingest.py` (871 LOC), `context_engine.py` (523 LOC), `reflect.py` (489 LOC)
- **Dependencies:** minimal — no framework, just stdlib + sqlite3

## What Already Happened

A prior pass did some easy wins:
- Consolidated duplicate `_ema` / `_clamp` utilities into `time_utils.py`
- Simplified content filtering in `provider.py` (51 → 16 lines)
- Deduplicated `calibrate_affect` calls in `recall.py`
- Added `_iso_or_none` helper in `export.py`

## Your Task

Plan the next phase of cleanup. The goal is to make the codebase **easier to upgrade, fix, and extend** — not just smaller.

Consider:
1. **Dead/vibe-coded code** — things that look like they were written to "just make it work" and should be rewritten properly
2. **Architecture issues** — `db.py` at 2,906 LOC is a red flag; what should be extracted?
3. **Inconsistent patterns** — same thing done different ways in different files
4. **Missing abstractions** — duplicated logic that should be shared but isn't
5. **Test gaps** — areas with no test coverage that are risky to touch
6. **The big files** — `db.py`, `provider.py`, `recall.py`, `context_engine.py` all over 500 LOC

## Deliverables

For each major area you identify, provide:
- **What the problem is** (specific, with file:line references)
- **Why it matters** (maintainability, correctness, or upgrade risk)
- **Recommended fix** (specific approach, not vague advice)
- **Estimated effort** (low/medium/high)

Keep it actionable — a senior dev reading this should be able to start tomorrow.
