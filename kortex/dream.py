"""Kortex Dream State — Async offline memory maintenance.

Two tiers:
- DayDream: Quick pass after compaction (~30s). Keeps memory fresh.
- REMSleep: Deep pass when idle (~5min). Full optimization.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kortex.db import DEFAULT_USER_ID, KortexDB
from kortex.config import KortexConfig
from kortex.consolidate import Consolidator
from kortex.linker import Linker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DREAM] %(message)s")
logger = logging.getLogger("kortex.dream")


def get_db_stats(db: KortexDB) -> dict:
    return {
        "episodes": db.count_episodes(),
        "facts": db.count_facts(),
        "open_loops": db.count_open_loops(),
        "entity_links": db.count_links(),
    }


# ── DayDream: Quick maintenance (~30s) ──────────────────────────────────

def daydream(
    db_path: str,
    max_age_days: int = 7,
    fact_decay_days: int = 49,
    reflection_decay_days: int = 98,
) -> dict:
    """Quick maintenance pass. Runs after compaction.

    Does:
    - Expire stale open loops
    - Decay stale facts
    - Decay stale reflections
    - Light consolidation if threshold met
    """
    logger.info("☁️ Starting DayDream...")
    start_time = time.time()

    db = KortexDB(db_path)
    initial = get_db_stats(db)
    logger.info(f"Initial stats: {initial}")

    # Phase 1: Expire stale open loops
    logger.info("Phase 1 (Pruning): Expiring stale open loops...")
    expired = db.expire_old_loops(days_threshold=max_age_days)
    logger.info(f"  Expired {expired} stale loops")

    # Phase 2: Decay stale facts
    logger.info("Phase 2 (Decay): Decaying stale facts...")
    decayed = db.decay_stale_facts(days_threshold=fact_decay_days)
    logger.info(f"  Decayed {decayed} stale facts")

    # Phase 3: Decay stale reflections
    logger.info("Phase 3 (Reflections): Decaying stale reflections...")
    reflections_decayed = db.decay_stale_reflections(days_threshold=reflection_decay_days)
    logger.info(f"  Decayed {reflections_decayed} stale reflections")

    # Phase 4: Light consolidation (only if threshold met)
    logger.info("Phase 4 (Consolidation): Checking consolidation threshold...")
    config = KortexConfig()
    linker = Linker(db)
    consolidator = Consolidator(db, linker, config)
    result = consolidator.maybe_consolidate()
    if result.get("triggered"):
        logger.info(f"  Consolidated {result.get('episodes_consolidated', 0)} episodes")
        logger.info(f"  Created {result.get('summary_episodes_created', 0)} summary episodes")
    else:
        logger.info(f"  Threshold not met ({result.get('active_episodes', 0)} < {result.get('threshold', 0)})")

    final = get_db_stats(db)
    logger.info(f"Final stats: {final}")

    elapsed = time.time() - start_time
    logger.info(f"☁️ DayDream complete after {elapsed:.1f}s")

    db.close()

    return {
        "mode": "daydream",
        "elapsed_seconds": elapsed,
        "consolidation": result,
        "expired_loops": expired,
        "decayed_facts": decayed,
        "decayed_reflections": reflections_decayed,
        "initial_stats": initial,
        "final_stats": final,
    }


# ── REMSleep: Deep optimization (~5min) ────────────────────────────────

def rem_sleep(
    db_path: str,
    max_age_days: int = 7,
    fact_decay_days: int = 49,
    reflection_decay_days: int = 98,
) -> dict:
    """Deep optimization pass. Runs when system is idle (e.g. 3am).

    Does everything in DayDream PLUS:
    - Full forced consolidation
    - Database vacuum
    """
    logger.info("🌙 Starting REMSleep...")
    start_time = time.time()

    db = KortexDB(db_path)
    initial = get_db_stats(db)
    logger.info(f"Initial stats: {initial}")

    # Phase 1: Full forced consolidation
    logger.info("Phase 1 (REM): Full episode consolidation...")
    config = KortexConfig()
    linker = Linker(db)
    consolidator = Consolidator(db, linker, config)
    result = consolidator.consolidate(limit=200)
    logger.info(f"  Consolidated {result.get('episodes_consolidated', 0)} episodes")
    logger.info(f"  Created {result.get('summary_episodes_created', 0)} summary episodes")

    # Phase 2: Expire stale open loops
    logger.info("Phase 2 (Pruning): Expiring stale open loops...")
    expired = db.expire_old_loops(days_threshold=max_age_days)
    logger.info(f"  Expired {expired} stale loops")

    # Phase 3: Decay stale facts
    logger.info("Phase 3 (Decay): Decaying stale facts...")
    decayed = db.decay_stale_facts(days_threshold=fact_decay_days)
    logger.info(f"  Decayed {decayed} stale facts")

    # Phase 4: Decay stale reflections
    logger.info("Phase 4 (Reflections): Decaying stale reflections...")
    reflections_decayed = db.decay_stale_reflections(days_threshold=reflection_decay_days)
    logger.info(f"  Decayed {reflections_decayed} stale reflections")

    # Phase 5: Database optimization (vacuum + reindex + analyze + compound indexes)
    logger.info("Phase 5 (Optimize): Compacting and analyzing database...")
    db.optimize_database()
    logger.info("  Optimization complete")

    final = get_db_stats(db)
    logger.info(f"Final stats: {final}")

    elapsed = time.time() - start_time
    logger.info(f"☀️ REMSleep complete after {elapsed:.1f}s")

    db.close()

    return {
        "mode": "rem_sleep",
        "elapsed_seconds": elapsed,
        "consolidation": result,
        "expired_loops": expired,
        "decayed_facts": decayed,
        "decayed_reflections": reflections_decayed,
        "initial_stats": initial,
        "final_stats": final,
    }


def dream(db_path: str, max_age_days: int = 7):
    """Legacy alias for daydream."""
    return daydream(db_path, max_age_days=max_age_days)


def _audit_evidence(db_path: str, fact_id: int | None = None) -> dict:
    """Run evidence trace audit: orphaned facts, evidence summary, optional fact chain."""
    db = KortexDB(db_path)
    results = {}
    results["orphaned_facts"] = db.get_orphaned_facts()
    results["evidence_summary"] = db.get_evidence_summary(limit=50)
    if fact_id:
        chain = db.get_fact_evidence_chain(fact_id)
        results["fact_chain"] = chain if chain else {"error": f"fact_id {fact_id} not found"}
    return results


def batch_embed(db_path: str, user_id: str = DEFAULT_USER_ID) -> dict:
    """Batch-embed all unembedded episodes and facts.

    Run this once to backfill embeddings for existing data,
    or periodically to ensure full coverage.
    """
    logger.info("🔢 Starting batch embedding migration...")
    start_time = time.time()

    db = KortexDB(db_path)
    try:
        from kortex.semantic import SemanticSearch
        search = SemanticSearch(db)
        search.build_vocab()

        eps = search.batch_embed_episodes(user_id=user_id)
        logger.info(f"  Embedded {eps} episodes")

        facts = search.batch_embed_facts(user_id=user_id)
        logger.info(f"  Embedded {facts} facts")

        elapsed = time.time() - start_time
        logger.info(f"🔢 Batch embedding complete: {eps} episodes, {facts} facts in {elapsed:.1f}s")
        return {"mode": "batch_embed", "episodes": eps, "facts": facts, "elapsed_seconds": elapsed}
    except Exception:
        logger.exception("Batch embedding failed")
        return {"mode": "batch_embed", "error": str(Exception)}
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Kortex Dream State")
    parser.add_argument(
        "mode",
        choices=["daydream", "rem", "audit-evidence", "batch-embed"],
        help="Dream mode: 'daydream' (quick), 'rem' (deep), 'audit-evidence' (fact evidence), 'batch-embed' (backfill embeddings)"
    )
    parser.add_argument("--db", default=os.path.expanduser("~/.hermes/kortex.db"))
    parser.add_argument("--max-age-days", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fact-id", type=int, default=None,
                        help="Fact ID for evidence chain lookup (audit-evidence mode)")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID,
                        help="User ID for batch embedding")
    args = parser.parse_args()

    if args.mode == "daydream":
        result = daydream(args.db, max_age_days=args.max_age_days)
    elif args.mode == "rem":
        result = rem_sleep(args.db, max_age_days=args.max_age_days)
    elif args.mode == "batch-embed":
        result = batch_embed(args.db, user_id=args.user_id)
    else:
        result = _audit_evidence(args.db, fact_id=args.fact_id)

    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
