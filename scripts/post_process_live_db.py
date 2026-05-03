#!/usr/bin/env python3
"""Post-process script for live KORTEX database.

Backfills embeddings for all existing episodes and facts,
validates semantic search works, and tests recall context building.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kortex.db import KortexDB
from kortex.semantic import SemanticSearch


def main():
    db_path = os.path.expanduser("~/.hermes/kortex.db")
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    print(f"=== KORTEX Live DB Post-Process ===")
    print(f"DB: {db_path}")
    print()

    db = KortexDB(db_path)
    conn = db._get_conn()

    # 1. Schema info
    uv = conn.execute("PRAGMA user_version").fetchone()[0]
    print(f"Schema version: {uv}")

    # 2. Current state
    ep_total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    emb_ep = conn.execute("SELECT COUNT(*) FROM embeddings WHERE entity_type='episode'").fetchone()[0]
    emb_fact = conn.execute("SELECT COUNT(*) FROM embeddings WHERE entity_type='fact'").fetchone()[0]
    print(f"Episodes: {ep_total}, Episode embeddings: {emb_ep}, Fact embeddings: {emb_fact}")

    # 3. User IDs
    user_ids = [r[0] for r in conn.execute("SELECT DISTINCT user_id FROM episodes LIMIT 10").fetchall()]
    print(f"User IDs: {user_ids}")
    print()

    # 4. Batch embed for all users
    print("Running batch embedding...")
    for uid in user_ids:
        search = SemanticSearch(db)
        search.build_vocab()
        eps = search.batch_embed_episodes(user_id=uid)
        facts = search.batch_embed_facts(user_id=uid)
        print(f"  {uid}: {eps} episodes, {facts} facts embedded")

    # 5. Verify embeddings
    print()
    print("Verifying embeddings...")
    emb_ep = conn.execute("SELECT COUNT(*) FROM embeddings WHERE entity_type='episode'").fetchone()[0]
    emb_fact = conn.execute("SELECT COUNT(*) FROM embeddings WHERE entity_type='fact'").fetchone()[0]
    print(f"Episode embeddings: {emb_ep}/{ep_total}")
    print(f"Fact embeddings: {emb_fact}")

    # 6. Test semantic search
    print()
    print("Testing semantic search...")
    search = SemanticSearch(db)
    search.build_vocab()

    for uid in user_ids:
        results = search.search_episodes_hybrid("python", limit=5, user_id=uid, min_similarity=0.1)
        print(f"  Query 'python' for {uid}: {len(results)} results")
        if results:
            top = results[0]
            print(f"    Top: score={top.get('hybrid_score'):.3f}, text={top.get('user_text', '')[:60]!r}")

    # 7. Test context building
    print()
    print("Testing context building...")
    from kortex.config import KortexConfig
    from kortex.recall import Recall
    from kortex.linker import Linker

    config = KortexConfig()
    linker = Linker(db)
    recall = Recall(db, config, linker=linker)

    for uid in user_ids:
        ctx = recall.build_context("testing", session_id="", user_id=uid, lightweight=True)
        print(f"  Context for {uid} (lightweight): {len(ctx)} chars")
        if ctx:
            print(f"    Preview: {ctx[:200]!r}...")

    # 8. Test non-lightweight context
    graph_budget = config.budget.get("graph", 300)
    ctx_full = recall.build_context("testing", session_id="", user_id=user_ids[0], lightweight=False)
    print(f"  Context for {user_ids[0]} (full): {len(ctx_full)} chars")

    print()
    print("=== Post-process complete ===")

    db.close()


if __name__ == "__main__":
    main()
