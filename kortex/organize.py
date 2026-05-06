"""Memory organization engine for KORTEX.

Provides hierarchical memory navigation:
- Query by domain, category, or topic
- Navigate from broad (domain) to specific (topic)
- Discover related facts across levels

Integrates with the topics module for classification.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional

from .topics import (
    classify_fact,
    get_all_categories,
    get_all_domains,
    get_category_domain,
    get_domain_categories,
)

logger = logging.getLogger(__name__)


class MemoryOrganizer:
    """Organizes and navigates hierarchical memory."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

    def close(self):
        self._conn.close()

    def get_topics_summary(self, user_id: str = "") -> Dict[str, Any]:
        """Get a summary of all topics organized hierarchically.
        
        Returns: {
            "domains": {
                "personal": {
                    "categories": {
                        "identity": {
                            "facts": [...],
                            "count": N,
                            "avg_confidence": X
                        }
                    }
                }
            },
            "total_facts": N,
            "total_topics": N
        }
        """
        conn = self._conn
        cursor = conn.cursor()
        
        where_clause = "WHERE 1=1"
        params = []
        if user_id:
            where_clause += " AND user_id = ?"
            params.append(user_id)

        # Get all facts with their classifications
        try:
            cursor.execute(f"""
                SELECT id, predicate, object_text, confidence, user_id
                FROM facts {where_clause}
                ORDER BY confidence DESC
            """, params)
            rows = cursor.fetchall()
        except Exception:
            # Fallback if table structure differs
            return {"domains": {}, "total_facts": 0, "total_topics": 0}

        # Classify and organize
        domains = {}
        topic_set = set()

        for row in rows:
            fact_id, predicate, object_text, confidence, fact_user_id = row
            classification = classify_fact(object_text, predicate)
            
            domain = classification["domain"]
            category = classification["category"]
            topic = classification["topic"]
            
            topic_set.add(topic)
            
            if domain not in domains:
                domains[domain] = {"categories": {}}
            
            if category not in domains[domain]["categories"]:
                domains[domain]["categories"][category] = {
                    "facts": [],
                    "count": 0,
                    "avg_confidence": 0,
                    "total_confidence": 0,
                }
            
            cat_data = domains[domain]["categories"][category]
            cat_data["facts"].append({
                "id": fact_id,
                "predicate": predicate,
                "object_text": object_text,
                "confidence": confidence,
                "topic": topic,
            })
            cat_data["count"] += 1
            cat_data["total_confidence"] += (confidence or 0)

        # Calculate averages
        for domain in domains.values():
            for cat_data in domain["categories"].values():
                if cat_data["count"] > 0:
                    cat_data["avg_confidence"] = round(
                        cat_data["total_confidence"] / cat_data["count"], 2
                    )
                del cat_data["total_confidence"]

        return {
            "domains": domains,
            "total_facts": len(rows),
            "total_topics": len(topic_set),
        }

    def get_domain_facts(self, domain: str, user_id: str = "") -> List[Dict[str, Any]]:
        """Get all facts in a domain."""
        return self._get_classified_facts(lambda c: c["domain"] == domain, user_id)

    def get_category_facts(self, category: str, user_id: str = "") -> List[Dict[str, Any]]:
        """Get all facts in a category."""
        return self._get_classified_facts(lambda c: c["category"] == category, user_id)

    def get_topic_facts(self, topic: str, user_id: str = "") -> List[Dict[str, Any]]:
        """Get all facts on a specific topic."""
        return self._get_classified_facts(lambda c: c["topic"] == topic, user_id)

    def _get_classified_facts(
        self, filter_fn, user_id: str = ""
    ) -> List[Dict[str, Any]]:
        """Get facts filtered by classification."""
        conn = self._conn
        cursor = conn.cursor()
        
        where_clause = "WHERE 1=1"
        params = []
        if user_id:
            where_clause += " AND user_id = ?"
            params.append(user_id)

        cursor.execute(f"""
            SELECT id, predicate, object_text, confidence
            FROM facts {where_clause}
            ORDER BY confidence DESC
        """, params)
        
        results = []
        for row in cursor.fetchall():
            classification = classify_fact(row["object_text"], row["predicate"])
            if filter_fn(classification):
                results.append({
                    "id": row["id"],
                    "predicate": row["predicate"],
                    "object_text": row["object_text"],
                    "confidence": row["confidence"],
                    "classification": classification,
                })
        
        return results

    def discover_topics(self, query: str, user_id: str = "") -> List[str]:
        """Discover relevant topics for a query string.
        
        Compares the query against known fact texts to surface topics.
        """
        conn = self._conn
        cursor = conn.cursor()
        
        where_clause = "WHERE 1=1"
        params = []
        if user_id:
            where_clause += " AND user_id = ?"
            params.append(user_id)

        # Find facts matching the query
        cursor.execute(f"""
            SELECT predicate, object_text, confidence
            FROM facts {where_clause}
            WHERE object_text LIKE ? OR predicate LIKE ?
            ORDER BY confidence DESC
            LIMIT 10
        """, params + [f"%{query}%", f"%{query}%"])
        
        topics = set()
        for row in cursor.fetchall():
            classification = classify_fact(row["object_text"], row["predicate"])
            topics.add(classification["topic"])
        
        return list(topics)

    def navigate_from_topic(
        self, topic: str, direction: str = "broader", user_id: str = ""
    ) -> Dict[str, Any]:
        """Navigate memory hierarchy from a topic.
        
        direction: "broader" (category → domain), "narrower" (domain → category → topic)
        """
        # Find a sample fact for this topic
        sample_facts = self.get_topic_facts(topic, user_id)
        
        if not sample_facts:
            return {"topic": topic, "facts": [], "navigation": {}}
        
        classification = sample_facts[0]["classification"]
        
        if direction == "broader":
            domain = classification["domain"]
            return {
                "topic": topic,
                "category": classification["category"],
                "domain": domain,
                "related_topics": self._get_sibling_topics(classification["category"], user_id),
                "facts": sample_facts[:5],
            }
        else:
            # Narrower: find sub-topics within this category
            category = classification["category"]
            return {
                "category": category,
                "sub_topics": self._get_sub_topics(category, user_id),
            }

    def _get_sibling_topics(self, category: str, user_id: str = "") -> List[str]:
        """Get sibling topics within the same category."""
        facts = self.get_category_facts(category, user_id)
        topics = set()
        for fact in facts:
            if "classification" in fact:
                topics.add(fact["classification"]["topic"])
        return list(topics)

    def _get_sub_topics(self, category: str, user_id: str = "") -> List[str]:
        """Get sub-topics within a category."""
        return self._get_sibling_topics(category, user_id)
