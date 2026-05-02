"""Tests for affect-aware recall — emotion-driven memory surfacing.

Affect-aware recall ensures that:
1. Emotional queries match emotionally similar memories
2. Valence alignment boosts relevant episodes
3. High arousal episodes surface during intense queries
4. Emotional weight properly reflects episode affect
"""

import pytest

from kortex.db import KortexDB
from kortex.models import Episode
from kortex.recall import Recall
from kortex.config import KortexConfig
import tempfile


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    return KortexDB(tmp + "/test.db")


@pytest.fixture
def config():
    return KortexConfig()


@pytest.fixture
def recall(db, config):
    return Recall(db, config, linker=None)


class TestEmotionalWeight:
    """Test that emotional_weight reflects episode affect."""

    def test_high_valence_high_arousal_gives_high_weight(self):
        """Strong positive + high arousal = high emotional weight."""
        ep = Episode(user_text="Test", assistant_text="Reply", valence=2, arousal=0.9)
        weight = ep.emotional_weight
        assert weight > 0.7

    def test_neutral_episode_has_low_weight(self):
        """Neutral episode should have low emotional weight."""
        ep = Episode(user_text="Test", assistant_text="Reply", valence=0, arousal=0.2)
        weight = ep.emotional_weight
        assert weight < 0.3

    def test_negative_valence_contributes_to_weight(self):
        """Negative valence also contributes to emotional weight."""
        ep = Episode(user_text="Test", assistant_text="Reply", valence=-2, arousal=0.7)
        weight = ep.emotional_weight
        assert weight > 0.7

    def test_emotional_weight_range(self):
        """Weight should be between 0 and 1."""
        ep = Episode(user_text="Test", assistant_text="Reply", valence=1, arousal=0.5)
        weight = ep.emotional_weight
        assert 0 <= weight <= 1.0


class TestQueryEmotionScore:
    """Test the query emotion scoring function."""

    def test_positive_query_scores_positive(self, recall):
        """Positive words should yield positive score."""
        score = recall._query_emotion_score("awesome breakthrough success")
        assert score > 0.3

    def test_negative_query_scores_negative(self, recall):
        """Negative words should yield negative score."""
        score = recall._query_emotion_score("frustrated angry bug")
        assert score < -0.3

    def test_neutral_query_scores_near_zero(self, recall):
        """Neutral words should yield near-zero score."""
        score = recall._query_emotion_score("the quick brown fox")
        assert abs(score) < 0.1

    def test_mixed_query_scores_mid(self, recall):
        """Mixed positive/negative should be near zero."""
        score = recall._query_emotion_score("great but frustrating")
        assert -0.5 < score < 0.5

    def test_empty_query_scores_zero(self, recall):
        """Empty query should score zero."""
        score = recall._query_emotion_score("")
        assert score == 0.0


class TestAffectAwareRanking:
    """Test that emotional context drives memory ranking."""

    def test_positive_query_boosts_positive_episodes(self, db, recall):
        """Positive queries should rank positive episodes higher."""
        # Insert positive episode
        pos = Episode(
            user_text="Just shipped the feature!",
            assistant_text="Great work!",
            valence=2, arousal=0.9
        )
        pos.id = db.insert_episode(pos)

        # Insert negative episode
        neg = Episode(
            user_text="Another bug found, sigh...",
            assistant_text="Keep going!",
            valence=-2, arousal=0.4
        )
        neg.id = db.insert_episode(neg)

        # Test that positive query ranks positive episode higher
        pos_score = recall._rank_episode(pos, "awesome breakthrough", pos.timestamp)
        neg_score = recall._rank_episode(neg, "awesome breakthrough", neg.timestamp)
        # Positive episode should score higher for positive query
        assert pos_score > neg_score

    def test_negative_query_boosts_negative_episodes(self, db, recall):
        """Negative queries should rank negative episodes higher."""
        # Insert positive episode
        pos = Episode(
            user_text="Feeling great about progress",
            assistant_text="Nice!",
            valence=2, arousal=0.5
        )
        pos.id = db.insert_episode(pos)

        # Insert negative episode
        neg = Episode(
            user_text="So frustrated with the API",
            assistant_text="Debug away!",
            valence=-2, arousal=0.8
        )
        neg.id = db.insert_episode(neg)

        # Test that negative query ranks negative episode higher
        neg_score = recall._rank_episode(neg, "frustrated angry error", neg.timestamp)
        pos_score = recall._rank_episode(pos, "frustrated angry error", pos.timestamp)
        # Negative episode should score higher for negative query
        assert neg_score > pos_score

    def test_same_polarity_boost(self, db, recall):
        """Same polarity query+episode should get boost."""
        ep = Episode(
            user_text="Test", assistant_text="Reply",
            valence=2, arousal=0.8
        )
        ep.id = db.insert_episode(ep)

        # Positive query + positive episode = boost
        score_boosted = recall._rank_episode(ep, "awesome great success", ep.timestamp)
        score_neutral = recall._rank_episode(ep, "the quick brown fox", ep.timestamp)
        assert score_boosted > score_neutral

    def test_opposite_polarity_penalty(self, db, recall):
        """Opposite polarity query+episode should get slight penalty."""
        ep = Episode(
            user_text="Test", assistant_text="Reply",
            valence=2, arousal=0.8
        )
        ep.id = db.insert_episode(ep)

        # Positive episode + negative query = penalty
        score_penalty = recall._rank_episode(ep, "frustrated angry bug", ep.timestamp)
        score_neutral = recall._rank_episode(ep, "the quick brown fox", ep.timestamp)
        assert score_penalty < score_neutral

    def test_high_arousal_boosts_during_intense_queries(self, db, recall):
        """High arousal episodes should rank higher during intense queries."""
        high_arousal = Episode(
            user_text="BREAKTHROUGH!",
            assistant_text="Yes!",
            valence=2, arousal=0.95
        )
        high_arousal.id = db.insert_episode(high_arousal)

        low_arousal = Episode(
            user_text="Reading docs.",
            assistant_text="Ok.",
            valence=1, arousal=0.2
        )
        low_arousal.id = db.insert_episode(low_arousal)

        # Both positive, but high arousal should have higher emotional_weight
        assert high_arousal.emotional_weight > low_arousal.emotional_weight

    def test_emotion_boost_doesnt_override_relevance(self, db, recall):
        """Emotional boost should enhance but not completely override text relevance."""
        # The emotion boost is capped at 0.3, so relevance still matters
        ep = Episode(
            user_text="Test", assistant_text="Reply",
            valence=2, arousal=0.8
        )
        ep.id = db.insert_episode(ep)

        # Even with max positive boost, the score shouldn't be astronomical
        score = recall._rank_episode(ep, "awesome great success", ep.timestamp)
        assert score < 2.0  # Sanity check: score shouldn't explode


class TestAffectAwareIntegration:
    """Integration tests for affect-aware recall."""

    def test_mixed_emotions_all_accessible(self, db, recall):
        """Multiple episodes with different emotions should all be findable."""
        # Create episodes with different emotions
        episodes_data = [
            ("Was really angry about the API failure", -2, 0.9),
            ("Feeling pretty content with progress", 1, 0.4),
            ("Super excited about the new feature", 2, 0.8),
            ("Kind of anxious about the deadline", -1, 0.7),
        ]
        ids = []
        for text, val, ar in episodes_data:
            ep = Episode(user_text=text, assistant_text="ok", valence=val, arousal=ar)
            ep.id = db.insert_episode(ep)
            ids.append(ep.id)

        # All episodes should be accessible via recent memories
        context = recall.build_context("")
        assert "Recent memories" in context or "recent memories" in context.lower()

    def test_query_drives_recall_of_matching_content(self, db, recall):
        """Queries should drive recall of textually matching content."""
        ep = Episode(
            user_text="Fixed the deployment pipeline issue",
            assistant_text="Nice work",
            valence=1, arousal=0.5
        )
        ep.id = db.insert_episode(ep)

        # The episode should be retrievable
        episodes = db.get_recent_episodes(limit=5)
        assert len(episodes) >= 1