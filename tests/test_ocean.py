"""Tests for OCEAN personality trait modeling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kortex.ocean import OCEANScore, score_turn


def test_score_turn_basic():
    """Score a basic conversation turn."""
    user_text = "I love exploring new ideas and patterns!"
    scored = score_turn(user_text, "Great! Let's dig into those patterns.")
    assert isinstance(scored, OCEANScore)
    assert scored.turn_count == 1


def test_score_turn_ocean_values_in_range():
    """All OCEAN scores should be between 0 and 1."""
    user_text = "Curious and creative!"
    scored = score_turn(user_text, "Interesting.")
    assert 0 <= scored.openness <= 1
    assert 0 <= scored.conscientiousness <= 1
    assert 0 <= scored.extraversion <= 1
    assert 0 <= scored.agreeableness <= 1
    assert 0 <= scored.neuroticism <= 1


def test_score_turn_high_openness():
    """Curiosity/creativity boosts openness."""
    user_text = "This is so fascinating! I wonder if we could explore the pattern of how algorithms discover new concepts. Imagine a framework that connects abstract models with creative paradigms!"
    scored = score_turn(user_text, "Definitely! Let's investigate.")
    assert scored.openness >= 0.5, f"Openness should be high for creative text: {scored.openness}"


def test_score_turn_high_conscientiousness():
    """Planning/structure boosts conscientiousness."""
    user_text = "We need a structured plan with specific goals, detailed steps, and a clear validation process to make sure we're checking all the deliverables."
    scored = score_turn(user_text, "Good plan.")
    assert scored.conscientiousness >= 0.4, f"Conscientiousness should be elevated: {scored.conscientiousness}"


def test_score_turn_high_extraversion():
    """Energy/enthusiasm boosts extraversion."""
    user_text = "This is amazing! So exciting! Let's celebrate this win together and share the amazing results!"
    scored = score_turn(user_text, "Awesome!")
    assert scored.extraversion >= 0.4, f"Extraversion should be elevated: {scored.extraversion}"


def test_score_turn_high_agreeableness():
    """Empathy/cooperation boosts agreeableness."""
    user_text = "I really appreciate how empathetic you are. Thank you for being so patient and gentle. It helps us collaborate as a team and support each other."
    scored = score_turn(user_text, "You too!")
    assert scored.agreeableness >= 0.4, f"Agreeableness should be elevated: {scored.agreeableness}"


def test_score_turn_high_neuroticism():
    """Anxiety/stress boosts neuroticism."""
    user_text = "I'm so frustrated and overwhelmed right now. Everything is chaotic and messy. I keep worrying about the tension and stress!"
    scored = score_turn(user_text, "Take a breath.")
    assert scored.neuroticism >= 0.4, f"Neuroticism should be elevated: {scored.neuroticism}"


def test_score_turn_ema_smoothing():
    """EMA smoothing blends new scores with existing ones."""
    base = score_turn("Hello there, just checking in.", "")
    # Score a highly creative turn
    creative = score_turn("Explore the fascinating pattern of creative algorithms and abstract concepts!", "", base)
    # Openness should increase after EMA smoothing
    assert creative.openness >= base.openness
    assert creative.turn_count == base.turn_count + 1


def test_score_turn_confidence_grows():
    """Confidence should grow logarithmically with turn count."""
    score = OCEANScore()
    for i in range(20):
        score = score_turn(f"Turn number {i} happening now.", "", score)
    assert score.turn_count == 20
    assert score.confidence > 0.3


def test_ocean_score_to_dict():
    """OCEANScore.to_dict returns proper structure."""
    score = OCEANScore(openness=0.7, extraversion=0.5, turn_count=10)
    d = score.to_dict()
    assert d["openness"] == 0.7
    assert d["turn_count"] == 10


def test_ocean_score_to_compact_text():
    """OCEANScore.to_compact_text returns readable bar chart."""
    score = OCEANScore(openness=0.7, extraversion=0.5, turn_count=5, confidence=0.6)
    text = score.to_compact_text()
    assert "Openness" in text
    assert "Extraversion" in text
