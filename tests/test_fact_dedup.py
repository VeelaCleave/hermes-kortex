"""Tests for improved fact dedup with trigram Jaccard."""
import pytest
from kortex.ingest import Ingestor


class TestFactDedupEquivalence:
    """Test the three-pass _facts_are_equivalent method."""

    def test_identical_facts_match(self):
        assert Ingestor._facts_are_equivalent("dark mode", "dark mode") is True

    def test_exact_match_short(self):
        assert Ingestor._facts_are_equivalent("uses Python", "uses Python") is True

    def test_reworded_match_bigram(self):
        assert Ingestor._facts_are_equivalent(
            "my cat is named Luna", "cat is named Luna"
        ) is True

    def test_different_facts_no_match(self):
        assert Ingestor._facts_are_equivalent("dark mode", "light theme") is False

    def test_empty_vs_nonempty(self):
        assert Ingestor._facts_are_equivalent("", "test") is False

    def test_both_empty(self):
        assert Ingestor._facts_are_equivalent("", "") is False

    def test_length_filter_short_vs_long(self):
        assert Ingestor._facts_are_equivalent(
            "uses Python",
            "has been using Python for over five years consistently across multiple projects",
        ) is False

    def test_similar_but_not_equivalent(self):
        assert Ingestor._facts_are_equivalent(
            "has a dog named Max", "has a cat named Luna"
        ) is False

    def test_case_insensitive(self):
        assert Ingestor._facts_are_equivalent("Uses Python", "uses python") is True

    def test_multilingual_basic(self):
        assert Ingestor._facts_are_equivalent("parle français", "parle français") is True

    def test_number_sensitive(self):
        assert Ingestor._facts_are_equivalent("version 3", "version 5") is False

    def test_normalized_punctuation(self):
        # After normalization, punctuation is stripped
        result = Ingestor._facts_are_equivalent("uses python", "uses python.")
        # This should match because normalization strips the period
        assert result is True

    def test_normalize_text_removes_punctuation(self):
        normalized = Ingestor._normalize_text("Hello, World!")
        assert normalized == "hello world"

    def test_normalize_text_collapses_spaces(self):
        normalized = Ingestor._normalize_text("  multiple   spaces  ")
        assert normalized == "multiple spaces"
