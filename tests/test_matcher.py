"""Unit tests for app/matcher.py 3-tier entity resolution engine."""

import pytest
from datetime import datetime
from app.matcher import entity_resolver, resolve_natural_date


def test_rapidfuzz_tier1_match():
    candidates = [
        {"title": "Pay Berkshire Dining Hall bill"},
        {"title": "Submit CICS scholarship app"},
        {"title": "Review LeetCode binary trees"},
    ]
    matched, tier, score = entity_resolver.resolve_entity(
        query="Berkshire Dining",
        candidates=candidates,
    )
    assert matched is not None
    assert matched["title"] == "Pay Berkshire Dining Hall bill"
    assert tier == "exact_or_rapidfuzz"


def test_minilm_semantic_tier2_match():
    if entity_resolver._get_embedding_model() is None:
        pytest.skip("SentenceTransformer model is not available in current environment")

    candidates = [
        {"title": "Pay Berkshire Dining Hall bill"},
        {"title": "Submit CICS scholarship app"},
        {"title": "Year 1 Budget Plan"},
    ]
    # "cafeteria bill" has zero word overlap with "Pay Berkshire Dining Hall bill"
    matched, tier, score = entity_resolver.resolve_entity(
        query="pay cafeteria bill",
        candidates=candidates,
    )
    assert matched is not None
    assert matched["title"] == "Pay Berkshire Dining Hall bill"
    assert tier in ("exact_or_rapidfuzz", "minilm_embedding")



def test_natural_date_resolution():
    ref_dt = datetime(2026, 8, 21)  # Friday

    d1 = resolve_natural_date("tomorrow", reference_dt=ref_dt)
    assert d1 == "2026-08-22"

    d2 = resolve_natural_date("2026-08-30", reference_dt=ref_dt)
    assert d2 == "2026-08-30"

    d3 = resolve_natural_date("in 3 days", reference_dt=ref_dt)
    assert d3 == "2026-08-24"


def test_filter_candidates_symbol_and_filler_normalization():
    candidates = [
        {"title": "Read & Annotate: The Pragmatic Programmer Chapter 1"},
        {"title": "Read & Annotate: The Pragmatic Programmer Chapter 2"},
        {"title": "Ask GOATmini to make Ocean a little smarter"},
        {"title": "Build AI voice agent"},
    ]

    # Query with "and" instead of "&" and filler word "tasks"
    matched = entity_resolver.filter_candidates(
        query="read and annotate tasks",
        candidates=candidates,
        use_semantic_fallback=False,
    )
    assert len(matched) == 2
    assert matched[0]["title"] == "Read & Annotate: The Pragmatic Programmer Chapter 1"
    assert matched[1]["title"] == "Read & Annotate: The Pragmatic Programmer Chapter 2"

    # Query with raw "&"
    matched_amp = entity_resolver.filter_candidates(
        query="Read & Annotate",
        candidates=candidates,
        use_semantic_fallback=False,
    )
    assert len(matched_amp) == 2


def test_filter_candidates_false_positive_resistance():
    candidates = [
        {"title": "Deep Learning paper reading"},
        {"title": "Earning statement review for Q3"},
        {"title": "Machine Learning pipeline deployment"},
        {"title": "Clean kitchen and buy groceries"},
    ]

    # "learning" must NOT match "Earning"
    matched = entity_resolver.filter_candidates(
        query="learning",
        candidates=candidates,
        use_semantic_fallback=False,
    )
    assert len(matched) == 2
    matched_titles = [m["title"] for m in matched]
    assert "Deep Learning paper reading" in matched_titles
    assert "Machine Learning pipeline deployment" in matched_titles
    assert "Earning statement review for Q3" not in matched_titles


def test_filter_candidates_tag_filtering():
    candidates = [
        {"title": "Berkshire Dining Onboarding", "tag": "UMass Admin"},
        {"title": "Review LeetCode Trees", "tag": "Leetcode"},
        {"title": "Pay Tuition Bill", "tag": "UMass Admin"},
    ]

    # Batch query matching tag "UMass"
    matched = entity_resolver.filter_candidates(
        query="umass tasks",
        candidates=candidates,
        use_semantic_fallback=False,
    )
    assert len(matched) == 2
    matched_titles = [m["title"] for m in matched]
    assert "Berkshire Dining Onboarding" in matched_titles
    assert "Pay Tuition Bill" in matched_titles
    assert "Review LeetCode Trees" not in matched_titles
