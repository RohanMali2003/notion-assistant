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
