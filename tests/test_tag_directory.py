"""Unit tests for Centralized Tag Directory and Re-tagging Suggestions."""

from app.tag_directory import (
    CANONICAL_TAG_NAMES,
    TAG_DIRECTORY,
    find_tag_reclassification_suggestions,
    match_closest_tag,
)


def test_canonical_tags_present():
    assert "AI Research" in CANONICAL_TAG_NAMES
    assert "System Design" in CANONICAL_TAG_NAMES
    assert "Distributed Systems" in CANONICAL_TAG_NAMES
    assert "Leetcode" in CANONICAL_TAG_NAMES
    assert "Miscellaneous" in CANONICAL_TAG_NAMES


def test_match_closest_tag_exact_and_fuzzy():
    assert match_closest_tag("ai research") == "AI Research"
    assert match_closest_tag("system design") == "System Design"
    assert match_closest_tag("distributed system") == "Distributed Systems"
    assert match_closest_tag("unknown tag xyz") == "Miscellaneous"


def test_match_closest_tag_from_text_keywords():
    moe_text = "Scaling Mixture of Experts with Top-2 Gating and ArXiv Transformer benchmarks"
    assert match_closest_tag(None, moe_text) == "AI Research"

    dist_text = "Raft consensus election safety and Paxos fault tolerance"
    assert match_closest_tag(None, dist_text) == "Distributed Systems"

    sys_text = "Database sharding, cache replication, and load balancer latency"
    assert match_closest_tag(None, sys_text) == "System Design"


def test_find_tag_reclassification_suggestions():
    items = [
        {
            "id": "item-1",
            "title": "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer",
            "url": "https://notion.so/item-1",
            "current_tag": "Miscellaneous",
            "text": "Deep learning paper on transformer scaling and attention weights",
        },
        {
            "id": "item-2",
            "title": "Random thoughts on my day",
            "url": "https://notion.so/item-2",
            "current_tag": "Miscellaneous",
            "text": "Just ate lunch and went for a walk",
        },
        {
            "id": "item-3",
            "title": "Raft Consensus Protocol",
            "url": "https://notion.so/item-3",
            "current_tag": "Distributed Systems",
            "text": "Leader election and log replication",
        },
    ]

    suggestions = find_tag_reclassification_suggestions(items)
    assert len(suggestions) == 1
    assert suggestions[0]["id"] == "item-1"
    assert suggestions[0]["suggested_tag"] == "AI Research"
