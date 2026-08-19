"""Unit tests for Notion duplicate detection and cleanup reporting."""

from unittest.mock import MagicMock, patch
import pytest
from app.duplicate_detector import (
    DuplicateCluster,
    DuplicateItem,
    calculate_similarity,
    find_duplicate_clusters,
    normalize_title,
)
from app.cleanup_reporter import (
    CLEANUP_PAGE_TITLE,
    NotionCleanupReporter,
    _extract_page_title,
    _extract_page_url,
    _extract_status,
)


def test_normalize_title():
    assert normalize_title("Deep Dive into Mixture of Experts (MoE) Architectures!", remove_stopwords=False) == "deep dive into mixture of experts moe architectures"
    assert "moe" in normalize_title("Deep Dive into Mixture of Experts (MoE) Architectures!", remove_stopwords=True)
    assert normalize_title("", remove_stopwords=False) == ""


def test_calculate_similarity_exact_and_fuzzy():
    # Exact match
    assert calculate_similarity("Mixture of Experts", "Mixture of Experts") == 1.0

    # High fuzzy similarity (User's real world example)
    t1 = "Deep Dive into Mixture of Experts (MoE) Architectures: From Foundations to State-of-the-Art Scaling"
    t2 = "Mixture of Experts (MoE) Architecture: Theory, Mechanics, and Scaling"
    sim = calculate_similarity(t1, t2)
    assert sim >= 0.70

    # High fuzzy similarity for Gemini papers
    g1 = "Architecture, Training, and Scaling of Google Gemini and Gemma Models"
    g2 = "Google Gemini AI and Gemma Models Fundamentals"
    g_sim = calculate_similarity(g1, g2)
    assert g_sim >= 0.70

    # Completely different topics
    diff_sim = calculate_similarity("Distributed Systems Raft Consensus", "Cooking Pasta Carbonara")
    assert diff_sim < 0.30


def test_find_duplicate_clusters_subjects():
    items = [
        DuplicateItem(
            id="s1",
            title="Deep Dive into Mixture of Experts (MoE) Architectures: From Foundations to State-of-the-Art Scaling",
            url="https://notion.so/s1",
            created_time="2026-08-19T23:02:00.000Z",
        ),
        DuplicateItem(
            id="s2",
            title="Mixture of Experts (MoE) Architecture: Theory, Mechanics, and Scaling",
            url="https://notion.so/s2",
            created_time="2026-08-19T22:57:00.000Z",
        ),
        DuplicateItem(
            id="s3",
            title="Graph Neural Networks Fundamentals",
            url="https://notion.so/s3",
            created_time="2026-08-19T21:48:00.000Z",
        ),
    ]

    clusters = find_duplicate_clusters(items, category="Subject", threshold=0.68)
    assert len(clusters) == 1
    assert clusters[0].category == "Subject"
    assert len(clusters[0].items) == 2
    assert "Review resources" in clusters[0].recommended_action


def test_find_duplicate_clusters_resources_identical_url():
    items = [
        DuplicateItem(
            id="r1",
            title="Gemma Technical Report",
            url="https://arxiv.org/abs/2403.08295",
        ),
        DuplicateItem(
            id="r2",
            title="Gemma: Open Models Paper",
            url="https://arxiv.org/abs/2403.08295",
        ),
        DuplicateItem(
            id="r3",
            title="Attention is All You Need",
            url="https://arxiv.org/abs/1706.03762",
        ),
    ]

    clusters = find_duplicate_clusters(items, category="Resource", threshold=0.80)
    assert len(clusters) == 1
    assert "Identical URL" in clusters[0].match_reason
    assert len(clusters[0].items) == 2


def test_build_report_blocks_empty_and_populated():
    reporter = NotionCleanupReporter(notion_client=MagicMock())

    # Empty results
    empty_results = {"subjects": [], "tasks": [], "resources": []}
    blocks_empty = reporter.build_report_blocks(empty_results)
    assert len(blocks_empty) == 1
    assert "Zero duplicates found" in blocks_empty[0]["callout"]["rich_text"][0]["text"]["content"]

    # Populated results
    cluster = DuplicateCluster(
        category="Subject",
        similarity_score=0.85,
        match_reason="85% title similarity",
        items=[
            DuplicateItem(id="s1", title="MoE 1", url="https://notion.so/s1", created_time="2026-08-19"),
            DuplicateItem(id="s2", title="MoE 2", url="https://notion.so/s2", created_time="2026-08-19"),
        ],
        recommended_action="Merge and delete redundant subject.",
    )
    pop_results = {"subjects": [cluster], "tasks": [], "resources": []}
    blocks_pop = reporter.build_report_blocks(pop_results)

    assert any(b["type"] == "heading_2" and "Duplicate Subjects" in b["heading_2"]["rich_text"][0]["text"]["content"] for b in blocks_pop)
    bullet_items = [b for b in blocks_pop if b["type"] == "bulleted_list_item"]
    assert len(bullet_items) == 2
    assert bullet_items[0]["bulleted_list_item"]["rich_text"][0]["text"]["link"]["url"] == "https://notion.so/s1"


def test_reporter_update_cleanup_report_page():
    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client
    mock_notion.subjects_db_id = "subjects_db_123"
    mock_notion.tasks_db_id = "tasks_db_123"
    mock_notion.resources_db_id = "resources_db_123"

    # Mock DB queries
    mock_notion._query_database.side_effect = [
        # Subjects
        {
            "results": [
                {
                    "id": "s1",
                    "properties": {"Subject": {"type": "title", "title": [{"plain_text": "Mixture of Experts"}]}},
                    "url": "https://notion.so/s1",
                    "created_time": "2026-08-19T22:00:00.000Z",
                },
                {
                    "id": "s2",
                    "properties": {"Subject": {"type": "title", "title": [{"plain_text": "Mixture of Experts (MoE) Architecture"}]}},
                    "url": "https://notion.so/s2",
                    "created_time": "2026-08-19T22:05:00.000Z",
                },
            ]
        },
        # Tasks
        {"results": []},
        # Resources
        {"results": []},
    ]

    # Mock search finding existing page
    mock_notion._request_with_retry.side_effect = [
        # search
        {"results": [{"id": "page-999", "properties": {"title": {"type": "title", "title": [{"plain_text": CLEANUP_PAGE_TITLE}]}}, "url": "https://notion.so/page-999"}]},
        # list children
        {"results": [{"id": "old-block-1"}]},
        # delete child
        {},
        # append children
        {},
    ]

    reporter = NotionCleanupReporter(notion_client=mock_notion)
    res = reporter.update_cleanup_report_page()

    assert res["status"] == "ok"
    assert res["page_id"] == "page-999"
    assert res["duplicate_subjects"] == 1
    assert res["total_duplicate_clusters"] == 1
