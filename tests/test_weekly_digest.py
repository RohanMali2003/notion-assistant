"""Unit tests for Sunday Weekly Velocity Digest Service."""

from unittest.mock import MagicMock, patch
import pytest
from app.schemas import WeeklyVelocityReport
from app.weekly_digest_service import (
    create_notion_weekly_review_page,
    execute_weekly_digest_pipeline,
    fetch_past_week_workspace_activity,
    synthesize_velocity_digest,
)


def test_fetch_past_week_workspace_activity():
    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client
    mock_notion.tasks_db_id = "tasks-db"
    mock_notion.subjects_db_id = "subj-db"
    mock_notion.resources_db_id = "res-db"

    # Tasks response
    tasks_page = {
        "id": "task-1",
        "properties": {
            "Task name": {"type": "title", "title": [{"plain_text": "Implement MoE model"}]},
            "Status": {"type": "status", "status": {"name": "Done"}},
            "Tags": {"type": "multi_select", "multi_select": [{"name": "AI Research"}]},
        },
    }
    mock_notion._query_database.side_effect = [
        {"results": [tasks_page]},  # Tasks DB query
        {"results": [{"id": "subj-1", "properties": {"Subject name": {"type": "title", "title": [{"plain_text": "Mixture of Experts"}]}}}]},  # Subj DB query
        {"results": [{"id": "res-1", "properties": {"Resource title": {"type": "title", "title": [{"plain_text": "Sparsely Gated MoE"}]}, "Type": {"type": "select", "select": {"name": "Paper"}}}}]}]  # Res DB query

    activity = fetch_past_week_workspace_activity(notion_client=mock_notion, days=7)
    assert activity["total_completed"] == 1
    assert len(activity["subjects"]) == 1
    assert len(activity["resources"]) == 1
    assert activity["tasks_completed"][0]["title"] == "Implement MoE model"


@patch("app.weekly_digest_service.get_gemini_client")
def test_synthesize_velocity_digest(mock_gemini_cls):
    mock_gemini = MagicMock()
    mock_gemini_cls.return_value = mock_gemini
    mock_resp = MagicMock()
    mock_resp.parsed = WeeklyVelocityReport(
        velocity_score=90,
        verdict="High Momentum",
        headline="Completed flagship MoE research and cleared critical systems tasks.",
        tasks_completed_count=5,
        tasks_pending_count=2,
        completed_highlights=["Shipped MoE curriculum", "Completed distributed key-value store"],
        learning_progress=["Mixture of Experts Architecture", "Gemini Foundational Papers"],
        bottlenecks=["UMass dining form submission"],
        next_week_priorities=["Advance Raft consensus implementation", "Solve Monotonic Stack problems"],
    )
    mock_gemini.models.generate_content.return_value = mock_resp

    activity_data = {
        "cutoff_days": 7,
        "tasks_completed": [{"title": "Shipped MoE curriculum"}],
        "tasks_in_progress": [],
        "tasks_overdue": [],
        "subjects": [{"title": "Mixture of Experts"}],
        "resources": [{"title": "MoE paper", "type": "Paper"}],
    }

    report = synthesize_velocity_digest(activity_data)
    assert report.velocity_score == 90
    assert report.verdict == "High Momentum"
    assert "MoE research" in report.headline
    assert "90/100" in report.full_digest_markdown
    assert "High Momentum" in report.full_digest_markdown


def test_create_notion_weekly_review_page():
    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client
    mock_notion.tasks_db_id = "tasks-db"
    mock_notion._query_database.return_value = {"results": []}
    mock_notion._request_with_retry.return_value = {
        "id": "review-page-123",
        "url": "https://notion.so/review-page-123",
    }

    report = WeeklyVelocityReport(
        velocity_score=88,
        verdict="Steady Execution",
        headline="Productive week across systems and AI.",
        completed_highlights=["Shipped v2.1"],
        learning_progress=["Distributed Systems"],
        bottlenecks=["LeetCode Daily"],
        next_week_priorities=["Continue momentum"],
    )

    page_id, page_url = create_notion_weekly_review_page(report, notion_client=mock_notion)
    assert page_id == "review-page-123"
    assert page_url == "https://notion.so/review-page-123"
    mock_notion._request_with_retry.assert_called_once()


def test_update_existing_notion_weekly_review_page():
    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client
    mock_notion.tasks_db_id = "tasks-db"
    mock_notion._query_database.return_value = {
        "results": [{"id": "existing-page-999", "url": "https://notion.so/existing-page-999"}]
    }
    mock_notion._request_with_retry.side_effect = [
        {"results": [{"id": "block-1"}]},  # children.list
        {},  # blocks.delete
        {},  # children.append
    ]

    report = WeeklyVelocityReport(
        velocity_score=95,
        verdict="High Momentum",
        headline="Updated review.",
        completed_highlights=["Shipped feature"],
        learning_progress=[],
        bottlenecks=[],
        next_week_priorities=[],
    )

    page_id, page_url = create_notion_weekly_review_page(report, notion_client=mock_notion)
    assert page_id == "existing-page-999"
    assert page_url == "https://notion.so/existing-page-999"


@patch("app.weekly_digest_service.fetch_past_week_workspace_activity")
@patch("app.weekly_digest_service.synthesize_velocity_digest")
@patch("app.weekly_digest_service.create_notion_weekly_review_page")
def test_execute_weekly_digest_pipeline(mock_create_page, mock_synthesize, mock_fetch):
    mock_fetch.return_value = {"cutoff_days": 7}
    mock_report = WeeklyVelocityReport(
        velocity_score=92,
        verdict="High Momentum",
        headline="Breakthrough velocity.",
        full_digest_markdown="📊 *Ocean Weekly Velocity Digest*\n⚡ *Score:* 92/100",
    )
    mock_synthesize.return_value = mock_report
    mock_create_page.return_value = ("page-id", "https://notion.so/weekly-review")

    mock_wa = MagicMock()
    mock_tg = MagicMock()

    result = execute_weekly_digest_pipeline(
        to_phone="15551234567",
        chat_id="777",
        whatsapp_client=mock_wa,
        telegram_client=mock_tg,
    )

    assert result["status"] == "ok"
    assert result["velocity_score"] == 92
    mock_wa.send_message.assert_called_once()
    assert "https://notion.so/weekly-review" in mock_wa.send_message.call_args[1]["text"]
    mock_tg.send_message.assert_called_once()
