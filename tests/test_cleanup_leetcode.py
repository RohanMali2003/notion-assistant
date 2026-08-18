from unittest.mock import MagicMock, patch
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))

from cron.cleanup_leetcode import cleanup_expired_leetcode_tasks


def test_cleanup_missing_env_vars(monkeypatch):
    """Test that missing required environment variables raises ValueError."""
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_TASKS_DB_ID", raising=False)
    monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)

    with pytest.raises(ValueError) as exc_info:
        cleanup_expired_leetcode_tasks()

    assert "Missing required environment variable(s)" in str(exc_info.value)


def test_cleanup_no_expired_tasks(monkeypatch):
    """Test when no expired LeetCode tasks are returned by Notion query."""
    monkeypatch.setenv("NOTION_API_KEY", "fake_key")
    monkeypatch.setenv("NOTION_TASKS_DB_ID", "fake_tasks_db")

    with patch("cron.cleanup_leetcode.Client") as mock_client_cls:
        mock_notion = MagicMock()
        mock_notion.databases.query.return_value = {"results": [], "has_more": False}
        mock_client_cls.return_value = mock_notion

        closed_count = cleanup_expired_leetcode_tasks()

        assert closed_count == 0
        mock_notion.databases.query.assert_called_once()
        mock_notion.pages.update.assert_not_called()


def test_cleanup_successful_close_with_pagination(monkeypatch):
    """Test finding and closing expired tasks with pagination handling."""
    monkeypatch.setenv("NOTION_API_KEY", "fake_key")
    monkeypatch.setenv("NOTION_TASKS_DB_ID", "fake_tasks_db")

    page_1 = {
        "results": [{"id": "task-1"}, {"id": "task-2"}],
        "has_more": True,
        "next_cursor": "cursor_123",
    }
    page_2 = {
        "results": [{"id": "task-3"}],
        "has_more": False,
        "next_cursor": None,
    }

    with patch("cron.cleanup_leetcode.Client") as mock_client_cls:
        mock_notion = MagicMock()
        mock_notion.databases.query.side_effect = [page_1, page_2]
        mock_notion.pages.update.return_value = {"id": "updated"}
        mock_client_cls.return_value = mock_notion

        closed_count = cleanup_expired_leetcode_tasks()

        assert closed_count == 3
        assert mock_notion.databases.query.call_count == 2
        assert mock_notion.pages.update.call_count == 3

        # Check call arguments for updates
        mock_notion.pages.update.assert_any_call(
            page_id="task-1",
            properties={"Status": {"status": {"name": "Done"}}},
        )
        mock_notion.pages.update.assert_any_call(
            page_id="task-3",
            properties={"Status": {"status": {"name": "Done"}}},
        )


def test_cleanup_fallback_on_query_error(monkeypatch):
    """Test that query failure falls back and attempts secondary query."""
    monkeypatch.setenv("NOTION_API_KEY", "fake_key")
    monkeypatch.setenv("NOTION_TASKS_DB_ID", "fake_tasks_db")

    fallback_response = {
        "results": [{"id": "task-fallback-1"}],
        "has_more": False,
    }

    with patch("cron.cleanup_leetcode.Client") as mock_client_cls:
        mock_notion = MagicMock()
        mock_notion.databases.query.side_effect = [
            Exception("Invalid filter property"),
            fallback_response,
        ]
        mock_client_cls.return_value = mock_notion

        closed_count = cleanup_expired_leetcode_tasks()

        assert closed_count == 1
        assert mock_notion.databases.query.call_count == 2
        mock_notion.pages.update.assert_called_once_with(
            page_id="task-fallback-1",
            properties={"Status": {"status": {"name": "Done"}}},
        )


def test_cleanup_page_update_fallback(monkeypatch):
    """Test that page update falls back to select if status schema raises error."""
    monkeypatch.setenv("NOTION_API_KEY", "fake_key")
    monkeypatch.setenv("NOTION_TASKS_DB_ID", "fake_tasks_db")

    with patch("cron.cleanup_leetcode.Client") as mock_client_cls:
        mock_notion = MagicMock()
        mock_notion.databases.query.return_value = {
            "results": [{"id": "task-select-only"}],
            "has_more": False,
        }
        # First call (status) fails, second call (select) succeeds
        mock_notion.pages.update.side_effect = [
            Exception("Status is not a status property"),
            {"id": "task-select-only"},
        ]
        mock_client_cls.return_value = mock_notion

        closed_count = cleanup_expired_leetcode_tasks()

        assert closed_count == 1
        assert mock_notion.pages.update.call_count == 2
        mock_notion.pages.update.assert_any_call(
            page_id="task-select-only",
            properties={"Status": {"select": {"name": "Done"}}},
        )


def test_cleanup_query_fallback_to_request(monkeypatch):
    """Test that query falls back to client.request when databases endpoint has no query method."""
    monkeypatch.setenv("NOTION_API_KEY", "fake_key")
    monkeypatch.setenv("NOTION_TASKS_DB_ID", "fake_tasks_db")

    class MockDatabasesWithoutQuery:
        pass

    with patch("cron.cleanup_leetcode.Client") as mock_client_cls:
        mock_notion = MagicMock()
        mock_notion.databases = MockDatabasesWithoutQuery()
        mock_notion.request.return_value = {
            "results": [{"id": "task-req-1"}],
            "has_more": False,
        }
        mock_notion.pages.update.return_value = {"id": "task-req-1"}
        mock_client_cls.return_value = mock_notion

        closed_count = cleanup_expired_leetcode_tasks()

        assert closed_count == 1
        assert mock_notion.request.call_count == 1
        mock_notion.pages.update.assert_called_once_with(
            page_id="task-req-1",
            properties={"Status": {"status": {"name": "Done"}}},
        )

