from unittest.mock import MagicMock, patch
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))

from cron.check_reminders import check_reminders


def test_missing_env_vars(monkeypatch):
    """Test that missing required environment variables raises ValueError."""
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_TASKS_DB_ID", raising=False)
    monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(ValueError) as exc_info:
        check_reminders()

    assert "Missing required environment variable(s)" in str(exc_info.value)
    assert "NOTION_API_KEY" in str(exc_info.value)


def test_empty_candidates(monkeypatch):
    """Test that if both candidate lists are empty, script exits with 0 and no requests are made."""
    monkeypatch.setenv("NOTION_API_KEY", "fake_key")
    monkeypatch.setenv("NOTION_TASKS_DB_ID", "fake_db")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat_id")

    with patch("cron.check_reminders.NotionAssistantClient") as mock_notion_cls, \
         patch("cron.check_reminders.requests.post") as mock_post:
        
        mock_notion_inst = MagicMock()
        mock_notion_inst.get_reminder_candidates.return_value = ([], [])
        mock_notion_cls.return_value = mock_notion_inst

        with pytest.raises(SystemExit) as exc_info:
            check_reminders()

        assert exc_info.value.code == 0
        mock_post.assert_not_called()


def test_successful_reminder_send(monkeypatch):
    """Test formatting and sending when candidate lists contain items."""
    monkeypatch.setenv("NOTION_API_KEY", "fake_key")
    monkeypatch.setenv("NOTION_TASKS_DB_ID", "fake_db")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat_id")

    due_soon_items = [
        {"title": "Fix critical bug", "due_date": "2026-08-14"}
    ]
    high_priority_items = [
        {"title": "Refactor codebase", "due_date": None}
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True}

    with patch("cron.check_reminders.NotionAssistantClient") as mock_notion_cls, \
         patch("cron.check_reminders.requests.post", return_value=mock_response) as mock_post:
        
        mock_notion_inst = MagicMock()
        mock_notion_inst.get_reminder_candidates.return_value = (due_soon_items, high_priority_items)
        mock_notion_cls.return_value = mock_notion_inst

        check_reminders()

        mock_post.assert_called_once()
        url_arg = mock_post.call_args[0][0]
        assert "fake_bot_token" in url_arg

        payload = mock_post.call_args[1]["json"]
        assert payload["chat_id"] == "fake_chat_id"
        assert "Due soon" in payload["text"]
        assert "Fix critical bug (Due: 2026-08-14)" in payload["text"]
        assert "High priority, no due date" in payload["text"]
        assert "Refactor codebase" in payload["text"]


def test_telegram_send_failure(monkeypatch):
    """Test that a failed Telegram send exits with a non-zero exit code."""
    monkeypatch.setenv("NOTION_API_KEY", "fake_key")
    monkeypatch.setenv("NOTION_TASKS_DB_ID", "fake_db")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat_id")

    due_soon_items = [{"title": "Important task", "due_date": "2026-08-14"}]

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("cron.check_reminders.NotionAssistantClient") as mock_notion_cls, \
         patch("cron.check_reminders.requests.post", return_value=mock_response):
        
        mock_notion_inst = MagicMock()
        mock_notion_inst.get_reminder_candidates.return_value = (due_soon_items, [])
        mock_notion_cls.return_value = mock_notion_inst

        with pytest.raises(SystemExit) as exc_info:
            check_reminders()

        assert exc_info.value.code != 0
