from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import app.main
from app.schemas import ReminderItem, TaskAnalysis


@pytest.fixture
def env_setup(monkeypatch):
    monkeypatch.setenv("NOTION_TOKEN", "fake_notion_token")
    monkeypatch.setenv("NOTION_DATABASE_ID", "fake_db_id")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345678")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test_secret_123")


def test_health_check(env_setup):
    from app.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": None}


def test_webhook_secret_token_unauthorized(env_setup):
    from app.main import app
    client = TestClient(app)
    payload = {"update_id": 1, "message": {"message_id": 10, "text": "Hello", "chat": {"id": 123}}}
    # Invalid header token
    response = client.post(
        "/webhook",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong_secret"}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid secret token"}


def test_webhook_non_text_message_ignored(env_setup):
    from app.main import app
    client = TestClient(app)
    # Update with a sticker (no message.text)
    payload = {
        "update_id": 2,
        "message": {
            "message_id": 11,
            "chat": {"id": 123},
            "sticker": {"file_id": "abc"}
        }
    }
    response = client.post(
        "/webhook",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}


@patch("app.main.TelegramAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_create_task(mock_genai_cls, mock_notion_cls, mock_tg_cls, env_setup):
    # Mock Gemini response
    mock_genai_inst = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = TaskAnalysis(
        intent="CREATE_TASK",
        title="Study Algorithms",
        priority="High",
        tag="Schoolwork",
        due_date="2026-08-20",
        description="Chapter 4 exercises"
    )
    mock_genai_inst.models.generate_content.return_value = mock_response
    mock_genai_cls.return_value = mock_genai_inst

    mock_notion_inst = MagicMock()
    mock_notion_cls.return_value = mock_notion_inst

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 3,
        "message": {
            "message_id": 12,
            "text": "Study Algorithms Chapter 4",
            "chat": {"id": 9999}
        }
    }
    response = client.post(
        "/webhook",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Verify Notion create_task called
    mock_notion_inst.create_task.assert_called_once_with(
        title="Study Algorithms",
        priority="High",
        tag="Schoolwork",
        due_date="2026-08-20",
        description="Chapter 4 exercises"
    )
    # Verify Telegram send_message called
    mock_tg_inst.send_message.assert_called_once()
    call_args = mock_tg_inst.send_message.call_args[1]
    assert "Study Algorithms" in call_args["text"]
    assert call_args["chat_id"] == "9999"


@patch("app.main.TelegramAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_query_pending(mock_genai_cls, mock_notion_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = TaskAnalysis(
        intent="QUERY_PENDING",
        title="Query tasks"
    )
    mock_genai_inst.models.generate_content.return_value = mock_response
    mock_genai_cls.return_value = mock_genai_inst

    mock_notion_inst = MagicMock()
    mock_notion_inst.get_pending.return_value = [
        ReminderItem(page_id="p1", title="Task 1", due_date="2026-08-15")
    ]
    mock_notion_cls.return_value = mock_notion_inst

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 4,
        "message": {
            "message_id": 13,
            "text": "What are my pending tasks?",
            "chat": {"id": 8888}
        }
    }
    response = client.post(
        "/webhook",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_notion_inst.get_pending.assert_called_once_with(limit=5)
    mock_tg_inst.send_message.assert_called_once()
    assert "Task 1" in mock_tg_inst.send_message.call_args[1]["text"]


@patch("app.main.TelegramAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_gemini_fallback_on_error(mock_genai_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    mock_genai_inst.models.generate_content.side_effect = Exception("API rate limit exceeded")
    mock_genai_cls.return_value = mock_genai_inst

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    raw_text = "Just finished my workout today"
    payload = {
        "update_id": 5,
        "message": {
            "message_id": 14,
            "text": raw_text,
            "chat": {"id": 7777}
        }
    }
    response = client.post(
        "/webhook",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Telegram send_message should receive fallback daily log with raw text
    mock_tg_inst.send_message.assert_called_once()
    assert raw_text in mock_tg_inst.send_message.call_args[1]["text"]


@patch("app.main.TelegramAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_telegram_send_failure_logged_not_crashed(mock_genai_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = TaskAnalysis(intent="DAILY_LOG", log_content="Daily summary")
    mock_genai_inst.models.generate_content.return_value = mock_response
    mock_genai_cls.return_value = mock_genai_inst

    mock_tg_inst = MagicMock()
    mock_tg_inst.send_message.side_effect = Exception("Telegram API timeout")
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 6,
        "message": {
            "message_id": 15,
            "text": "Daily summary note",
            "chat": {"id": 6666}
        }
    }
    response = client.post(
        "/webhook",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    # Must not crash request, returns ok
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.TelegramAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_update_task(mock_genai_cls, mock_notion_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = TaskAnalysis(
        intent="UPDATE_TASK",
        title="pack for college",
        target_status="In progress"
    )
    mock_genai_inst.models.generate_content.return_value = mock_response
    mock_genai_cls.return_value = mock_genai_inst

    mock_notion_inst = MagicMock()
    mock_notion_inst.update_task_status.return_value = (True, "Pack for College", {"id": "page-123"})
    mock_notion_cls.return_value = mock_notion_inst

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 7,
        "message": {
            "message_id": 16,
            "text": "mark pack for college in progress",
            "chat": {"id": 5555}
        }
    }
    response = client.post(
        "/webhook",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_notion_inst.update_task_status.assert_called_once_with(
        title_query="pack for college",
        status_name="In progress",
        new_due_date=None
    )
    mock_tg_inst.send_message.assert_called_once()
    assert "Pack for College" in mock_tg_inst.send_message.call_args[1]["text"]

