from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import app.main
from app.main import (
    analyze_user_text_two_stage,
    analyze_user_text_with_gemini,
    classify_module_stage1,
    parse_document_append_stage2,
    parse_learning_stage2,
    parse_leetcode_stage2,
    parse_mind_stage2,
    parse_task_action_stage2,
    parse_tasks_stage2,
)
from app.schemas import (
    DocumentAppendAnalysis,
    LearningRequest,
    LeetcodeReviewRequest,
    MindEntry,
    ModuleClassification,
    ReminderItem,
    TaskActionAnalysis,
    TaskAnalysis,
)


@pytest.fixture
def env_setup(monkeypatch):
    monkeypatch.setenv("NOTION_TOKEN", "fake_notion_token")
    monkeypatch.setenv("NOTION_DATABASE_ID", "fake_db_id")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345678")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test_secret_123")


# --- Stage 1 Unit Tests ---

@patch("app.main.genai.Client")
def test_classify_module_stage1_success(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.parsed = ModuleClassification(module="MIND", raw_text="Just drafted a new substack post")
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    result = classify_module_stage1("Just drafted a new substack post")
    assert result.module == "MIND"
    assert result.raw_text == "Just drafted a new substack post"


@patch("app.main.genai.Client")
def test_classify_module_stage1_json_text_fallback(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.parsed = None
    mock_resp.text = '{"module": "LEARNING", "raw_text": "Study Raft consensus"}'
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    result = classify_module_stage1("Study Raft consensus")
    assert result.module == "LEARNING"
    assert result.raw_text == "Study Raft consensus"


@patch("app.main.genai.Client")
def test_classify_module_stage1_error_fallback(mock_genai_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Stage 1 API error")
    mock_genai_cls.return_value = mock_client

    result = classify_module_stage1("buy groceries tomorrow")
    assert result.module == "TASKS"
    assert result.raw_text == "buy groceries tomorrow"


# --- Stage 2 Parser Unit Tests ---

@patch("app.main.genai.Client")
def test_parse_tasks_stage2_success(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.parsed = TaskAnalysis(intent="CREATE_TASK", title="Finish report", priority="High")
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    res = parse_tasks_stage2("Finish report High priority")
    assert res.intent == "CREATE_TASK"
    assert res.title == "Finish report"
    assert res.priority == "High"


@patch("app.main.genai.Client")
def test_parse_tasks_stage2_error_fallback(mock_genai_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Tasks stage 2 error")
    mock_genai_cls.return_value = mock_client

    res = parse_tasks_stage2("Some random note")
    assert res.intent == "DAILY_LOG"
    assert res.log_content == "Some random note"


@patch("app.main.genai.Client")
def test_parse_mind_stage2_success(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.parsed = MindEntry(
        entry_type="DRAFT_SUBSTACK",
        title="Simplicity in Software",
        core_thesis="Simple systems reduce cognitive load and failure domains.",
        content="Draft body here explaining simplicity principles...",
        summary="A draft on simplicity",
        tags=["Substack", "Tech"],
    )
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    res = parse_mind_stage2("Drafting an essay on software simplicity")
    assert res.entry_type == "DRAFT_SUBSTACK"
    assert res.sub_intent == "DRAFT_SUBSTACK"
    assert res.title == "Simplicity in Software"
    assert res.core_thesis == "Simple systems reduce cognitive load and failure domains."
    assert res.content == "Draft body here explaining simplicity principles..."
    assert res.tags == ["Substack", "Tech"]


@patch("app.main.genai.Client")
def test_parse_mind_stage2_error_fallback(mock_genai_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Mind stage 2 error")
    mock_genai_cls.return_value = mock_client

    res = parse_mind_stage2("My evening thoughts on architecture. More details follow.")
    assert res.entry_type == "DAILY_LOG"
    assert res.sub_intent == "DAILY_LOG"
    assert res.content == "My evening thoughts on architecture. More details follow."
    assert "My evening thoughts" in res.core_thesis


@patch("app.main.genai.Client")
def test_parse_learning_stage2_success(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.parsed = LearningRequest(
        topic="Distributed Systems",
        category="Computer Science",
        goal="Master Paxos vs Raft",
        proficiency_level="Intermediate",
    )
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    res = parse_learning_stage2("I want to learn Distributed Systems, focusing on Paxos vs Raft")
    assert res.topic == "Distributed Systems"
    assert res.category == "Computer Science"
    assert res.goal == "Master Paxos vs Raft"
    assert res.proficiency_level == "Intermediate"


@patch("app.main.genai.Client")
def test_parse_learning_stage2_error_fallback(mock_genai_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Learning stage 2 error")
    mock_genai_cls.return_value = mock_client

    res = parse_learning_stage2("Learn Quantum Mechanics")
    assert "Quantum" in res.topic or res.topic == "Learn Quantum Mechanics"[:50]
    assert res.goal == "Learn Quantum Mechanics"


@patch("app.main.genai.Client")
def test_parse_leetcode_stage2_success(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.parsed = LeetcodeReviewRequest(
        problem_name="Trapping Rain Water",
        problem_number=42,
        difficulty="Hard",
        patterns=["Two Pointers"],
        status="Solved",
        review_notes="O(N) time O(1) space using two pointer technique",
    )
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    res = parse_leetcode_stage2("Review LC 42 Trapping Rain Water using Two Pointers")
    assert res.problem_name == "Trapping Rain Water"
    assert res.problem_number == 42
    assert res.difficulty == "Hard"
    assert res.patterns == ["Two Pointers"]
    assert res.status == "Solved"


@patch("app.main.genai.Client")
def test_parse_leetcode_stage2_error_fallback(mock_genai_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Leetcode stage 2 error")
    mock_genai_cls.return_value = mock_client

    res = parse_leetcode_stage2("Review LRU Cache problem")
    assert res.review_notes == "Review LRU Cache problem"


# --- Two-Stage Pipeline Tests ---

@patch("app.main.parse_tasks_stage2")
@patch("app.main.classify_module_stage1")
def test_analyze_user_text_two_stage_tasks(mock_stage1, mock_tasks_stage2):
    mock_stage1.return_value = ModuleClassification(module="TASKS", raw_text="buy milk")
    mock_tasks_stage2.return_value = TaskAnalysis(intent="CREATE_TASK", title="buy milk")

    module, parsed = analyze_user_text_two_stage("buy milk")
    assert module == "TASKS"
    assert parsed.intent == "CREATE_TASK"
    mock_tasks_stage2.assert_called_once_with("buy milk")


@patch("app.main.parse_mind_stage2")
@patch("app.main.classify_module_stage1")
def test_analyze_user_text_two_stage_mind(mock_stage1, mock_mind_stage2):
    mock_stage1.return_value = ModuleClassification(module="MIND", raw_text="thoughts today")
    mock_mind_stage2.return_value = MindEntry(entry_type="DAILY_LOG", content="thoughts today")

    module, parsed = analyze_user_text_two_stage("thoughts today")
    assert module == "MIND"
    assert parsed.entry_type == "DAILY_LOG"
    mock_mind_stage2.assert_called_once_with("thoughts today")


@patch("app.main.parse_tasks_stage2")
def test_analyze_user_text_with_gemini_legacy(mock_tasks_stage2):
    mock_tasks_stage2.return_value = TaskAnalysis(intent="CREATE_TASK", title="legacy test")
    res = analyze_user_text_with_gemini("legacy test")
    assert res.title == "legacy test"
    mock_tasks_stage2.assert_called_once_with("legacy test")


# --- Endpoint & Webhook Integration Tests ---

def test_health_check(env_setup):
    from app.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": None}


# --- WhatsApp Webhook Handshake & Event Tests ---

def test_whatsapp_webhook_handshake_success(env_setup, monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my_secret_verify_token")
    from app.main import app
    client = TestClient(app)
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "my_secret_verify_token",
            "hub.challenge": "1158201444",
        }
    )
    assert response.status_code == 200
    assert response.text == "1158201444"


def test_whatsapp_webhook_handshake_invalid_token(env_setup, monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my_secret_verify_token")
    from app.main import app
    client = TestClient(app)
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "1158201444",
        }
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "Verification token mismatch"}


def test_whatsapp_webhook_handshake_invalid_mode(env_setup, monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my_secret_verify_token")
    from app.main import app
    client = TestClient(app)
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "unsubscribe",
            "hub.verify_token": "my_secret_verify_token",
            "hub.challenge": "1158201444",
        }
    )
    assert response.status_code == 403


def test_whatsapp_webhook_handshake_missing_params(env_setup):
    from app.main import app
    client = TestClient(app)
    response = client.get("/webhook")
    assert response.status_code == 403


def test_whatsapp_webhook_post_event_received(env_setup):
    from app.main import app
    client = TestClient(app)
    response = client.post("/webhook", json={"object": "whatsapp_business_account", "entry": []})
    assert response.status_code == 200
    assert response.text == "EVENT_RECEIVED"


# --- Telegram Webhook Tests ---

def test_webhook_secret_token_unauthorized(env_setup):
    from app.main import app
    client = TestClient(app)
    payload = {"update_id": 1, "message": {"message_id": 10, "text": "Hello", "chat": {"id": 123}}}
    response = client.post(
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong_secret"}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid secret token"}


def test_webhook_non_text_message_ignored(env_setup):
    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 2,
        "message": {
            "message_id": 11,
            "chat": {"id": 123},
            "sticker": {"file_id": "abc"}
        }
    }
    response = client.post(
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}


@patch("app.main.TelegramAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_create_task(mock_genai_cls, mock_notion_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="TASKS", raw_text="Study Algorithms Chapter 4")

    stage2_resp = MagicMock()
    stage2_resp.parsed = TaskAnalysis(
        intent="CREATE_TASK",
        title="Study Algorithms",
        priority="High",
        tag="Schoolwork",
        due_date="2026-08-20",
        description="Chapter 4 exercises"
    )
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
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
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_notion_inst.create_task.assert_called_once_with(
        title="Study Algorithms",
        priority="High",
        tag="Schoolwork",
        due_date="2026-08-20",
        description="Chapter 4 exercises"
    )
    mock_tg_inst.send_message.assert_called_once()
    call_args = mock_tg_inst.send_message.call_args[1]
    assert "Study Algorithms" in call_args["text"]
    assert call_args["chat_id"] == "9999"


@patch("app.main.TelegramAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_update_task(mock_genai_cls, mock_notion_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="TASKS", raw_text="mark pack for college in progress")

    stage2_resp = MagicMock()
    stage2_resp.parsed = TaskAnalysis(
        intent="UPDATE_TASK",
        title="pack for college",
        target_status="In progress"
    )
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
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
        "/webhook/telegram",
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


@patch("app.main.TelegramAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_query_pending(mock_genai_cls, mock_notion_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="TASKS", raw_text="What are my pending tasks?")

    stage2_resp = MagicMock()
    stage2_resp.parsed = TaskAnalysis(intent="QUERY_PENDING", title="Query tasks")
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
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
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_notion_inst.get_pending.assert_called_once_with(limit=5)
    mock_tg_inst.send_message.assert_called_once()
    assert "Task 1" in mock_tg_inst.send_message.call_args[1]["text"]


@patch("app.main.TelegramAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_mind_module_substack(mock_genai_cls, mock_notion_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="MIND", raw_text="Idea for new Substack post about AI tools")

    stage2_resp = MagicMock()
    stage2_resp.parsed = MindEntry(
        entry_type="DRAFT_SUBSTACK",
        title="AI Tools in 2026",
        core_thesis="Modern AI tools transform workflow efficiency when chained properly.",
        content="Full draft content here...",
        summary="A review of modern AI tools.",
        tags=["Substack", "AI"],
    )
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_notion_inst = MagicMock()
    mock_notion_cls.return_value = mock_notion_inst

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 8,
        "message": {
            "message_id": 17,
            "text": "Idea for new Substack post about AI tools",
            "chat": {"id": 4444}
        }
    }
    response = client.post(
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_notion_inst.create_mind_entry.assert_called_once_with(
        entry_type="DRAFT_SUBSTACK",
        title="AI Tools in 2026",
        content="Full draft content here...",
        core_thesis="Modern AI tools transform workflow efficiency when chained properly.",
        tags=["Substack", "AI"],
    )
    mock_tg_inst.send_message.assert_called_once()
    sent_text = mock_tg_inst.send_message.call_args[1]["text"]
    assert "Substack Draft Created (Idea)" in sent_text
    assert "AI Tools in 2026" in sent_text
    assert "Modern AI tools transform" in sent_text


@patch("app.main.TelegramAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_mind_module_rambling(mock_genai_cls, mock_notion_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="MIND", raw_text="Random thoughts on distributed cache invalidation")

    stage2_resp = MagicMock()
    stage2_resp.parsed = MindEntry(
        entry_type="RAMBLING",
        title="Cache Invalidation Musings",
        core_thesis="Lease-based cache expiration prevents stale reads.",
        content="Random thoughts on distributed cache invalidation...",
        tags=["Cache", "Systems"],
    )
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_notion_inst = MagicMock()
    mock_notion_cls.return_value = mock_notion_inst

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 81,
        "message": {
            "message_id": 171,
            "text": "Random thoughts on distributed cache invalidation",
            "chat": {"id": 4444}
        }
    }
    response = client.post(
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_notion_inst.create_mind_entry.assert_called_once_with(
        entry_type="RAMBLING",
        title="Cache Invalidation Musings",
        content="Random thoughts on distributed cache invalidation...",
        core_thesis="Lease-based cache expiration prevents stale reads.",
        tags=["Cache", "Systems"],
    )
    mock_tg_inst.send_message.assert_called_once()
    sent_text = mock_tg_inst.send_message.call_args[1]["text"]
    assert "Rambling Recorded" in sent_text
    assert "Cache Invalidation Musings" in sent_text


@patch("app.main.TelegramAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_mind_module_daily_log(mock_genai_cls, mock_notion_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="MIND", raw_text="Daily wrap-up: finished sprint goals")

    stage2_resp = MagicMock()
    stage2_resp.parsed = MindEntry(
        entry_type="DAILY_LOG",
        title="Daily Wrap-Up",
        core_thesis="Sprint goals successfully completed on schedule.",
        content="Daily wrap-up: finished sprint goals...",
    )
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_notion_inst = MagicMock()
    mock_notion_cls.return_value = mock_notion_inst

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 82,
        "message": {
            "message_id": 172,
            "text": "Daily wrap-up: finished sprint goals",
            "chat": {"id": 4444}
        }
    }
    response = client.post(
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_notion_inst.create_mind_entry.assert_called_once_with(
        entry_type="DAILY_LOG",
        title="Daily Wrap-Up",
        content="Daily wrap-up: finished sprint goals...",
        core_thesis="Sprint goals successfully completed on schedule.",
        tags=[],
    )
    mock_tg_inst.send_message.assert_called_once()
    sent_text = mock_tg_inst.send_message.call_args[1]["text"]
    assert "Daily Log Recorded" in sent_text
    assert "Daily Wrap-Up" in sent_text


@patch("app.main.execute_learning_background_pipeline")
@patch("app.main.TelegramAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_learning_module(mock_genai_cls, mock_tg_cls, mock_bg_task, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="LEARNING", raw_text="I want to study Distributed Consensus")

    stage2_resp = MagicMock()
    stage2_resp.parsed = LearningRequest(
        topic="Distributed Consensus",
        category="Computer Science",
        proficiency_level="Intermediate",
        goal="Understand Raft algorithm details",
    )
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 9,
        "message": {
            "message_id": 18,
            "text": "I want to study Distributed Consensus",
            "chat": {"id": 3333}
        }
    }
    response = client.post(
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Immediate acknowledgement sent to Telegram
    mock_tg_inst.send_message.assert_called_once()
    sent_text = mock_tg_inst.send_message.call_args[1]["text"]
    assert "Building your study plan..." in sent_text

    # Background task enqueued
    mock_bg_task.assert_called_once()


@patch("app.main.execute_learning_background_pipeline")
@patch("app.main.WhatsAppAssistantClient")
@patch("app.main.genai.Client")
def test_whatsapp_webhook_learning_module(mock_genai_cls, mock_wa_cls, mock_bg_task, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="LEARNING", raw_text="I want to study Rust")

    stage2_resp = MagicMock()
    stage2_resp.parsed = LearningRequest(
        topic="Rust Programming",
        category="Computer Science",
        proficiency_level="Beginner",
    )
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_wa_inst = MagicMock()
    mock_wa_cls.return_value = mock_wa_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "111",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {
                                    "from": "1234567890",
                                    "id": "wamid.test",
                                    "text": {"body": "I want to study Rust"},
                                    "type": "text",
                                }
                            ]
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.text == "EVENT_RECEIVED"

    # Immediate acknowledgement sent to WhatsApp
    mock_wa_inst.send_message.assert_called_once_with(
        to="1234567890",
        text="Building your study plan...",
    )

    # Background task enqueued
    mock_bg_task.assert_called_once()


@patch("app.main.WhatsAppAssistantClient")
@patch("app.main.NotionAssistantClient")
@patch("app.main.genai.Client")
def test_whatsapp_webhook_tasks_module(mock_genai_cls, mock_notion_cls, mock_wa_cls, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="TASKS", raw_text="Submit report tomorrow")

    stage2_resp = MagicMock()
    stage2_resp.parsed = TaskAnalysis(
        intent="CREATE_TASK",
        title="Submit report",
        due_date="2026-08-16",
        priority="High",
    )
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_notion_inst = MagicMock()
    mock_notion_cls.return_value = mock_notion_inst

    mock_wa_inst = MagicMock()
    mock_wa_cls.return_value = mock_wa_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "111",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {
                                    "from": "9876543210",
                                    "id": "wamid.task",
                                    "text": {"body": "Submit report tomorrow"},
                                    "type": "text",
                                }
                            ]
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.text == "EVENT_RECEIVED"

    mock_notion_inst.create_task.assert_called_once()
    mock_wa_inst.send_message.assert_called_once()
    sent_text = mock_wa_inst.send_message.call_args[1]["text"]
    assert "Task created" in sent_text
    assert "Submit report" in sent_text



@patch("app.main.execute_leetcode_background_pipeline")
@patch("app.main.TelegramAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_leetcode_module(mock_genai_cls, mock_tg_cls, mock_bg_task, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="LEETCODE", raw_text="Review LC 42 Trapping Rain Water")

    stage2_resp = MagicMock()
    stage2_resp.parsed = LeetcodeReviewRequest(
        problem_name="Trapping Rain Water",
        problem_number=42,
        difficulty="Hard",
        patterns=["Two Pointers"],
        status="Review Needed",
        review_notes="Pay attention to left_max vs right_max boundaries",
    )
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 10,
        "message": {
            "message_id": 19,
            "text": "Review LC 42 Trapping Rain Water",
            "chat": {"id": 2222}
        }
    }
    response = client.post(
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Immediate ack sent to Telegram
    mock_tg_inst.send_message.assert_called_once_with(
        text="Pulling your latest solution...",
        chat_id="2222",
    )
    # Background task enqueued
    mock_bg_task.assert_called_once()


@patch("app.main.execute_leetcode_background_pipeline")
@patch("app.main.WhatsAppAssistantClient")
@patch("app.main.genai.Client")
def test_whatsapp_webhook_leetcode_module(mock_genai_cls, mock_wa_cls, mock_bg_task, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="LEETCODE", raw_text="Review my Two Sum code")

    stage2_resp = MagicMock()
    stage2_resp.parsed = LeetcodeReviewRequest(problem_name="Two Sum")
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_wa_inst = MagicMock()
    mock_wa_cls.return_value = mock_wa_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "111",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {
                                    "from": "1234567890",
                                    "id": "wamid.lc",
                                    "text": {"body": "Review my Two Sum code"},
                                    "type": "text",
                                }
                            ]
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.text == "EVENT_RECEIVED"

    # Immediate ack sent to WhatsApp
    mock_wa_inst.send_message.assert_called_once_with(
        to="1234567890",
        text="Pulling your latest solution...",
    )
    # Background task enqueued
    mock_bg_task.assert_called_once()


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
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_tg_inst.send_message.assert_called_once()
    assert raw_text in mock_tg_inst.send_message.call_args[1]["text"]


@patch("app.main.TelegramAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_telegram_send_failure_logged_not_crashed(mock_genai_cls, mock_tg_cls, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="TASKS", raw_text="Daily summary note")

    stage2_resp = MagicMock()
    stage2_resp.parsed = TaskAnalysis(intent="DAILY_LOG", log_content="Daily summary")
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
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
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Ocean v3.0 Stage 2 & Webhook Tests ---

@patch("app.main.genai.Client")
def test_parse_task_action_stage2_success(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.parsed = TaskActionAnalysis(action="MARK_DONE", task_target_title="Berkshire Dining", new_status_name="Done")
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    res = parse_task_action_stage2("Mark Berkshire Dining as done")
    assert res.action == "MARK_DONE"
    assert res.task_target_title == "Berkshire Dining"
    assert res.new_status_name == "Done"


@patch("app.main.genai.Client")
def test_parse_document_append_stage2_success(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.parsed = DocumentAppendAnalysis(
        target_document_title="Ideas for projects",
        content_to_append="Build AI voice agent for Notion",
        block_type="bulleted_list_item",
    )
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    res = parse_document_append_stage2("Add Build AI voice agent for Notion to my Ideas for projects note")
    assert res.target_document_title == "Ideas for projects"
    assert res.content_to_append == "Build AI voice agent for Notion"
    assert res.block_type == "bulleted_list_item"


@patch("app.main.execute_task_action")
@patch("app.main.TelegramAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_telegram_task_action_integration(mock_genai_cls, mock_tg_cls, mock_execute, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="TASK_ACTION", raw_text="Mark Berkshire Dining done")

    stage2_resp = MagicMock()
    stage2_resp.parsed = TaskActionAnalysis(action="MARK_DONE", task_target_title="Berkshire Dining")
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_execute.return_value = {
        "status": "ok",
        "reply_text": "✅ Marked task as **Done**!\n📌 **[Berkshire Dining](https://notion.so/123)**",
    }

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 10,
        "message": {
            "message_id": 20,
            "text": "Mark Berkshire Dining done",
            "chat": {"id": 8888}
        }
    }
    response = client.post(
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_execute.assert_called_once()
    mock_tg_inst.send_message.assert_called_once()


@patch("app.main.append_blocks_to_document")
@patch("app.main.TelegramAssistantClient")
@patch("app.main.genai.Client")
def test_webhook_telegram_document_append_integration(mock_genai_cls, mock_tg_cls, mock_append, env_setup):
    mock_genai_inst = MagicMock()
    stage1_resp = MagicMock()
    stage1_resp.parsed = ModuleClassification(module="DOCUMENT_APPEND", raw_text="Add idea to Ideas for projects")

    stage2_resp = MagicMock()
    stage2_resp.parsed = DocumentAppendAnalysis(target_document_title="Ideas for projects", content_to_append="Build AI agent")
    mock_genai_inst.models.generate_content.side_effect = [stage1_resp, stage2_resp]
    mock_genai_cls.return_value = mock_genai_inst

    mock_append.return_value = {
        "status": "ok",
        "reply_text": "📝 *Appended to Note!*",
    }

    mock_tg_inst = MagicMock()
    mock_tg_cls.return_value = mock_tg_inst

    from app.main import app
    client = TestClient(app)
    payload = {
        "update_id": 11,
        "message": {
            "message_id": 21,
            "text": "Add idea to Ideas for projects",
            "chat": {"id": 8888}
        }
    }
    response = client.post(
        "/webhook/telegram",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_append.assert_called_once()
    mock_tg_inst.send_message.assert_called_once()




