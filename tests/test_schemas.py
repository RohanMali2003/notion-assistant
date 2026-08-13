import pytest
from pydantic import ValidationError
from app.schemas import AgentAction, ReminderItem, TaskAnalysis, TelegramWebhookUpdate, WebhookResponse


# --- AgentAction / TaskAnalysis Tests ---

def test_agent_action_valid_full_payload():
    """Test instantiating AgentAction with a full set of valid fields."""
    action = AgentAction(
        intent="CREATE_TASK",
        title="Complete quarterly report",
        priority="High",
        tag="Projects",
        due_date="2026-08-25",
        description="Write and finalize Q3 summary",
        log_content=None,
    )
    assert action.intent == "CREATE_TASK"
    assert action.title == "Complete quarterly report"
    assert action.priority == "High"
    assert action.tag == "Projects"
    assert action.due_date == "2026-08-25"
    assert action.description == "Write and finalize Q3 summary"
    assert action.log_content is None


def test_agent_action_valid_minimal_payload():
    """Test instantiating AgentAction with only the required intent field."""
    action = AgentAction(intent="CREATE_TASK")
    assert action.intent == "CREATE_TASK"
    assert action.title == ""
    assert action.priority == "Medium"
    assert action.tag == "Miscellaneous"
    assert action.due_date is None
    assert action.description is None
    assert action.log_content is None


@pytest.mark.parametrize("intent_val", ["CREATE_TASK", "DAILY_LOG", "QUERY_PENDING", "UPDATE_TASK", "QUERY_TODAY"])
def test_agent_action_valid_intents(intent_val):
    """Test that all supported intents are valid."""
    action = AgentAction(intent=intent_val)
    assert action.intent == intent_val


def test_agent_action_update_task_fields():
    """Test AgentAction with UPDATE_TASK intent and target_status field."""
    action = AgentAction(
        intent="UPDATE_TASK",
        title="pack for college",
        target_status="In progress",
        new_due_date="2026-08-28",
    )
    assert action.intent == "UPDATE_TASK"
    assert action.title == "pack for college"
    assert action.target_status == "In progress"
    assert action.new_due_date == "2026-08-28"


@pytest.mark.parametrize("priority_val", ["High", "Medium", "Low"])
def test_agent_action_valid_priorities(priority_val):
    """Test that all supported priorities are valid."""
    action = AgentAction(intent="CREATE_TASK", priority=priority_val)
    assert action.priority == priority_val


@pytest.mark.parametrize("tag_val", [
    "Finances", "UMass Admin", "Writing", "Personal Site", "Substack",
    "Open Source", "Learning", "Leetcode", "Projects", "Schoolwork", "Miscellaneous"
])
def test_agent_action_valid_tags(tag_val):
    """Test that all supported tags are valid."""
    action = AgentAction(intent="CREATE_TASK", tag=tag_val)
    assert action.tag == tag_val


def test_agent_action_invalid_intent():
    """Test that an unsupported intent raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        AgentAction(intent="DELETE_TASK")
    assert "intent" in str(exc_info.value)


def test_agent_action_invalid_priority():
    """Test that an unsupported priority raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        AgentAction(intent="CREATE_TASK", priority="Urgent")
    assert "priority" in str(exc_info.value)


def test_agent_action_invalid_tag():
    """Test that an unsupported tag raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        AgentAction(intent="CREATE_TASK", tag="NonExistentTag")
    assert "tag" in str(exc_info.value)


def test_agent_action_missing_required_intent():
    """Test that omitting the required intent field raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        AgentAction.model_validate({"title": "No intent provided"})
    assert "intent" in str(exc_info.value)


def test_agent_action_invalid_field_type():
    """Test that passing an invalid type for a field raises a ValidationError."""
    with pytest.raises(ValidationError):
        AgentAction(intent="CREATE_TASK", title={"invalid": "type"})


# --- ReminderItem Tests ---

def test_reminder_item_valid():
    item = ReminderItem(
        page_id="page-123",
        title="Buy groceries",
        due_date="2026-08-15",
        status="Pending"
    )
    assert item.page_id == "page-123"
    assert item.title == "Buy groceries"
    assert item.due_date == "2026-08-15"
    assert item.status == "Pending"


def test_reminder_item_minimal():
    item = ReminderItem(
        page_id="page-456",
        title="Call mom"
    )
    assert item.page_id == "page-456"
    assert item.title == "Call mom"
    assert item.due_date is None
    assert item.status is None


# --- Webhook Response & Telegram Update Tests ---

def test_webhook_response_default():
    resp = WebhookResponse()
    assert resp.status == "ok"
    assert resp.message is None


def test_telegram_update_schema():
    update = TelegramWebhookUpdate(
        update_id=1001,
        message={"id": 1, "text": "/start", "chat": {"id": 999}}
    )
    assert update.update_id == 1001
    assert update.message["text"] == "/start"

