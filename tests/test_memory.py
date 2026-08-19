import pytest
from app.memory import ConversationMemory
from app.schemas import ModuleClassification, TaskAnalysis
from app.main import analyze_user_text_two_stage, _handle_module_action
from unittest.mock import MagicMock, patch, AsyncMock


def test_conversation_memory_sliding_window():
    mem = ConversationMemory(max_history=4, ttl_seconds=300)
    sender = "1234567890"

    mem.add_user_message(sender, "Message 1")
    mem.add_assistant_message(sender, "Reply 1")
    mem.add_user_message(sender, "Message 2")
    mem.add_assistant_message(sender, "Reply 2")
    mem.add_user_message(sender, "Message 3")

    history = mem.get_history(sender, max_turns=10)
    # Should only retain max_history=4 items
    assert len(history) == 4
    assert history[0]["content"] == "Reply 1"
    assert history[1]["content"] == "Message 2"
    assert history[2]["content"] == "Reply 2"
    assert history[3]["content"] == "Message 3"


def test_conversation_memory_format_context_prompt():
    mem = ConversationMemory(max_history=6)
    sender = "user_test"

    mem.add_user_message(sender, "list out my high priority tasks")
    mem.add_assistant_message(sender, "📋 Pending Tasks (High Priority): ...")

    context = mem.format_context_prompt(sender)
    assert "User: list out my high priority tasks" in context
    assert "Assistant: 📋 Pending Tasks (High Priority):" in context


def test_conversation_memory_query_state():
    mem = ConversationMemory()
    sender = "user_query"

    mem.update_query_state(sender, last_module="TASKS", last_intent="QUERY_PENDING", priority_filter="High", last_offset=5)
    state = mem.get_last_query_state(sender)

    assert state.get("priority_filter") == "High"
    assert state.get("last_offset") == 5

    mem.clear(sender)
    assert mem.get_last_query_state(sender) == {}


@patch("app.main.classify_module_stage1")
@patch("app.main.parse_tasks_stage2")
def test_anti_rambling_guardrail_redirects_short_query(mock_parse_tasks, mock_classify):
    mock_classify.return_value = ModuleClassification(module="MIND", raw_text="others?")
    mock_parse_tasks.return_value = TaskAnalysis(intent="QUERY_PENDING", is_followup=True)

    module, parsed = analyze_user_text_two_stage("others?")
    assert module == "TASKS"
    mock_parse_tasks.assert_called_once()


@pytest.mark.anyio
async def test_handle_module_action_pagination():
    from app.memory import conversation_memory

    sender = "paginate_user"
    conversation_memory.clear(sender)
    conversation_memory.update_query_state(
        sender,
        last_module="TASKS",
        last_intent="QUERY_PENDING",
        priority_filter="High",
        last_offset=0,
    )

    mock_notion = MagicMock()
    mock_notion.get_pending.return_value = [
        {"title": "Task 6", "due_date": "2026-08-25", "page_id": "p6", "url": "https://notion.so/p6"},
        {"title": "Task 7", "due_date": "2026-08-26", "page_id": "p7", "url": "https://notion.so/p7"},
    ]

    task_res = TaskAnalysis(intent="QUERY_PENDING", is_followup=True)
    reply = await _handle_module_action("TASKS", task_res, "others?", mock_notion, sender_id=sender)

    assert "Items 6-7" in reply
    assert "High Priority" in reply
    assert "Task 6" in reply
    assert "https://notion.so/p6" in reply

    # Check updated offset in state
    state = conversation_memory.get_last_query_state(sender)
    assert state.get("last_offset") == 5
