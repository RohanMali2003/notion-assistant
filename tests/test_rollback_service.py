"""Unit tests for Autonomous Rollback & Compound Correction Service."""

import pytest
from unittest.mock import MagicMock, patch

from app.memory import conversation_memory
from app.rollback_service import execute_rollback
from app.schemas import RollbackAnalysis, WorkspaceEntryItem


@pytest.fixture(autouse=True)
def clean_memory():
    """Reset conversation memory before each test."""
    conversation_memory.clear()
    yield
    conversation_memory.clear()


def test_rollback_no_mutations():
    """Test rollback when no recent actions exist in memory."""
    mock_notion = MagicMock()
    res = execute_rollback("user_123", notion_client=mock_notion)
    assert res["status"] == "not_found"
    assert "No recent actions" in res["reply_text"]


def test_rollback_created_tasks():
    """Test rolling back created tasks archives them in Notion."""
    mock_notion = MagicMock()
    mock_notion._request_with_retry.return_value = {"id": "page_abc", "archived": True}

    # Record a task creation mutation
    conversation_memory.record_mutation(
        sender_id="user_123",
        action_type="CREATE_TASK",
        target_title="Read The Pragmatic Programmer",
        affected_items=[
            {"id": "page_1", "title": "Task 1", "type": "task"},
            {"id": "page_2", "title": "Task 2", "type": "task"},
        ],
        summary="Created 2 tasks",
    )

    res = execute_rollback("user_123", notion_client=mock_notion)
    assert res["status"] == "ok"
    assert len(res["rolled_back_items"]) == 2
    assert "Rolled Back Last Action" in res["reply_text"]
    assert "~Task 1~" in res["reply_text"]
    assert "~Task 2~" in res["reply_text"]

    # Verify Notion API called twice with archived=True
    assert mock_notion._request_with_retry.call_count == 2
    for call in mock_notion._request_with_retry.call_args_list:
        assert call.kwargs["archived"] is True

    # Verify mutation popped from stack
    assert conversation_memory.get_last_mutation("user_123") is None


def test_rollback_updated_task_status():
    """Test rolling back a task update restores previous properties."""
    mock_notion = MagicMock()
    mock_notion._request_with_retry.return_value = {"id": "page_task_1"}

    conversation_memory.record_mutation(
        sender_id="user_123",
        action_type="UPDATE_TASK",
        target_title="Berkshire Dining",
        affected_items=[{"id": "page_task_1", "title": "Berkshire Dining", "type": "task"}],
        rollback_data={
            "page_id": "page_task_1",
            "previous_status": "Not started",
        },
        summary="Marked Berkshire Dining done",
    )

    res = execute_rollback("user_123", notion_client=mock_notion)
    assert res["status"] == "ok"
    assert "Rolled Back Last Action" in res["reply_text"]

    # Verify Notion update called with previous status
    mock_notion._request_with_retry.assert_called_once()
    call_kwargs = mock_notion._request_with_retry.call_args.kwargs
    assert call_kwargs["page_id"] == "page_task_1"
    assert call_kwargs["properties"] == {"Status": {"status": {"name": "Not started"}}}


def test_compound_correction_and_reroute():
    """Test compound correction: undo last mistaken tasks and put items into Reading List."""
    mock_notion = MagicMock()
    mock_notion._request_with_retry.return_value = {
        "id": "new_page_book",
        "url": "https://app.notion.com/p/book1",
        "properties": {
            "Title": {"type": "title"},
            "Status": {"type": "status", "status": {"options": [{"name": "Want to Read"}]}},
        },
    }

    # Simulate the initial mistake where 2 tasks were created
    conversation_memory.record_mutation(
        sender_id="user_123",
        action_type="CREATE_TASK",
        target_title="Read The Pragmatic Programmer and A Philosophy of Software Design",
        affected_items=[
            {"id": "task_id_1", "title": "The Pragmatic Programmer", "type": "task"},
            {"id": "task_id_2", "title": "A Philosophy of Software Design", "type": "task"},
        ],
        summary="Created 2 tasks",
    )

    rollback_analysis = RollbackAnalysis(
        command="CORRECTION_AND_REROUTE",
        new_target_title="Reading List",
        correction_instruction="no, delete those two tasks and put it in reading list",
        extracted_items=["The Pragmatic Programmer", "A Philosophy of Software Design"],
    )

    with patch("app.workspace_service.find_page_node_in_workspace") as mock_find:
        from app.schemas import WorkspacePageNode
        mock_find.return_value = WorkspacePageNode(
            id="reading_list_db_id",
            title="Reading List",
            parent_type="database_id",
            is_container=True,
            children_pages=[{"id": "reading_list_db_id", "title": "Reading List", "type": "database"}],
        )

        res = execute_rollback("user_123", rollback_analysis=rollback_analysis, notion_client=mock_notion)
        assert res["status"] == "ok"
        assert res["action"] == "CORRECTION_AND_REROUTE"
        assert len(res["rolled_back_items"]) == 2
        assert "Deleted Mistaken Tasks" in res["reply_text"]
        assert "Added to Reading List" in res["reply_text"]
