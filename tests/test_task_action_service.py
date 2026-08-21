"""Unit tests for task_action_service.py (Ocean v3.0 Task Actions)."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from app.memory import conversation_memory
from app.schemas import TaskActionAnalysis, TaskActionType
from app.task_action_service import (
    execute_task_action,
    resolve_ordinal_index,
    resolve_relative_date_string,
    resolve_target_task,
)

IST_TZ = timezone(timedelta(hours=5, minutes=30))


def test_resolve_relative_date_string_iso():
    assert resolve_relative_date_string("2026-08-30") == "2026-08-30"


def test_resolve_relative_date_string_tomorrow():
    ref = datetime(2026, 8, 21, 10, 0, 0, tzinfo=IST_TZ)
    assert resolve_relative_date_string("tomorrow", reference_dt=ref) == "2026-08-22"


def test_resolve_relative_date_string_today():
    ref = datetime(2026, 8, 21, 10, 0, 0, tzinfo=IST_TZ)
    assert resolve_relative_date_string("today", reference_dt=ref) == "2026-08-21"


def test_resolve_ordinal_index():
    assert resolve_ordinal_index("first task") == 1
    assert resolve_ordinal_index("mark the 1st done") == 1
    assert resolve_ordinal_index("the second one") == 2
    assert resolve_ordinal_index("#3") == 3
    assert resolve_ordinal_index("random task title") is None


def test_resolve_target_task_from_memory():
    user_id = "test_user_memory_1"
    conversation_memory.set_last_query_results(user_id, [
        {"title": "Berkshire Dining", "page_id": "p1", "url": "https://notion.so/p1"},
        {"title": "Leetcode 100", "page_id": "p2", "url": "https://notion.so/p2"},
    ])

    analysis = TaskActionAnalysis(action="MARK_DONE", task_target_title="first task", ordinal_index=1)
    mock_notion = MagicMock()

    task, reason = resolve_target_task(analysis, mock_notion, user_id=user_id)
    assert task is not None
    assert task["title"] == "Berkshire Dining"
    assert reason == "memory_ordinal"


def test_resolve_target_task_fuzzy_match():
    mock_notion = MagicMock()
    mock_notion.get_pending.return_value = [
        {"title": "Complete GPAF Form", "page_id": "p10", "url": "https://notion.so/p10"},
        {"title": "Pay Berkshire Dining", "page_id": "p11", "url": "https://notion.so/p11"},
    ]

    analysis = TaskActionAnalysis(action="MARK_DONE", task_target_title="GPAF form")
    task, reason = resolve_target_task(analysis, mock_notion)

    assert task is not None
    assert task["title"] == "Complete GPAF Form"
    assert "tier_" in reason or reason == "fuzzy_match"


def test_execute_task_action_mark_done():
    mock_notion = MagicMock()
    mock_notion.get_pending.return_value = [
        {"title": "Berkshire Dining", "page_id": "page-123", "url": "https://notion.so/page123"}
    ]

    analysis = TaskActionAnalysis(action="MARK_DONE", task_target_title="Berkshire Dining")
    res = execute_task_action(analysis, notion_client=mock_notion)

    assert res["status"] == "ok"
    assert res["action"] == "MARK_DONE"
    assert res["task_title"] == "Berkshire Dining"
    assert "Marked task as **Done**" in res["reply_text"]
    mock_notion.client.pages.update.assert_called_once_with(
        page_id="page-123",
        properties={"Status": {"status": {"name": "Done"}}},
    )


def test_execute_task_action_update_due_date():
    mock_notion = MagicMock()
    mock_notion.get_pending.return_value = [
        {"title": "Scholarship App", "page_id": "page-456", "url": "https://notion.so/page456"}
    ]

    ref = datetime(2026, 8, 21, 10, 0, 0, tzinfo=IST_TZ)
    analysis = TaskActionAnalysis(action="UPDATE_DUE_DATE", task_target_title="Scholarship App", new_due_date_iso="tomorrow")
    res = execute_task_action(analysis, notion_client=mock_notion, reference_dt=ref)

    assert res["status"] == "ok"
    assert res["action"] == "UPDATE_DUE_DATE"
    assert res["new_due_date"] == "2026-08-22"
    assert "2026-08-22" in res["reply_text"]
    mock_notion.client.pages.update.assert_called_once_with(
        page_id="page-456",
        properties={"Due Date": {"date": {"start": "2026-08-22"}}},
    )


def test_execute_task_action_delete_task():
    mock_notion = MagicMock()
    mock_notion.get_pending.return_value = [
        {"title": "Duplicate Leetcode", "page_id": "page-789", "url": "https://notion.so/page789"}
    ]

    analysis = TaskActionAnalysis(action="DELETE_TASK", task_target_title="Duplicate Leetcode")
    res = execute_task_action(analysis, notion_client=mock_notion)

    assert res["status"] == "ok"
    assert res["action"] == "DELETE_TASK"
    assert "Archived task" in res["reply_text"]
    mock_notion.client.pages.update.assert_called_once_with(
        page_id="page-789",
        archived=True,
    )


def test_execute_task_action_not_found():
    mock_notion = MagicMock()
    mock_notion.get_pending.return_value = []

    analysis = TaskActionAnalysis(action="MARK_DONE", task_target_title="Nonexistent task")
    res = execute_task_action(analysis, notion_client=mock_notion)

    assert res["status"] == "not_found"
    assert "Could not find" in res["reply_text"]
