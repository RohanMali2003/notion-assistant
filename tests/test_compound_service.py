"""Unit tests for compound_service.py (Ocean v3.2 Compound Intent Execution)."""

from unittest.mock import MagicMock, patch
import pytest

from app.compound_service import execute_compound_plan
from app.schemas import CompoundAction, CompoundActionType, CompoundPlan


def test_execute_compound_plan_empty():
    plan = CompoundPlan(steps=[])
    mock_notion = MagicMock()
    res = execute_compound_plan(plan, notion_client=mock_notion)
    assert res["status"] == "error"
    assert "No steps" in res["reply_text"]


def test_execute_compound_plan_multi_step_dispatch():
    mock_notion = MagicMock()
    mock_notion.get_pending.return_value = [
        {"title": "Read & Annotate: The Pragmatic Programmer Chapter 1", "page_id": "p1"},
        {"title": "Read & Annotate: The Pragmatic Programmer Chapter 2", "page_id": "p2"},
        {"title": "Ask GOATmini to make Ocean a little smarter", "page_id": "p3"},
        {"title": "Deep Learning paper reading", "page_id": "p4"},
    ]

    plan = CompoundPlan(
        steps=[
            # Step 1: Batch delete Read & Annotate tasks
            CompoundAction(
                action_type=CompoundActionType.BATCH_DELETE,
                target_query="Read & Annotate",
            ),
            # Step 2: Set GOATmini task to High priority
            CompoundAction(
                action_type=CompoundActionType.TASK_SET_PRIO,
                target_title="Ask GOATmini to make Ocean a little smarter",
                priority="High",
            ),
            # Step 3: Set learning tasks to Low priority
            CompoundAction(
                action_type=CompoundActionType.BATCH_SET_PRIO,
                target_query="learning",
                priority="Low",
            ),
            # Step 4: Move books to Reading List
            CompoundAction(
                action_type=CompoundActionType.MOVE_TO_LIST,
                dest_target="Reading List",
                items=["The Pragmatic Programmer", "A Philosophy of Software Design"],
            ),
        ]
    )

    with patch("app.compound_service.add_entries_to_workspace_target") as mock_workspace:
        mock_workspace.return_value = {
            "status": "ok",
            "reply_text": "📚 Added 'The Pragmatic Programmer', 'A Philosophy of Software Design' to Reading List.",
        }

        res = execute_compound_plan(plan, notion_client=mock_notion, sender_id="test_user_compound")

        assert res["status"] == "ok"
        assert res["steps_completed"] == 4
        reply = res["reply_text"]
        assert "4 steps completed" in reply
        assert "1. 🗑️ Archived 2 task(s) matching 'Read & Annotate'." in reply
        assert "2. 🔴 Set 'Ask GOATmini to make Ocean a little smarter' to High priority." in reply
        assert "3. 🟢 Set 1 task(s) matching 'learning' to Low priority." in reply
        assert "4. 📚 Added 'The Pragmatic Programmer', 'A Philosophy of Software Design' to Reading List." in reply
