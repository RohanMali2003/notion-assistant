from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.dispatcher import execute_module_action, process_incoming_text_message
from app.module_handlers import (
    BaseModuleHandler,
    HandlerRegistry,
    TaskActionHandler,
    handler_registry,
)
from app.schemas import TaskActionAnalysis, TaskAnalysis


@pytest.mark.anyio
async def test_handler_registry_polymorphic_execution():
    mock_notion = MagicMock()
    mock_notion.get_pending.return_value = [
        {"title": "Pay Tuition Bill", "page_id": "p100", "url": "https://notion.so/p100"}
    ]

    analysis = TaskActionAnalysis(action="MARK_DONE", task_target_title="Pay Tuition Bill")

    reply = await handler_registry.execute(
        module="TASK_ACTION",
        parsed_result=analysis,
        text="done with Pay Tuition Bill",
        notion_client=mock_notion,
        sender_id="test_user_handler",
    )

    assert "Marked task as **Done**" in reply
    assert "Pay Tuition Bill" in reply


@pytest.mark.anyio
async def test_custom_handler_registration():
    class DummyCustomHandler(BaseModuleHandler):
        @property
        def module_name(self) -> str:
            return "CUSTOM_TEST_MODULE"

        async def execute(self, parsed_result: Any, text: str, notion_client: Any, sender_id: Optional[str] = None) -> str:
            return f"Custom handler executed for: {text}"

    custom_registry = HandlerRegistry()
    custom_registry.register(DummyCustomHandler())

    res = await custom_registry.execute("CUSTOM_TEST_MODULE", None, "Hello world", MagicMock())
    assert res == "Custom handler executed for: Hello world"


@pytest.mark.anyio
async def test_fast_path_today_tasks():
    mock_notion = MagicMock()
    mock_notion.get_today_tasks.return_value = [
        {"title": "Review PRs", "due_date": "2026-08-22", "page_id": "p_today"}
    ]

    mock_bg = MagicMock()

    # Fast-path "today" should directly call QUERY_TODAY without Stage 1/2 LLM parsing
    reply = await process_incoming_text_message(
        text="today",
        sender_id="fast_user_1",
        background_tasks=mock_bg,
        notion_client=mock_notion,
    )

    assert reply is not None
    assert "Today's Tasks" in reply
    assert "Review PRs" in reply


@pytest.mark.anyio
async def test_fast_path_undo_rollback():
    mock_notion = MagicMock()

    with patch("app.dispatcher.execute_module_action") as mock_exec:
        mock_exec.return_value = "🔄 Rolled Back Last Action!"

        mock_bg = MagicMock()
        reply = await process_incoming_text_message(
            text="undo that",
            sender_id="fast_user_2",
            background_tasks=mock_bg,
            notion_client=mock_notion,
        )

        assert reply == "🔄 Rolled Back Last Action!"
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[0] == "ROLLBACK"
        assert args[1].command == "ROLLBACK_LAST"
