"""Compound Intent Execution Service for Ocean v3.2.

Executes a multi-step CompoundPlan produced by the COMPOUND module parser,
dispatching each atomic action to the appropriate underlying service and
collating results into a single reply.
"""

import logging
from typing import Any, Dict, List, Optional

from app.notion_client import NotionAssistantClient
from app.schemas import CompoundAction, CompoundActionType, CompoundPlan, WorkspaceEntryItem
from app.task_action_service import (
    TaskActionAnalysis,
    execute_batch_set_priority,
    execute_batch_task_action,
    execute_set_priority,
    execute_task_action,
)
from app.schemas import BatchTaskActionAnalysis

logger = logging.getLogger("notion-assistant.compound_service")


def _execute_batch_delete(step: CompoundAction, notion_client: NotionAssistantClient) -> str:
    analysis = BatchTaskActionAnalysis(
        action="DELETE_TASK",
        target_query=step.target_query or "",
    )
    res = execute_batch_task_action(analysis, notion_client=notion_client)
    count = res.get("count", 0)
    query = step.target_query or "matching"
    if count == 0:
        return f"No tasks found matching '{query}' to archive."
    return f"Archived {count} task(s) matching '{query}'."


def _execute_archive_task(step: CompoundAction, notion_client: NotionAssistantClient, user_id: Optional[str]) -> str:
    analysis = TaskActionAnalysis(
        action="DELETE_TASK",
        task_target_title=step.target_title or "",
    )
    res = execute_task_action(analysis, notion_client=notion_client, user_id=user_id)
    title = res.get("task_title") or step.target_title or "task"
    if res.get("status") == "ok":
        return f"Archived '{title}'."
    return res.get("reply_text") or f"Could not archive '{title}'."


def _execute_task_set_prio(step: CompoundAction, notion_client: NotionAssistantClient, user_id: Optional[str]) -> str:
    res = execute_set_priority(
        task_query=step.target_title or "",
        priority=step.priority or "Medium",
        notion_client=notion_client,
        user_id=user_id,
    )
    priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(step.priority or "", "⚡")
    title = res.get("task_title") or step.target_title or "task"
    if res.get("status") == "ok":
        return f"{priority_emoji} Set '{title}' to {step.priority} priority."
    return res.get("reply_text") or f"Could not set priority on '{title}'."


def _execute_batch_set_prio(step: CompoundAction, notion_client: NotionAssistantClient) -> str:
    res = execute_batch_set_priority(
        target_query=step.target_query or "",
        priority=step.priority or "Medium",
        notion_client=notion_client,
    )
    priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(step.priority or "", "⚡")
    count = res.get("count", 0)
    query = step.target_query or "matching tasks"
    if count == 0:
        return f"No tasks matched '{query}' for priority update."
    return f"{priority_emoji} Set {count} task(s) matching '{query}' to {step.priority} priority."


def _execute_move_to_list(step: CompoundAction, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    from app.workspace_service import add_entries_to_workspace_target
    if not step.items:
        return f"No items specified to move to '{step.dest_target}'."
    workspace_items = [WorkspaceEntryItem(title=item) for item in step.items]
    res = add_entries_to_workspace_target(
        target_query=step.dest_target or "Reading List",
        items=workspace_items,
        notion_client=notion_client,
        sender_id=sender_id,
    )
    count = len(step.items)
    target = step.dest_target or "Reading List"
    if res.get("status") in ("ok", "partial"):
        titles = ", ".join(f"'{i}'" for i in step.items[:3])
        suffix = f" + {count - 3} more" if count > 3 else ""
        return f"📚 Added {titles}{suffix} to {target}."
    return res.get("reply_text") or f"Could not add items to '{target}'."


def execute_compound_plan(
    plan: CompoundPlan,
    notion_client: Optional[NotionAssistantClient] = None,
    sender_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute each CompoundAction step sequentially and collate results."""
    notion = notion_client or NotionAssistantClient()
    if notion.client is None:
        return {"status": "error", "reply_text": "❌ Notion integration is not configured or connected."}
    if not plan.steps:
        return {"status": "error", "reply_text": "⚠️ No steps found in compound plan."}

    step_results: List[str] = []
    total = len(plan.steps)

    for i, step in enumerate(plan.steps, start=1):
        try:
            action = step.action_type
            if action == CompoundActionType.BATCH_DELETE:
                result = _execute_batch_delete(step, notion)
            elif action == CompoundActionType.ARCHIVE_TASK:
                result = _execute_archive_task(step, notion, sender_id)
            elif action == CompoundActionType.TASK_SET_PRIO:
                result = _execute_task_set_prio(step, notion, sender_id)
            elif action == CompoundActionType.BATCH_SET_PRIO:
                result = _execute_batch_set_prio(step, notion)
            elif action == CompoundActionType.MOVE_TO_LIST:
                result = _execute_move_to_list(step, notion, sender_id)
            else:
                result = f"Unknown action type: {action}"
            step_results.append(f"{i}. {result}")
            logger.info("Compound step %d/%d (%s): %s", i, total, action, result)
        except Exception as exc:
            logger.error("Compound step %d/%d (%s) failed: %s", i, total, step.action_type, exc, exc_info=True)
            step_results.append(f"{i}. ❌ Step failed: {exc}")

    s = "s" if total != 1 else ""
    reply = f"⚡ *Compound action — {total} step{s} completed:*\n\n" + "\n".join(step_results)
    return {"status": "ok", "steps_completed": total, "reply_text": reply}
