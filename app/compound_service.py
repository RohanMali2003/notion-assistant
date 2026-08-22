"""Compound Intent Execution Service for Ocean v3.2.

Executes a multi-step CompoundPlan produced by the COMPOUND module parser,
dispatching each atomic action to the appropriate canonical service (task_action,
batch_task_action, workspace_ingest, task creation) and collating results into
a single, unified summary reply.
"""

import logging
from typing import Any, Dict, List, Optional

from app.notion_client import NotionAssistantClient
from app.schemas import (
    BatchTaskActionAnalysis,
    CompoundAction,
    CompoundActionType,
    CompoundPlan,
    TaskActionAnalysis,
    WorkspaceEntryItem,
)
from app.task_action_service import execute_batch_task_action, execute_task_action
from app.workspace_service import add_entries_to_workspace_target

logger = logging.getLogger("notion-assistant.compound_service")


def _dispatch_atomic_step(
    step: CompoundAction,
    notion: NotionAssistantClient,
    sender_id: Optional[str] = None,
) -> str:
    """Dispatch an atomic CompoundAction directly to its canonical module handler."""
    action = step.action_type

    # --- 1. Single Task Actions (1-to-1) ---
    if action in (CompoundActionType.ARCHIVE_TASK, "ARCHIVE_TASK", "DELETE_TASK"):
        analysis = TaskActionAnalysis(
            action="DELETE_TASK",
            task_target_title=step.target_title or "",
        )
        res = execute_task_action(analysis, notion_client=notion, user_id=sender_id)
        title = res.get("task_title") or step.target_title or "task"
        if res.get("status") == "ok":
            return f"🗑️ Archived '{title}'."
        return f"❓ Could not find task matching '{step.target_title}' to archive."

    elif action in (CompoundActionType.TASK_SET_PRIO, "TASK_SET_PRIO", "SET_PRIORITY"):
        prio = step.priority or "Medium"
        analysis = TaskActionAnalysis(
            action="SET_PRIORITY",
            task_target_title=step.target_title or "",
            new_priority=prio,  # type: ignore[arg-type]
        )
        res = execute_task_action(analysis, notion_client=notion, user_id=sender_id)
        title = res.get("task_title") or step.target_title or "task"
        emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(prio, "⚡")
        if res.get("status") == "ok":
            return f"{emoji} Set '{title}' to {prio} priority."
        return f"❓ Could not find task matching '{step.target_title}' to set priority."

    elif action in (CompoundActionType.MARK_DONE, "MARK_DONE"):
        analysis = TaskActionAnalysis(
            action="MARK_DONE",
            task_target_title=step.target_title or "",
        )
        res = execute_task_action(analysis, notion_client=notion, user_id=sender_id)
        title = res.get("task_title") or step.target_title or "task"
        if res.get("status") == "ok":
            return f"✅ Marked '{title}' as Done."
        return f"❓ Could not find task matching '{step.target_title}' to complete."

    # --- 2. Batch Task Actions (1-to-Many) ---
    elif action in (CompoundActionType.BATCH_DELETE, "BATCH_DELETE"):
        query = step.target_query or "matching tasks"
        analysis = BatchTaskActionAnalysis(
            action="DELETE_TASK",
            target_query=step.target_query or "",
        )
        res = execute_batch_task_action(analysis, notion_client=notion, user_id=sender_id)
        count = res.get("count", 0)
        if count == 0:
            return f"⚠️ No active tasks found matching '{query}' to archive."
        return f"🗑️ Archived {count} task(s) matching '{query}'."

    elif action in (CompoundActionType.BATCH_SET_PRIO, "BATCH_SET_PRIO"):
        query = step.target_query or "matching tasks"
        prio = step.priority or "Medium"
        emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(prio, "⚡")
        analysis = BatchTaskActionAnalysis(
            action="SET_PRIORITY",
            target_query=step.target_query or "",
            new_priority=prio,  # type: ignore[arg-type]
        )
        res = execute_batch_task_action(analysis, notion_client=notion, user_id=sender_id)
        count = res.get("count", 0)
        if count == 0:
            return f"⚠️ No active tasks found matching '{query}' for priority update."
        return f"{emoji} Set {count} task(s) matching '{query}' to {prio} priority."

    elif action in (CompoundActionType.BATCH_MARK_DONE, "BATCH_MARK_DONE"):
        query = step.target_query or "matching tasks"
        analysis = BatchTaskActionAnalysis(
            action="MARK_DONE",
            target_query=step.target_query or "",
        )
        res = execute_batch_task_action(analysis, notion_client=notion, user_id=sender_id)
        count = res.get("count", 0)
        if count == 0:
            return f"⚠️ No active tasks found matching '{query}' to complete."
        return f"✅ Marked {count} task(s) matching '{query}' as Done."

    # --- 3. Workspace Ingestion / Append ---
    elif action in (CompoundActionType.MOVE_TO_LIST, "MOVE_TO_LIST", "DOCUMENT_APPEND"):
        target = step.dest_target or "Reading List"
        if not step.items:
            return f"⚠️ No items specified to add to '{target}'."
        workspace_items = [WorkspaceEntryItem(title=item) for item in step.items]
        res = add_entries_to_workspace_target(
            target_query=target,
            items=workspace_items,
            notion_client=notion,
            sender_id=sender_id,
        )
        count = len(step.items)
        if res.get("status") in ("ok", "partial"):
            titles = ", ".join(f"'{i}'" for i in step.items[:3])
            suffix = f" + {count - 3} more" if count > 3 else ""
            return f"📚 Added {titles}{suffix} to {target}."
        return res.get("reply_text") or f"⚠️ Could not add items to '{target}'."

    # --- 4. Create Task ---
    elif action in (CompoundActionType.CREATE_TASK, "CREATE_TASK"):
        title = step.target_title or "New Task"
        created = notion.create_task(
            title=title,
            priority=step.priority,
            due_date=step.due_date,
        )
        if created:
            return f"📌 Created task '{title}'."
        return f"⚠️ Could not create task '{title}'."

    else:
        return f"❓ Unsupported compound action: {action}"


def execute_compound_plan(
    plan: CompoundPlan,
    notion_client: Optional[NotionAssistantClient] = None,
    sender_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute each CompoundAction step sequentially via canonical handlers and collate results."""
    notion = notion_client or NotionAssistantClient()
    if notion.client is None:
        return {"status": "error", "reply_text": "❌ Notion integration is not configured or connected."}
    if not plan.steps:
        return {"status": "error", "reply_text": "⚠️ No steps found in compound plan."}

    step_results: List[str] = []
    total = len(plan.steps)

    for i, step in enumerate(plan.steps, start=1):
        try:
            result = _dispatch_atomic_step(step, notion=notion, sender_id=sender_id)
            step_results.append(f"{i}. {result}")
            logger.info("Compound step %d/%d (%s): %s", i, total, step.action_type, result)
        except Exception as exc:
            logger.error("Compound step %d/%d (%s) failed: %s", i, total, step.action_type, exc, exc_info=True)
            step_results.append(f"{i}. ❌ Step failed: {exc}")

    s = "s" if total != 1 else ""
    reply = f"⚡ *Compound action — {total} step{s} completed:*\n\n" + "\n".join(step_results)
    return {"status": "ok", "steps_completed": total, "reply_text": reply}
