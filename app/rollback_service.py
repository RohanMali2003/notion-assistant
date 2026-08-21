"""Autonomous Rollback, Transaction Recovery, and Compound Correction Service.

Ocean v3.3 Engine:
1. Audits and tracks atomic mutation records (created tasks, database rows, appended notes).
2. Executes full or partial rollbacks (archiving created pages/tasks, reverting modified properties).
3. Handles compound corrective commands (e.g. "no, delete those two tasks and put it in reading list").
"""

import logging
from typing import Any, Dict, List, Optional

from app.memory import conversation_memory
from app.notion_client import NotionAssistantClient, clean_math_and_markdown
from app.schemas import RollbackAnalysis, WorkspaceEntryItem
from app.workspace_service import add_entries_to_workspace_target

logger = logging.getLogger("notion-assistant.rollback")


def execute_rollback(
    sender_id: Optional[str],
    rollback_analysis: Optional[RollbackAnalysis] = None,
    notion_client: Optional[NotionAssistantClient] = None,
) -> Dict[str, Any]:
    """Execute undo / rollback of the last recorded mutation for the user."""
    notion = notion_client or NotionAssistantClient()

    if not sender_id:
        return {
            "status": "error",
            "message": "Missing sender_id for rollback.",
            "reply_text": "⚠️ Could not identify conversation session to roll back.",
        }

    if notion.client is None:
        return {
            "status": "error",
            "message": "Notion client not initialized.",
            "reply_text": "❌ Notion integration is not configured or connected.",
        }

    # 1. Retrieve the latest mutation record
    mutation = conversation_memory.get_last_mutation(sender_id)

    if not mutation:
        # Fallback: Check if user recently had a task action in memory
        return {
            "status": "not_found",
            "message": "No recent mutations found to roll back.",
            "reply_text": "🔄 *Rollback Status*\n\nNo recent actions or changes found in this session to undo.",
        }

    action_type = mutation.get("action_type", "")
    target_title = mutation.get("target_title", "items")
    affected_items = mutation.get("affected_items", [])
    rollback_data = mutation.get("rollback_data", {})

    rolled_back_items: List[Dict[str, Any]] = []
    errors: List[str] = []

    try:
        # CASE 1: CREATED ITEMS (Tasks Tracker or Database Rows)
        if action_type in ("CREATE_TASK", "CREATE_BATCH_TASKS", "WORKSPACE_INGEST", "TASKS_CREATE"):
            for item in affected_items:
                page_id = item.get("id")
                item_title = item.get("title", "Untitled")
                if page_id:
                    try:
                        notion._request_with_retry(
                            notion.client.pages.update,
                            page_id=page_id,
                            archived=True,
                        )
                        rolled_back_items.append(item)
                    except Exception as err:
                        logger.error("Failed to archive page %s during rollback: %s", page_id, err)
                        errors.append(f"{item_title}: {err}")
                else:
                    rolled_back_items.append(item)

        # CASE 2: UPDATED TASK PROPERTIES
        elif action_type in ("UPDATE_TASK", "UPDATE_TASK_STATUS"):
            page_id = rollback_data.get("page_id")
            prev_status = rollback_data.get("previous_status")
            prev_due = rollback_data.get("previous_due_date")
            props_to_restore = {}
            if prev_status:
                props_to_restore["Status"] = {"status": {"name": prev_status}}
            if prev_due:
                props_to_restore["Due Date"] = {"date": {"start": prev_due}}

            if page_id and props_to_restore:
                try:
                    notion._request_with_retry(
                        notion.client.pages.update,
                        page_id=page_id,
                        properties=props_to_restore,
                    )
                    rolled_back_items.extend(affected_items)
                except Exception as err:
                    logger.error("Failed to revert properties for page %s: %s", page_id, err)
                    errors.append(str(err))

        # CASE 3: DELETED / ARCHIVED TASK
        elif action_type in ("DELETE_TASK", "ARCHIVE_TASK"):
            page_id = rollback_data.get("page_id")
            if page_id:
                try:
                    notion._request_with_retry(
                        notion.client.pages.update,
                        page_id=page_id,
                        archived=False,
                    )
                    rolled_back_items.extend(affected_items)
                except Exception as err:
                    logger.error("Failed to unarchive page %s: %s", page_id, err)
                    errors.append(str(err))

        # Pop mutation from stack upon successful rollback
        conversation_memory.pop_last_mutation(sender_id)

    except Exception as exc:
        logger.error("Unexpected error executing rollback: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "reply_text": f"❌ Failed to roll back last action: {exc}",
        }

    # 2. Check for Compound Corrective Rerouting (e.g. "no, delete those two tasks and put it in reading list")
    is_compound_reroute = (
        rollback_analysis is not None
        and (
            rollback_analysis.command == "CORRECTION_AND_REROUTE"
            or bool(rollback_analysis.new_target_title)
        )
    )

    if is_compound_reroute:
        new_target = rollback_analysis.new_target_title or "Reading List"
        items_to_add = []

        if rollback_analysis.extracted_items:
            items_to_add = [WorkspaceEntryItem(title=it) for it in rollback_analysis.extracted_items]
        elif rolled_back_items:
            # Re-use titles of the rolled-back items if not explicitly extracted
            for it in rolled_back_items:
                raw_title = it.get("title", "")
                # Clean prefix words like "Read " if it was prepended during task creation
                clean_t = raw_title
                if clean_t.lower().startswith("read "):
                    clean_t = clean_t[5:].strip()
                if clean_t:
                    items_to_add.append(WorkspaceEntryItem(title=clean_t))

        if items_to_add:
            reroute_res = add_entries_to_workspace_target(
                target_query=new_target,
                items=items_to_add,
                notion_client=notion,
                sender_id=sender_id,
            )

            # Build unified compound message
            undo_lines = [f"• ~{it.get('title', 'Item')}~" for it in rolled_back_items]
            combined_reply = (
                f"🗑️ *Deleted Mistaken Tasks:*\n"
                + "\n".join(undo_lines)
                + f"\n\n"
                + reroute_res.get("reply_text", "")
            )

            return {
                "status": "ok",
                "action": "CORRECTION_AND_REROUTE",
                "rolled_back_items": rolled_back_items,
                "rerouted_target": new_target,
                "rerouted_result": reroute_res,
                "reply_text": clean_math_and_markdown(combined_reply),
            }

    # Standard Rollback confirmation
    if rolled_back_items:
        items_text = "\n".join(f"• ~{it.get('title', 'Item')}~" for it in rolled_back_items)
        reply = (
            f"🔄 *Rolled Back Last Action!*\n\n"
            f"Safely removed {len(rolled_back_items)} item(s) from **{target_title}**:\n"
            f"{items_text}"
        )
    else:
        reply = f"🔄 *Rolled Back Last Action!*\n\nReverted changes to **{target_title}**."

    return {
        "status": "ok",
        "action": "ROLLBACK",
        "rolled_back_items": rolled_back_items,
        "reply_text": clean_math_and_markdown(reply),
    }
