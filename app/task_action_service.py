"""Task Action Execution Service for Ocean v3.0.

Provides bidirectional task manipulation:
- Status updates (Mark Done, Mark In Progress)
- Rescheduling due dates with relative date evaluation
- Task archival and deletion
- Fuzzy matching & ordinal resolution against conversation memory
"""

import difflib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.matcher import entity_resolver, resolve_natural_date
from app.memory import conversation_memory
from app.notion_client import NotionAssistantClient
from app.schemas import BatchTaskActionAnalysis, TaskActionAnalysis, TaskActionType

logger = logging.getLogger("notion-assistant.task_action")

# Standard IST timezone offset (+05:30) for relative date calculation fallback
IST_TZ = timezone(timedelta(hours=5, minutes=30))


def resolve_relative_date_string(
    date_text: str,
    reference_dt: Optional[datetime] = None,
) -> Optional[str]:
    """Resolve relative or natural language date string using app.matcher date engine."""
    return resolve_natural_date(date_text, reference_dt=reference_dt)


def _calculate_similarity(query: str, target: str) -> float:
    """Calculate token overlap + SequenceMatcher similarity between query and candidate target."""
    q_clean = query.strip().lower()
    t_clean = target.strip().lower()

    if not q_clean or not t_clean:
        return 0.0
    if q_clean == t_clean:
        return 1.0
    if q_clean in t_clean or t_clean in q_clean:
        return 0.85 + (0.15 * min(len(q_clean), len(t_clean)) / max(len(q_clean), len(t_clean)))

    # Token set overlap ratio
    q_words = set(re.findall(r"\w+", q_clean))
    t_words = set(re.findall(r"\w+", t_clean))

    if not q_words or not t_words:
        return 0.0

    intersection = q_words.intersection(t_words)
    overlap_ratio = len(intersection) / float(len(q_words))

    seq_ratio = difflib.SequenceMatcher(None, q_clean, t_clean).ratio()
    return max(overlap_ratio, seq_ratio)


def resolve_ordinal_index(target_title: str) -> Optional[int]:
    """Parse ordinal (first, 2nd, 3rd) or cardinal words (one, two, three) from user target title."""
    clean = target_title.strip().lower()
    # Check ordinals first
    if "first" in clean or "1st" in clean:
        return 1
    elif "second" in clean or "2nd" in clean:
        return 2
    elif "third" in clean or "3rd" in clean:
        return 3
    elif "fourth" in clean or "4th" in clean:
        return 4
    elif "fifth" in clean or "5th" in clean:
        return 5

    # Check cardinals if explicit position words exist
    match = re.search(r"\b(one|two|three|four|five|\d+)\b", clean)
    if match:
        val = match.group(1)
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        if val in word_map:
            return word_map[val]
        elif val.isdigit():
            idx = int(val)
            if 1 <= idx <= 20:
                return idx
    return None


def resolve_target_task(
    action_analysis: TaskActionAnalysis,
    notion_client: NotionAssistantClient,
    user_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Resolve target Notion task page dict using ordinal lookup or 3-tier Entity Resolution Cascade."""
    target_title = action_analysis.task_target_title or ""
    ordinal = action_analysis.ordinal_index or resolve_ordinal_index(target_title)

    # 1. Check conversation memory for previous query results if ordinal referenced
    if ordinal and user_id:
        recent_results = conversation_memory.get_last_query_results(user_id)
        if recent_results and 1 <= ordinal <= len(recent_results):
            matched_task = recent_results[ordinal - 1]
            logger.info("Resolved target task by ordinal #%d from memory: %s", ordinal, matched_task.get("title"))
            return matched_task, "memory_ordinal"

    # 2. Fetch candidate tasks from Notion Tasks Tracker
    if notion_client.client is None:
        return None, "no_client"

    # Fetch active pending tasks first
    pending_tasks = notion_client.get_pending(limit=50)

    if not pending_tasks:
        # Fallback: query database directly if get_pending returned empty
        try:
            db_id = notion_client.database_id
            resp = notion_client._query_database(database_id=db_id, page_size=50)
            pending_tasks = [notion_client._parse_page_to_dict(p) for p in resp.get("results", [])]
        except Exception as exc:
            logger.warning("Failed to query tasks database for action target: %s", exc)
            return None, "query_failed"

    if not pending_tasks:
        return None, "no_tasks_found"

    # Handle ordinal reference without memory
    if ordinal and 1 <= ordinal <= len(pending_tasks):
        return pending_tasks[ordinal - 1], "ordinal"

    # 3. 3-Tier Entity Resolution Cascade (RapidFuzz -> MiniLM -> Gemini)
    best_task, tier_reason, score = entity_resolver.resolve_entity(
        query=target_title,
        candidates=pending_tasks,
        key_fn=lambda t: t.get("title", ""),
    )

    if best_task:
        logger.info("Entity resolved task '%s' via %s (score=%.2f) for query '%s'", best_task.get("title"), tier_reason, score, target_title)
        return best_task, f"tier_{tier_reason}"

    return None, "not_found"


def execute_task_action(
    action_analysis: TaskActionAnalysis,
    notion_client: Optional[NotionAssistantClient] = None,
    reference_dt: Optional[datetime] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute task action (status update, postpone due date, delete/archive) on target Notion task."""
    notion = notion_client or NotionAssistantClient()

    if notion.client is None:
        return {
            "status": "error",
            "message": "Notion client not initialized.",
            "reply_text": "❌ Notion integration is not configured or connected.",
        }

    task, match_reason = resolve_target_task(action_analysis, notion, user_id=user_id)

    if match_reason == "tier_ambiguous_menu" and isinstance(task, list):
        candidates = task
        if user_id:
            conversation_memory.set_pending_menu(user_id, {
                "module": "TASK_ACTION",
                "action_analysis": action_analysis.model_dump(),
                "candidates": candidates,
            })
        lines = [f"• **#{idx+1}**: {c.get('title', 'Untitled')}" for idx, c in enumerate(candidates)]
        reply = (
            f"🔍 *Multiple matching tasks found for '{action_analysis.task_target_title}':*\n\n"
            + "\n".join(lines)
            + "\n\n💡 *Reply with 1, 2, or 3 to select the intended task.*"
        )
        return {
            "status": "ambiguous_menu",
            "candidates": candidates,
            "reply_text": reply,
        }

    if not task or not isinstance(task, dict):
        return {
            "status": "not_found",
            "message": f"Task matching '{action_analysis.task_target_title}' not found.",
            "reply_text": f"❓ Could not find an active task matching **'{action_analysis.task_target_title}'**.",
        }

    page_id = task.get("page_id")
    task_title = task.get("title", "Untitled Task")
    task_url = task.get("url", f"https://notion.so/{page_id.replace('-', '')}" if page_id else "#")
    action_type = action_analysis.action

    try:
        # Action 1: MARK_DONE
        if action_type == TaskActionType.MARK_DONE:
            status_name = action_analysis.new_status_name or "Done"
            notion.client.pages.update(
                page_id=page_id,
                properties={"Status": {"status": {"name": status_name}}},
            )
            if user_id:
                conversation_memory.record_mutation(
                    sender_id=user_id,
                    action_type="UPDATE_TASK",
                    target_title=task_title,
                    affected_items=[{"id": page_id, "title": task_title, "url": task_url, "type": "task"}],
                    rollback_data={
                        "page_id": page_id,
                        "previous_status": task.get("status", "Not started"),
                    },
                    summary=f"Updated status of {task_title} to {status_name}",
                )
            reply = (
                f"✅ Marked task as **Done**!\n\n"
                f"📌 **[{task_title}]({task_url})**\n"
                f"🎉 Status updated to *{status_name}*."
            )
            try:
                from app.motion import evidence_ingestion_engine
                evidence_ingestion_engine.ingest_task_completion(
                    task_title=task_title,
                    notes=f"Completed task: {task_title}",
                    page_url=task_url,
                )
            except Exception as err:
                logger.debug("Motion evidence ingestion skipped: %s", err)

            return {
                "status": "ok",
                "action": "MARK_DONE",
                "task_title": task_title,
                "task_url": task_url,
                "reply_text": reply,
            }

        # Action 2: MARK_IN_PROGRESS
        elif action_type == TaskActionType.MARK_IN_PROGRESS:
            status_name = action_analysis.new_status_name or "In progress"
            notion.client.pages.update(
                page_id=page_id,
                properties={"Status": {"status": {"name": status_name}}},
            )
            if user_id:
                conversation_memory.record_mutation(
                    sender_id=user_id,
                    action_type="UPDATE_TASK",
                    target_title=task_title,
                    affected_items=[{"id": page_id, "title": task_title, "url": task_url, "type": "task"}],
                    rollback_data={
                        "page_id": page_id,
                        "previous_status": task.get("status", "Not started"),
                    },
                    summary=f"Updated status of {task_title} to {status_name}",
                )
            reply = (
                f"🔄 Task set to **In progress**!\n\n"
                f"📌 **[{task_title}]({task_url})**\n"
                f"⚡ Status updated to *{status_name}*."
            )
            return {
                "status": "ok",
                "action": "MARK_IN_PROGRESS",
                "task_title": task_title,
                "task_url": task_url,
                "reply_text": reply,
            }

        # Action 3: UPDATE_DUE_DATE
        elif action_type == TaskActionType.UPDATE_DUE_DATE:
            raw_date = action_analysis.new_due_date_iso
            resolved_date = resolve_relative_date_string(raw_date or "", reference_dt=reference_dt) or raw_date

            if not resolved_date:
                return {
                    "status": "error",
                    "message": "Invalid or missing due date.",
                    "reply_text": f"⚠️ Could not resolve new due date for **'{task_title}'**.",
                }

            notion.client.pages.update(
                page_id=page_id,
                properties={"Due Date": {"date": {"start": resolved_date}}},
            )
            if user_id:
                conversation_memory.record_mutation(
                    sender_id=user_id,
                    action_type="UPDATE_TASK",
                    target_title=task_title,
                    affected_items=[{"id": page_id, "title": task_title, "url": task_url, "type": "task"}],
                    rollback_data={
                        "page_id": page_id,
                        "previous_due_date": task.get("due_date"),
                    },
                    summary=f"Rescheduled {task_title} to {resolved_date}",
                )
            reply = (
                f"📅 Rescheduled task due date!\n\n"
                f"📌 **[{task_title}]({task_url})**\n"
                f"⏰ New due date: **{resolved_date}**."
            )
            return {
                "status": "ok",
                "action": "UPDATE_DUE_DATE",
                "new_due_date": resolved_date,
                "task_title": task_title,
                "task_url": task_url,
                "reply_text": reply,
            }

        # Action 4: DELETE_TASK (Archiving)
        elif action_type == TaskActionType.DELETE_TASK:
            notion.client.pages.update(
                page_id=page_id,
                archived=True,
            )
            if user_id:
                conversation_memory.record_mutation(
                    sender_id=user_id,
                    action_type="DELETE_TASK",
                    target_title=task_title,
                    affected_items=[{"id": page_id, "title": task_title, "url": task_url, "type": "task"}],
                    rollback_data={
                        "page_id": page_id,
                    },
                    summary=f"Archived task {task_title}",
                )
            reply = (
                f"🗑️ Archived task!\n\n"
                f"📌 **[{task_title}]({task_url})**\n"
                f"📦 Safely moved to Notion archive."
            )
            return {
                "status": "ok",
                "action": "DELETE_TASK",
                "task_title": task_title,
                "task_url": task_url,
                "reply_text": reply,
            }

        # Action 5: SET_PRIORITY
        elif action_type == TaskActionType.SET_PRIORITY:
            priority_name = action_analysis.new_priority
            if not priority_name:
                return {
                    "status": "error",
                    "message": "Missing new_priority for SET_PRIORITY action.",
                    "reply_text": f"⚠️ Could not set priority on **'{task_title}'** — no priority value provided.",
                }
            notion.client.pages.update(
                page_id=page_id,
                properties={"Priority": {"select": {"name": priority_name}}},
            )
            if user_id:
                conversation_memory.record_mutation(
                    sender_id=user_id,
                    action_type="UPDATE_TASK",
                    target_title=task_title,
                    affected_items=[{"id": page_id, "title": task_title, "url": task_url, "type": "task"}],
                    rollback_data={
                        "page_id": page_id,
                        "previous_priority": task.get("priority"),
                    },
                    summary=f"Set priority of {task_title} to {priority_name}",
                )
            priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(priority_name, "⚡")
            reply = (
                f"{priority_emoji} Priority updated!\n\n"
                f"📌 **[{task_title}]({task_url})**\n"
                f"⚡ Priority set to *{priority_name}*."
            )
            return {
                "status": "ok",
                "action": "SET_PRIORITY",
                "task_title": task_title,
                "task_url": task_url,
                "reply_text": reply,
            }

        else:
            return {
                "status": "error",
                "message": f"Unsupported action type '{action_type}'",
                "reply_text": f"⚠️ Unsupported task action: {action_type}.",
            }

    except Exception as exc:
        logger.error("Failed to execute task action %s on page %s: %s", action_type, page_id, exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "reply_text": f"❌ Failed to update task **'{task_title}'**: {exc}",
        }


def execute_batch_task_action(
    batch_analysis: BatchTaskActionAnalysis,
    notion_client: Optional[NotionAssistantClient] = None,
    reference_dt: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Execute batch task action across multiple active tasks (e.g. mark all UMass tasks as done)."""
    notion = notion_client or NotionAssistantClient()

    if notion.client is None:
        return {
            "status": "error",
            "message": "Notion client not initialized.",
            "reply_text": "❌ Notion integration is not configured or connected.",
        }

    # Query matching tasks — use high limit to ensure all tasks are fetched
    pending_tasks = notion.get_pending(
        limit=200,
        priority=batch_analysis.priority_filter,
        tag=batch_analysis.tag_filter,
    )

    raw_query = (batch_analysis.target_query or "").strip()
    if raw_query:
        # Normalize: match both '&' and 'and' variants so LLM phrasing doesn't matter
        query_lower = raw_query.lower()
        query_amp = query_lower.replace(" and ", " & ")   # "read and annotate" -> "read & annotate"
        query_and = query_lower.replace(" & ", " and ")   # "read & annotate" -> "read and annotate"
        pending_tasks = [
            t for t in pending_tasks
            if any(
                variant in t.get("title", "").lower()
                for variant in (query_lower, query_amp, query_and)
            )
        ]

    if not pending_tasks:
        return {
            "status": "not_found",
            "count": 0,
            "reply_text": "🎉 No matching active tasks found for batch update.",
        }

    action_type = batch_analysis.action
    updated_titles = []

    for task in pending_tasks:
        p_id = task.get("page_id")
        t_title = task.get("title", "Untitled")
        try:
            if action_type == TaskActionType.MARK_DONE:
                status_name = batch_analysis.new_status_name or "Done"
                notion.client.pages.update(page_id=p_id, properties={"Status": {"status": {"name": status_name}}})
            elif action_type == TaskActionType.MARK_IN_PROGRESS:
                status_name = batch_analysis.new_status_name or "In progress"
                notion.client.pages.update(page_id=p_id, properties={"Status": {"status": {"name": status_name}}})
            elif action_type == TaskActionType.UPDATE_DUE_DATE:
                raw_date = batch_analysis.new_due_date_iso or ""
                resolved_date = resolve_relative_date_string(raw_date, reference_dt=reference_dt) or raw_date
                if resolved_date:
                    notion.client.pages.update(page_id=p_id, properties={"Due Date": {"date": {"start": resolved_date}}})
            elif action_type == TaskActionType.DELETE_TASK:
                notion.client.pages.update(page_id=p_id, archived=True)

            updated_titles.append(t_title)
        except Exception as exc:
            logger.warning("Batch action item update failed for page %s: %s", p_id, exc)

    count = len(updated_titles)
    action_label = action_type.value.replace("_", " ").title()
    reply = (
        f"⚡ *Batch Task Action Completed!*\n\n"
        f"🎯 Action: **{action_label}**\n"
        f"📊 Updated **{count}** task(s):\n"
        + "\n".join(f"• {t}" for t in updated_titles[:10])
    )

    return {
        "status": "ok",
        "action": action_type.value,
        "count": count,
        "updated_titles": updated_titles,
        "reply_text": reply,
    }


def execute_set_priority(
    task_query: str,
    priority: str,
    notion_client: Optional[NotionAssistantClient] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Standalone helper: set priority on a single fuzzy-matched task by title query.

    Used by the compound service to execute TASK_SET_PRIO steps.
    """
    analysis = TaskActionAnalysis(
        action="SET_PRIORITY",
        task_target_title=task_query,
        new_priority=priority,  # type: ignore[arg-type]
    )
    return execute_task_action(analysis, notion_client=notion_client, user_id=user_id)


def execute_batch_set_priority(
    target_query: str,
    priority: str,
    notion_client: Optional[NotionAssistantClient] = None,
) -> Dict[str, Any]:
    """Set priority on all pending tasks whose title contains target_query.

    Used by the compound service to execute BATCH_SET_PRIO steps.
    """
    notion = notion_client or NotionAssistantClient()
    if notion.client is None:
        return {"status": "error", "reply_text": "❌ Notion not connected.", "count": 0}

    pending = notion.get_pending(limit=200)
    raw_q = target_query.strip()
    if raw_q:
        q_lower = raw_q.lower()
        q_amp = q_lower.replace(" and ", " & ")
        q_and = q_lower.replace(" & ", " and ")
        matched = [
            t for t in pending
            if any(v in t.get("title", "").lower() for v in (q_lower, q_amp, q_and))
        ]
    else:
        matched = pending

    if not matched:
        return {
            "status": "not_found",
            "count": 0,
            "reply_text": f"⚠️ No pending tasks found matching *'{target_query}'* to set priority.",
        }

    updated = []
    for task in matched:
        p_id = task.get("page_id")
        try:
            notion.client.pages.update(
                page_id=p_id,
                properties={"Priority": {"select": {"name": priority}}},
            )
            updated.append(task.get("title", "Untitled"))
        except Exception as exc:
            logger.warning("Batch set priority failed for page %s: %s", p_id, exc)

    count = len(updated)
    priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(priority, "⚡")
    return {
        "status": "ok",
        "count": count,
        "updated_titles": updated,
        "reply_text": f"{priority_emoji} Set **{count}** task(s) to *{priority}* priority.",
    }
