"""Unified Message and Module Action Dispatcher.

Orchestrates incoming messages across channels (WhatsApp, Telegram, API),
routes module executions, manages conversational state, and triggers background pipelines.
"""

import logging
from typing import Any, Dict, Optional
from fastapi import BackgroundTasks
from starlette.concurrency import run_in_threadpool

from app.graph_memory_service import graph_memory
from app.learning_service import execute_learning_background_pipeline
from app.leetcode_service import execute_leetcode_background_pipeline
from app.memory import conversation_memory
from app.motion import persona_router
from app.notion_client import NotionAssistantClient
from app.notifier import send_notification
from app.parsers import analyze_user_text_two_stage
from app.rollback_service import execute_rollback
from app.schemas import (
    BatchTaskActionAnalysis,
    DocumentAppendAnalysis,
    MemoryGovernanceAnalysis,
    MindEntry,
    RollbackAnalysis,
    SearchQueryAnalysis,
    TaskActionAnalysis,
    TaskActionType,
    TaskAnalysis,
    WorkspaceEntryItem,
)
from app.search_service import execute_second_brain_search_pipeline
from app.task_action_service import execute_batch_task_action, execute_task_action
from app.url_digest_service import (
    execute_url_digest_background_pipeline,
    extract_urls,
    is_url_dominant_message,
)
from app.weekly_digest_service import execute_weekly_digest_pipeline
from app.workspace_service import add_entries_to_workspace_target

logger = logging.getLogger("notion-assistant.dispatcher")


async def handle_pending_menu_reply(
    sender_id: str,
    text: str,
    notion_client: NotionAssistantClient,
) -> Optional[str]:
    """Check if user text resolves a pending interactive disambiguation menu."""
    pending_menu = conversation_memory.get_pending_menu(sender_id)
    if not pending_menu:
        return None

    clean = text.strip().lower()
    idx = None
    if clean in ("1", "1st", "first", "option 1", "choice 1", "#1"):
        idx = 0
    elif clean in ("2", "2nd", "second", "option 2", "choice 2", "#2"):
        idx = 1
    elif clean in ("3", "3rd", "third", "option 3", "choice 3", "#3"):
        idx = 2

    candidates = pending_menu.get("candidates", [])
    if idx is not None and 0 <= idx < len(candidates):
        selected_item = candidates[idx]
        conversation_memory.clear_pending_menu(sender_id)

        module = pending_menu.get("module")
        if module == "TASK_ACTION":
            raw_action = pending_menu.get("action_analysis", {})
            action_analysis = TaskActionAnalysis.model_validate(raw_action)
            page_id = selected_item.get("page_id")
            task_title = selected_item.get("title", "Untitled")
            task_url = selected_item.get("url", "#")
            action_type = action_analysis.action

            if action_type == TaskActionType.MARK_DONE:
                status_name = action_analysis.new_status_name or "Done"
                notion_client.client.pages.update(
                    page_id=page_id,
                    properties={"Status": {"status": {"name": status_name}}},
                )
                return f"✅ Selected **#{idx+1}**: Marked **[{task_title}]({task_url})** as *{status_name}*!"
            elif action_type == TaskActionType.MARK_IN_PROGRESS:
                status_name = action_analysis.new_status_name or "In progress"
                notion_client.client.pages.update(
                    page_id=page_id,
                    properties={"Status": {"status": {"name": status_name}}},
                )
                return f"🔄 Selected **#{idx+1}**: Set **[{task_title}]({task_url})** to *{status_name}*!"
            elif action_type == TaskActionType.DELETE_TASK:
                notion_client.client.pages.update(page_id=page_id, archived=True)
                return f"🗑️ Selected **#{idx+1}**: Archived **[{task_title}]({task_url})**!"

    return None


def _get_app_main_fn(name: str, fallback: Any) -> Any:
    import sys
    app_main = sys.modules.get("app.main")
    return getattr(app_main, name, fallback) if app_main else fallback


async def _handle_memory_control(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    mem_analysis: MemoryGovernanceAnalysis = parsed_result
    cmd = mem_analysis.command
    target = mem_analysis.target_entity or text

    if cmd == "FORGET":
        count, forgotten = graph_memory.forget_entity(target)
        if count > 0:
            return (
                f"🧠 *Memory Updated!* Soft-deleted {count} graph entity node(s) matching *'{target}'*:\n"
                + "\n".join(f"• ~{n}~" for n in forgotten)
            )
        return f"🧠 *Memory Inspection:* No active knowledge graph nodes found matching *'{target}'*."

    elif cmd == "UPDATE_STATUS":
        summary = mem_analysis.new_state_summary or f"Updated status to {target}"
        count = graph_memory.supersede_nodes(target, target, reason=summary)
        node = graph_memory.add_node(target, summary=summary, status="ACTIVE")
        return (
            f"🧠 *Knowledge Graph Updated!*\n\n📌 **{node['name']}**: {summary}\n*(Superseded {count} older related nodes.)*"
        )

    elif cmd in ("INSPECT_MEMORY", "SYNC_GRAPH"):
        if "sync" in text.lower() or cmd == "SYNC_GRAPH":
            sync_res = graph_memory.sync_graph_to_notion(notion_client)
            return sync_res.get("reply_text", "Knowledge Graph synced to Notion!")
        nodes = graph_memory.query_active_knowledge(query=target, limit=10)
        if nodes:
            lines = [f"• *{n['name']}* [{n['entity_type']}]: {n.get('summary', 'Active')}" for n in nodes]
            return f"🧠 *Active Knowledge Graph ({len(nodes)} items):*\n" + "\n".join(lines)
        return f"🧠 *Knowledge Graph:* No active nodes found matching *'{target}'*."

    return "🧠 Memory governance action processed."


async def _handle_batch_task_action(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    batch_analysis: BatchTaskActionAnalysis = parsed_result
    batch_fn = _get_app_main_fn("execute_batch_task_action", execute_batch_task_action)
    batch_res = await run_in_threadpool(
        batch_fn,
        batch_analysis=batch_analysis,
        notion_client=notion_client,
    )
    return batch_res.get("reply_text", "Batch action complete!")


async def _handle_task_action(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    action_analysis: TaskActionAnalysis = parsed_result
    action_fn = _get_app_main_fn("execute_task_action", execute_task_action)
    action_res = await run_in_threadpool(
        action_fn,
        action_analysis=action_analysis,
        notion_client=notion_client,
        user_id=sender_id,
    )
    return action_res.get("reply_text", "Done!")


async def _handle_rollback(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    rollback_analysis: RollbackAnalysis = parsed_result
    rollback_fn = _get_app_main_fn("execute_rollback", execute_rollback)
    rollback_res = await run_in_threadpool(
        rollback_fn,
        sender_id=sender_id,
        rollback_analysis=rollback_analysis,
        notion_client=notion_client,
    )
    return rollback_res.get("reply_text", "Rolled back changes.")


async def _handle_document_append(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    append_analysis: DocumentAppendAnalysis = parsed_result
    target_name = append_analysis.target_name or append_analysis.target_document_title
    items = append_analysis.items
    if not items and append_analysis.content_to_append:
        lines = [
            line.strip().lstrip("1234567890.-•* ").strip()
            for line in append_analysis.content_to_append.split("\n")
            if line.strip()
        ]
        items = [WorkspaceEntryItem(title=line) for line in lines] if lines else [WorkspaceEntryItem(title=append_analysis.content_to_append.strip())]

    append_fn = _get_app_main_fn("add_entries_to_workspace_target", add_entries_to_workspace_target)
    append_res = await run_in_threadpool(
        append_fn,
        target_query=target_name,
        items=items,
        default_status=append_analysis.default_status,
        block_type=append_analysis.block_type,
        notion_client=notion_client,
        sender_id=sender_id,
    )
    return append_res.get("reply_text", "Done!")


async def _handle_tasks(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    parsed_task: TaskAnalysis = parsed_result
    intent = parsed_task.intent

    if intent == "CREATE_TASK":
        created_page = await run_in_threadpool(
            notion_client.create_task,
            title=parsed_task.title or text,
            priority=parsed_task.priority,
            tag=parsed_task.tag,
            due_date=parsed_task.due_date,
            description=parsed_task.description,
        )
        page_url = created_page.get("url") if isinstance(created_page, dict) else None
        page_id = created_page.get("id") if isinstance(created_page, dict) else None
        if not page_url and page_id:
            clean_id = page_id.replace("-", "")
            page_url = f"https://www.notion.so/{clean_id}"

        reply_text = f"✅ Task created: *{parsed_task.title or text}*"
        if page_url:
            reply_text += f"\n🔗 {page_url}"
        if parsed_task.due_date:
            reply_text += f"\n📅 Due: {parsed_task.due_date}"
        if parsed_task.priority:
            reply_text += f"\n⚡ Priority: {parsed_task.priority}"
        if parsed_task.tag:
            reply_text += f"\n🏷 Tag: {parsed_task.tag}"

        if sender_id:
            conversation_memory.update_query_state(sender_id, last_module="TASKS", last_intent="CREATE_TASK")
            if page_id:
                conversation_memory.record_mutation(
                    sender_id=sender_id,
                    action_type="CREATE_TASK",
                    target_title=parsed_task.title or text,
                    affected_items=[{
                        "id": page_id,
                        "title": parsed_task.title or text,
                        "url": page_url or "",
                        "type": "task",
                    }],
                    summary=f"Created task: {parsed_task.title or text}",
                )
        return reply_text

    elif intent == "UPDATE_TASK":
        target_status = parsed_task.target_status or "In progress"
        title_query = parsed_task.title or text
        success, matched_title, updated_page = await run_in_threadpool(
            notion_client.update_task_status,
            title_query=title_query,
            status_name=target_status,
            new_due_date=parsed_task.new_due_date or parsed_task.due_date,
        )
        if success:
            page_url = updated_page.get("url") if isinstance(updated_page, dict) else None
            page_id = updated_page.get("id") if isinstance(updated_page, dict) else None
            if not page_url and page_id:
                clean_id = page_id.replace("-", "")
                page_url = f"https://www.notion.so/{clean_id}"

            reply_text = f"✅ Updated *{matched_title}*\n🔄 Status: *{target_status}*"
            if page_url:
                reply_text += f"\n🔗 {page_url}"
            if parsed_task.new_due_date or parsed_task.due_date:
                reply_text += f"\n📅 Due: {parsed_task.new_due_date or parsed_task.due_date}"

            if sender_id and page_id:
                conversation_memory.record_mutation(
                    sender_id=sender_id,
                    action_type="UPDATE_TASK",
                    target_title=matched_title,
                    affected_items=[{
                        "id": page_id,
                        "title": matched_title,
                        "url": page_url or "",
                        "type": "task",
                    }],
                    rollback_data={
                        "page_id": page_id,
                        "previous_status": "Not started" if target_status == "In progress" else "In progress",
                    },
                    summary=f"Updated status of {matched_title} to {target_status}",
                )
        else:
            reply_text = f"⚠️ Could not find an active task matching: *{title_query}*"

        if sender_id:
            conversation_memory.update_query_state(sender_id, last_module="TASKS", last_intent="UPDATE_TASK")
        return reply_text

    elif intent == "QUERY_TODAY":
        last_state = conversation_memory.get_last_query_state(sender_id) if sender_id else {}
        p_filter = parsed_task.priority_filter or (last_state.get("priority_filter") if parsed_task.is_followup else None)
        t_filter = parsed_task.tag_filter or (last_state.get("tag_filter") if parsed_task.is_followup else None)

        today_items = await run_in_threadpool(
            notion_client.get_today_tasks,
            priority=p_filter,
            tag=t_filter,
        )

        if sender_id:
            conversation_memory.update_query_state(
                sender_id,
                last_module="TASKS",
                last_intent="QUERY_TODAY",
                priority_filter=p_filter,
                tag_filter=t_filter,
                last_offset=0,
            )

        if today_items:
            if sender_id:
                today_dicts = [
                    item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else getattr(item, "__dict__", {}))
                    for item in today_items
                ]
                conversation_memory.set_last_query_results(sender_id, today_dicts)

            header = "📅 *Today's Tasks"
            if p_filter:
                header += f" ({p_filter} Priority)"
            header += f" ({len(today_items)}):*"

            tasks_lines = []
            for item in today_items:
                item_title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else str(item))
                item_due = getattr(item, "due_date", None) or (item.get("due_date") if isinstance(item, dict) else None)
                item_url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
                if not item_url:
                    p_id = getattr(item, "page_id", None) or (item.get("page_id") if isinstance(item, dict) else None)
                    if p_id:
                        item_url = f"https://www.notion.so/{str(p_id).replace('-', '')}"

                line = f"• *{item_title}*"
                if item_due:
                    line += f" (Due: {item_due})"
                if item_url:
                    line += f"\n  🔗 {item_url}"
                tasks_lines.append(line)
            tasks_str = "\n".join(tasks_lines)
            return f"{header}\n{tasks_str}"
        else:
            if p_filter:
                return f"🎉 No {p_filter} priority tasks due today!"
            return "🎉 No tasks due today!"

    elif intent == "QUERY_PENDING":
        last_state = conversation_memory.get_last_query_state(sender_id) if sender_id else {}
        text_clean = text.strip().lower()
        is_followup_word = text_clean in ("others", "others?", "more", "next", "what else", "what else?", "show more", "next page")

        if parsed_task.is_followup or is_followup_word:
            p_filter = parsed_task.priority_filter or last_state.get("priority_filter")
            t_filter = parsed_task.tag_filter or last_state.get("tag_filter")
            offset = last_state.get("last_offset", 0) + 5
        else:
            p_filter = parsed_task.priority_filter
            t_filter = parsed_task.tag_filter
            offset = parsed_task.offset or 0

        if p_filter or t_filter or offset > 0:
            pending_items = await run_in_threadpool(
                notion_client.get_pending,
                limit=5,
                offset=offset,
                priority=p_filter,
                tag=t_filter,
            )
        else:
            pending_items = await run_in_threadpool(
                notion_client.get_pending,
                limit=5,
            )

        if sender_id:
            conversation_memory.update_query_state(
                sender_id,
                last_module="TASKS",
                last_intent="QUERY_PENDING",
                priority_filter=p_filter,
                tag_filter=t_filter,
                last_offset=offset,
            )

        if pending_items:
            if sender_id:
                pending_dicts = [
                    item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else getattr(item, "__dict__", {}))
                    for item in pending_items
                ]
                conversation_memory.set_last_query_results(sender_id, pending_dicts)

            header = "📋 *Pending Tasks"
            if p_filter:
                header += f" ({p_filter} Priority)"
            if offset > 0:
                header += f" (Items {offset + 1}-{offset + len(pending_items)}):*"
            else:
                header += f" ({len(pending_items)}):*"

            tasks_lines = []
            for item in pending_items:
                item_title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else str(item))
                item_due = getattr(item, "due_date", None) or (item.get("due_date") if isinstance(item, dict) else None)
                item_url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
                if not item_url:
                    p_id = getattr(item, "page_id", None) or (item.get("page_id") if isinstance(item, dict) else None)
                    if p_id:
                        item_url = f"https://www.notion.so/{str(p_id).replace('-', '')}"

                line = f"• *{item_title}*"
                if item_due:
                    line += f" (Due: {item_due})"
                if item_url:
                    line += f"\n  🔗 {item_url}"
                tasks_lines.append(line)
            tasks_str = "\n".join(tasks_lines)
            return f"{header}\n{tasks_str}"
        else:
            if offset > 0:
                return f"🎉 No more {p_filter or ''} pending tasks found!".replace("  ", " ")
            elif p_filter:
                return f"🎉 No {p_filter} priority pending tasks found!"
            else:
                return "🎉 No pending tasks found!"

    else:
        log_content = parsed_task.log_content or text
        if sender_id:
            conversation_memory.update_query_state(sender_id, last_module="TASKS", last_intent="DAILY_LOG")
        return f"📝 *Daily Log Recorded:*\n{log_content}"


async def _handle_mind(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    mind_entry: MindEntry = parsed_result
    entry_title = mind_entry.title or (text[:50] if text else "Untitled Entry")
    entry_content = mind_entry.content or text
    entry_sub_intent = mind_entry.sub_intent or "DAILY_LOG"

    created_mind = await run_in_threadpool(
        notion_client.create_mind_entry,
        entry_type=entry_sub_intent,
        title=entry_title,
        content=entry_content,
        core_thesis=mind_entry.core_thesis,
        tags=mind_entry.tags,
    )
    mind_url = created_mind.get("url") if isinstance(created_mind, dict) else None
    if not mind_url and isinstance(created_mind, dict) and created_mind.get("id"):
        clean_id = created_mind["id"].replace("-", "")
        mind_url = f"https://www.notion.so/{clean_id}"

    if entry_sub_intent == "DRAFT_SUBSTACK":
        reply_text = f"✍️ *Substack Draft Created (Idea):* *{entry_title}*"
    elif entry_sub_intent == "RAMBLING":
        reply_text = f"💭 *Rambling Recorded:* *{entry_title}*"
    else:
        reply_text = f"📝 *Daily Log Recorded:* *{entry_title}*"

    if mind_url:
        reply_text += f"\n🔗 {mind_url}"
    if mind_entry.core_thesis:
        reply_text += f"\n💡 *Core Thesis:* {mind_entry.core_thesis}"
    elif mind_entry.summary:
        reply_text += f"\n📌 *Summary:* {mind_entry.summary}"
    if mind_entry.tags:
        reply_text += f"\n🏷 Tags: {', '.join(mind_entry.tags)}"

    if sender_id:
        conversation_memory.update_query_state(sender_id, last_module="MIND", last_intent=entry_sub_intent)
    return reply_text


async def _handle_leetcode(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    lc_req = parsed_result
    num_part = f"#{lc_req.problem_number} " if getattr(lc_req, "problem_number", None) else ""
    name = getattr(lc_req, "problem_name", None) or text
    reply_text = f"💻 *LeetCode Review Logged:* *{num_part}{name}*"
    if getattr(lc_req, "difficulty", None):
        reply_text += f"\n⚡ Difficulty: {lc_req.difficulty}"
    if getattr(lc_req, "patterns", None):
        reply_text += f"\n🧩 Patterns: {', '.join(lc_req.patterns)}"
    if getattr(lc_req, "status", None):
        reply_text += f"\n📊 Status: {lc_req.status}"
    if getattr(lc_req, "review_notes", None):
        reply_text += f"\n📝 Notes: {lc_req.review_notes}"
    return reply_text


async def _handle_search(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    search_analysis = parsed_result if isinstance(parsed_result, SearchQueryAnalysis) else SearchQueryAnalysis(query=text)
    search_fn = _get_app_main_fn("execute_second_brain_search_pipeline", execute_second_brain_search_pipeline)
    search_res = await run_in_threadpool(
        search_fn,
        query=search_analysis.query,
        search_analysis=search_analysis,
        notion_client=notion_client,
    )
    if sender_id:
        conversation_memory.update_query_state(sender_id, last_module="SEARCH", last_intent=search_analysis.search_type)
    return search_res.get("reply_text", "")


async def _handle_digest(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    digest_fn = _get_app_main_fn("execute_weekly_digest_pipeline", execute_weekly_digest_pipeline)
    digest_res = await run_in_threadpool(
        digest_fn,
        notion_client=notion_client,
    )
    if sender_id:
        conversation_memory.update_query_state(sender_id, last_module="DIGEST", last_intent="GENERATE_DIGEST")
    return digest_res.get("digest_text", "")


async def _handle_motion(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    motion_reply = await run_in_threadpool(
        persona_router.process_motion_request,
        text=text,
        sender_id=sender_id,
    )
    if sender_id:
        conversation_memory.update_query_state(sender_id, last_module="MOTION", last_intent="STRATEGIC_CONSULTATION")
    return motion_reply


async def _handle_compound(parsed_result: Any, text: str, notion_client: NotionAssistantClient, sender_id: Optional[str]) -> str:
    from app.compound_service import execute_compound_plan
    compound_fn = _get_app_main_fn("execute_compound_plan", execute_compound_plan)
    res = await run_in_threadpool(
        compound_fn,
        plan=parsed_result,
        notion_client=notion_client,
        sender_id=sender_id,
    )
    return res.get("reply_text", "Compound actions complete!")


_ACTION_HANDLERS = {
    "MEMORY_CONTROL": _handle_memory_control,
    "BATCH_TASK_ACTION": _handle_batch_task_action,
    "TASK_ACTION": _handle_task_action,
    "ROLLBACK": _handle_rollback,
    "DOCUMENT_APPEND": _handle_document_append,
    "TASKS": _handle_tasks,
    "MIND": _handle_mind,
    "LEETCODE": _handle_leetcode,
    "SEARCH": _handle_search,
    "DIGEST": _handle_digest,
    "MOTION": _handle_motion,
    "COMPOUND": _handle_compound,
}


from app.module_handlers import handler_registry


async def execute_module_action(
    module: str,
    parsed_result: Any,
    text: str,
    notion_client: NotionAssistantClient,
    sender_id: Optional[str] = None,
) -> str:
    """Execute module action polymorphically via handler_registry."""
    return await handler_registry.execute(
        module=module,
        parsed_result=parsed_result,
        text=text,
        notion_client=notion_client,
        sender_id=sender_id,
    )


async def process_incoming_text_message(
    text: str,
    sender_id: str,
    background_tasks: BackgroundTasks,
    notion_client: NotionAssistantClient,
    to_phone: Optional[str] = None,
    chat_id: Optional[str] = None,
    whatsapp_client: Optional[Any] = None,
    telegram_client: Optional[Any] = None,
) -> Optional[str]:
    """Unified text message processing pipeline across WhatsApp and Telegram.

    Returns reply text if synchronous reply is appropriate.
    """
    import sys
    app_main = sys.modules.get("app.main")
    get_fn = lambda name, fallback: getattr(app_main, name, fallback) if app_main else fallback

    # 1. Check for 1-Tap URL Ingestion
    if is_url_dominant_message(text):
        urls = extract_urls(text)
        if urls:
            primary_url = urls[0]
            send_notification(
                "Digesting link with Gemini...",
                to_phone=to_phone,
                chat_id=chat_id,
                whatsapp_client=whatsapp_client,
                telegram_client=telegram_client,
            )
            conversation_memory.add_assistant_message(sender_id, "Digesting link with Gemini...", module="LEARNING")
            url_bg = get_fn("execute_url_digest_background_pipeline", execute_url_digest_background_pipeline)
            background_tasks.add_task(
                url_bg,
                url=primary_url,
                user_comment=text,
                to_phone=to_phone,
                chat_id=chat_id,
            )
            return None

    # 2. Check for interactive disambiguation menu reply
    pending_menu_reply = await handle_pending_menu_reply(sender_id, text, notion_client)
    if pending_menu_reply:
        send_notification(
            pending_menu_reply,
            to_phone=to_phone,
            chat_id=chat_id,
            whatsapp_client=whatsapp_client,
            telegram_client=telegram_client,
        )
        conversation_memory.add_assistant_message(sender_id, pending_menu_reply, module="TASK_ACTION")
        return None

    # 3. Check for Motion Persona Routing (@motion or motion:)
    if persona_router.is_motion_invoked(text):
        conversation_memory.add_user_message(sender_id, text)
        reply_text = await run_in_threadpool(
            persona_router.process_motion_request,
            text=text,
            sender_id=sender_id,
        )
        send_notification(
            reply_text,
            to_phone=to_phone,
            chat_id=chat_id,
            whatsapp_client=whatsapp_client,
            telegram_client=telegram_client,
        )
        conversation_memory.add_assistant_message(sender_id, reply_text, module="MOTION")
        return None

    # 3.5 Check for Instant Deterministic Fast-Paths (0ms LLM Latency)
    clean_lower = text.strip().lower().rstrip(".!?")
    if clean_lower in ("today", "todays tasks", "today's tasks", "what's due today", "what is due today", "what's on my plate", "what is on my plate", "today tasks"):
        conversation_memory.add_user_message(sender_id, text)
        fast_result = TaskAnalysis(intent="QUERY_TODAY")
        reply_text = await execute_module_action("TASKS", fast_result, text, notion_client, sender_id=sender_id)
        if reply_text:
            send_notification(reply_text, to_phone=to_phone, chat_id=chat_id, whatsapp_client=whatsapp_client, telegram_client=telegram_client)
            conversation_memory.add_assistant_message(sender_id, reply_text, module="TASKS")
        return reply_text

    if clean_lower in ("undo", "undo that", "revert", "revert that", "cancel last action", "undo last task"):
        conversation_memory.add_user_message(sender_id, text)
        fast_result = RollbackAnalysis(command="ROLLBACK_LAST")
        reply_text = await execute_module_action("ROLLBACK", fast_result, text, notion_client, sender_id=sender_id)
        if reply_text:
            send_notification(reply_text, to_phone=to_phone, chat_id=chat_id, whatsapp_client=whatsapp_client, telegram_client=telegram_client)
            conversation_memory.add_assistant_message(sender_id, reply_text, module="ROLLBACK")
        return reply_text

    # 4. Context Enrichment
    conversation_memory.add_user_message(sender_id, text)
    context = conversation_memory.format_context_prompt(sender_id, max_turns=4)
    graph_ctx = graph_memory.format_graph_context_for_prompt(text)
    if graph_ctx:
        context = f"{graph_ctx}\n\n{context}" if context else graph_ctx

    # 5. Two-Stage Classification & Parsing
    analyze_fn = get_fn("analyze_user_text_two_stage", analyze_user_text_two_stage)
    module, parsed_result = await run_in_threadpool(analyze_fn, text, context=context)

    # Background entity extraction for knowledge modules
    if module in ("MIND", "LEARNING", "LEETCODE", "DOCUMENT_APPEND"):
        background_tasks.add_task(graph_memory.extract_and_index_entities_from_text, text, module)

    # 6. Asynchronous Background Pipelines
    if module == "LEARNING":
        send_notification(
            "Building your study plan...",
            to_phone=to_phone,
            chat_id=chat_id,
            whatsapp_client=whatsapp_client,
            telegram_client=telegram_client,
        )
        conversation_memory.add_assistant_message(sender_id, "Building your study plan...", module="LEARNING")
        learn_bg = get_fn("execute_learning_background_pipeline", execute_learning_background_pipeline)
        background_tasks.add_task(
            learn_bg,
            parsed_result,
            to_phone=to_phone,
            chat_id=chat_id,
        )
        return None

    if module == "LEETCODE":
        send_notification(
            "Pulling your latest solution...",
            to_phone=to_phone,
            chat_id=chat_id,
            whatsapp_client=whatsapp_client,
            telegram_client=telegram_client,
        )
        conversation_memory.add_assistant_message(sender_id, "Pulling your latest solution...", module="LEETCODE")
        lc_bg = get_fn("execute_leetcode_background_pipeline", execute_leetcode_background_pipeline)
        background_tasks.add_task(
            lc_bg,
            parsed_result,
            to_phone=to_phone,
            chat_id=chat_id,
        )
        return None

    if module == "DIGEST":
        send_notification(
            "Compiling your weekly velocity digest...",
            to_phone=to_phone,
            chat_id=chat_id,
            whatsapp_client=whatsapp_client,
            telegram_client=telegram_client,
        )
        conversation_memory.add_assistant_message(sender_id, "Compiling your weekly velocity digest...", module="DIGEST")
        digest_bg = get_fn("execute_weekly_digest_pipeline", execute_weekly_digest_pipeline)
        background_tasks.add_task(
            digest_bg,
            to_phone=to_phone,
            chat_id=chat_id,
        )
        return None

    if module == "SEARCH":
        search_analysis = parsed_result if isinstance(parsed_result, SearchQueryAnalysis) else SearchQueryAnalysis(query=text)
        send_notification(
            "Searching your second brain...",
            to_phone=to_phone,
            chat_id=chat_id,
            whatsapp_client=whatsapp_client,
            telegram_client=telegram_client,
        )
        conversation_memory.add_assistant_message(sender_id, "Searching your second brain...", module="SEARCH")
        search_bg = get_fn("execute_second_brain_search_pipeline", execute_second_brain_search_pipeline)
        background_tasks.add_task(
            search_bg,
            query=search_analysis.query,
            search_analysis=search_analysis,
            to_phone=to_phone,
            chat_id=chat_id,
        )
        return None

    # 7. Synchronous Module Actions
    try:
        reply_text = await execute_module_action(module, parsed_result, text, notion_client, sender_id=sender_id)
        if reply_text:
            send_notification(
                reply_text,
                to_phone=to_phone,
                chat_id=chat_id,
                whatsapp_client=whatsapp_client,
                telegram_client=telegram_client,
            )
            conversation_memory.add_assistant_message(sender_id, reply_text, module=module)
        return reply_text
    except Exception as exc:
        logger.error("Failed to execute module action for module=%s: %s", module, exc)
        raise

