import logging
import os
from typing import Any, Dict, Optional, Tuple, Union
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from app.config import settings
from app.notion_client import NotionAssistantClient
from app.schemas import (
    LearningRequest,
    LeetcodeReviewRequest,
    MindEntry,
    ModuleClassification,
    TaskAnalysis,
    TelegramWebhookUpdate,
    WebhookResponse,
)
from app.telegram_client import TelegramAssistantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

app = FastAPI(
    title="Notion Assistant Webhook API",
    description="FastAPI service handling Notion and Telegram webhook events.",
    version="1.0.0",
)


def get_gemini_client():
    """Create and return a google-genai Client instance with configured API keys."""
    if genai is None or types is None:
        raise RuntimeError("google-genai library is not installed or available")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def get_gemini_model() -> str:
    """Return the Gemini model identifier to use (default: gemini-3.5-flash-lite)."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


# --- Stage 1: Module Classification ---

STAGE1_SYSTEM_INSTRUCTION = (
    "You are a lightweight intent routing classifier. Classify the user's message into exactly one MODULE:\n"
    "- TASKS: Creating or querying tasks, managing to-do items, updating task status, checking today's or pending tasks.\n"
    "- MIND: Substack drafts, journaling, brain dumps, rambling, daily reflections/logs, personal thoughts.\n"
    "- LEARNING: New study topic requests, learning roadmaps, syllabus inquiries, concept exploration.\n"
    "- LEETCODE: LeetCode problem review requests, algorithm practice notes, problem solution tracking.\n"
    "Pass the raw user message into the raw_text field."
)


def classify_module_stage1(text: str) -> ModuleClassification:
    """Stage 1: Classify user message into target functional module using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=STAGE1_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ModuleClassification,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, ModuleClassification):
                result = response.parsed
            else:
                result = ModuleClassification.model_validate(response.parsed)
            if not result.raw_text:
                result.raw_text = text
            return result
        elif response.text:
            result = ModuleClassification.model_validate_json(response.text)
            if not result.raw_text:
                result.raw_text = text
            return result
        else:
            raise ValueError("Empty response from Gemini Stage 1 classification")
    except Exception as exc:
        logger.warning("Stage 1 classification failed (%s). Falling back to module=TASKS.", exc)
        return ModuleClassification(module="TASKS", raw_text=text)


# --- Stage 2: Module-Specific Parsers ---

STAGE2_TASKS_INSTRUCTION = (
    "Extract task management details from the user message:\n"
    "- CREATE_TASK: Adding a new task, with optional priority (High/Medium/Low), tag, due_date (YYYY-MM-DD), description.\n"
    "- UPDATE_TASK: Updating status of an existing task (In progress/Done/Not started) or due date.\n"
    "- QUERY_TODAY: Querying tasks due today.\n"
    "- QUERY_PENDING: Querying pending or upcoming tasks.\n"
    "- DAILY_LOG: Recording daily notes or log entry."
)


def parse_tasks_stage2(text: str) -> TaskAnalysis:
    """Stage 2: Parse TASKS module intent and task structure using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_TASKS_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=TaskAnalysis,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, TaskAnalysis):
                return response.parsed
            return TaskAnalysis.model_validate(response.parsed)
        elif response.text:
            return TaskAnalysis.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 TASKS parser")
    except Exception as exc:
        logger.warning("Stage 2 TASKS parsing failed (%s). Falling back to intent=DAILY_LOG.", exc)
        return TaskAnalysis(
            intent="DAILY_LOG",
            title=text[:50] if text else "Daily Log",
            log_content=text,
        )


STAGE2_MIND_INSTRUCTION = (
    "Extract mind/thought entry details from the user message:\n"
    "- entry_type: DRAFT_SUBSTACK (article drafts, newsletter ideas, essays), RAMBLING (stream of consciousness, quick thoughts, brain dumps), or DAILY_LOG (daily reflections, logs, journaling).\n"
    "- title: A concise, descriptive title or headline for the database entry.\n"
    "- core_thesis: Exactly one sentence summarizing the core thesis, key insight, or main premise.\n"
    "- content: The complete text, thoughts, or drafted body.\n"
    "- summary: Optional brief summary or key takeaways.\n"
    "- tags: Relevant topics or tags."
)


def parse_mind_stage2(text: str) -> MindEntry:
    """Stage 2: Parse MIND module entry (substack draft, rambling, daily log) using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_MIND_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=MindEntry,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, MindEntry):
                result = response.parsed
            else:
                result = MindEntry.model_validate(response.parsed)
            if not result.content:
                result.content = text
            if not result.title:
                result.title = text[:50] if text else "Untitled Entry"
            if not result.core_thesis:
                first_sent = (result.content or text).strip().split(".")[0].strip()
                result.core_thesis = (first_sent + ".") if first_sent else "Daily reflection."
            return result
        elif response.text:
            result = MindEntry.model_validate_json(response.text)
            if not result.content:
                result.content = text
            if not result.title:
                result.title = text[:50] if text else "Untitled Entry"
            if not result.core_thesis:
                first_sent = (result.content or text).strip().split(".")[0].strip()
                result.core_thesis = (first_sent + ".") if first_sent else "Daily reflection."
            return result
        else:
            raise ValueError("Empty response from Gemini Stage 2 MIND parser")
    except Exception as exc:
        logger.warning("Stage 2 MIND parsing failed (%s). Falling back to default MindEntry.", exc)
        first_sent = text.strip().split(".")[0].strip() if text else "Daily reflection"
        return MindEntry(
            entry_type="DAILY_LOG",
            title=text[:50] if text else "Daily Reflection",
            core_thesis=(first_sent + ".") if first_sent and not first_sent.endswith(".") else first_sent,
            content=text,
        )


STAGE2_LEARNING_INSTRUCTION = (
    "Extract learning/study request details from the user message:\n"
    "- topic: The subject or topic to study (e.g. Distributed Systems, Rust Borrow Checker).\n"
    "- category: Broader domain (e.g. Computer Science, Math, Systems Engineering).\n"
    "- goal: Specific learning objective, subtopics, or questions.\n"
    "- proficiency_level: Beginner, Intermediate, or Advanced.\n"
    "- resources_requested: Any specific resources, books, papers, tutorials, or roadmaps asked for."
)


def parse_learning_stage2(text: str) -> LearningRequest:
    """Stage 2: Parse LEARNING module request (new study topic) using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_LEARNING_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=LearningRequest,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, LearningRequest):
                return response.parsed
            return LearningRequest.model_validate(response.parsed)
        elif response.text:
            return LearningRequest.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 LEARNING parser")
    except Exception as exc:
        logger.warning("Stage 2 LEARNING parsing failed (%s). Falling back to default LearningRequest.", exc)
        return LearningRequest(
            topic=text[:50] if text else "New Study Topic",
            goal=text,
        )


STAGE2_LEETCODE_INSTRUCTION = (
    "Extract LeetCode review request details from the user message:\n"
    "- problem_name: Problem title or name (e.g. Two Sum, Trapping Rain Water).\n"
    "- problem_number: Problem number integer if mentioned.\n"
    "- difficulty: Easy, Medium, or Hard.\n"
    "- patterns: Algorithmic techniques used (e.g. Two Pointers, Dynamic Programming, Monotonic Stack).\n"
    "- review_notes: Key insights, complexities, pitfalls, or review notes.\n"
    "- status: Solved, Review Needed, Failed, or Mastered."
)


def parse_leetcode_stage2(text: str) -> LeetcodeReviewRequest:
    """Stage 2: Parse LEETCODE module review request using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_LEETCODE_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=LeetcodeReviewRequest,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, LeetcodeReviewRequest):
                return response.parsed
            return LeetcodeReviewRequest.model_validate(response.parsed)
        elif response.text:
            return LeetcodeReviewRequest.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 LEETCODE parser")
    except Exception as exc:
        logger.warning("Stage 2 LEETCODE parsing failed (%s). Falling back to default LeetcodeReviewRequest.", exc)
        return LeetcodeReviewRequest(
            problem_name=text[:50] if text else "LeetCode Review",
            review_notes=text,
        )


# --- Two-Stage Pipeline Orchestration ---

def analyze_user_text_two_stage(
    text: str,
) -> Tuple[str, Union[TaskAnalysis, MindEntry, LearningRequest, LeetcodeReviewRequest]]:
    """Execute two-stage Gemini pipeline: Stage 1 classification -> Stage 2 module-specific parsing."""
    stage1_res = classify_module_stage1(text)
    module = stage1_res.module
    raw_text = stage1_res.raw_text or text

    if module == "TASKS":
        parsed = parse_tasks_stage2(raw_text)
    elif module == "MIND":
        parsed = parse_mind_stage2(raw_text)
    elif module == "LEARNING":
        parsed = parse_learning_stage2(raw_text)
    elif module == "LEETCODE":
        parsed = parse_leetcode_stage2(raw_text)
    else:
        parsed = parse_tasks_stage2(raw_text)

    return module, parsed


def analyze_user_text_with_gemini(text: str) -> TaskAnalysis:
    """Legacy helper: calls Stage 2 TASKS parsing directly for backward compatibility."""
    return parse_tasks_stage2(text)


@app.get("/health", response_model=WebhookResponse)
def health_check():
    """Health check endpoint for uptime pings and Render health checks."""
    return WebhookResponse(status="ok")


@app.get("/webhook")
def whatsapp_webhook_handshake(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    """WhatsApp Cloud API webhook handshake verification endpoint."""
    expected_verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN") or getattr(settings, "WHATSAPP_VERIFY_TOKEN", "")
    if hub_mode == "subscribe" and hub_verify_token and hub_verify_token == expected_verify_token:
        logger.info("WhatsApp webhook handshake verified successfully.")
        return PlainTextResponse(content=hub_challenge or "", status_code=200)

    logger.warning(
        "WhatsApp webhook handshake failed: mode=%s, token match=%s",
        hub_mode,
        bool(hub_verify_token and hub_verify_token == expected_verify_token),
    )
    raise HTTPException(status_code=403, detail="Verification token mismatch")


@app.post("/webhook")
async def whatsapp_webhook():
    """WhatsApp Cloud API webhook event endpoint returning EVENT_RECEIVED immediately."""
    return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)


@app.post("/webhook/telegram")
async def telegram_webhook(
    update: TelegramWebhookUpdate,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None, alias="X-Telegram-Bot-Api-Secret-Token")
):
    """Receive and process incoming Telegram webhooks."""
    # 1. Verify secret token header against config
    secret_in_config = (
        os.getenv("TELEGRAM_WEBHOOK_SECRET")
        or os.getenv("WEBHOOK_SECRET")
        or getattr(settings, "TELEGRAM_WEBHOOK_SECRET", "")
        or getattr(settings, "WEBHOOK_SECRET", "")
    )
    if secret_in_config and x_telegram_bot_api_secret_token != secret_in_config:
        logger.warning("Unauthorized webhook request: secret token mismatch.")
        raise HTTPException(status_code=401, detail="Invalid secret token")

    # 2. Parse Telegram update. Return {"status": "ignored"} with 200 if no message.text
    if not update.message or not update.message.get("text"):
        logger.info("Webhook request ignored: no message text present.")
        return {"status": "ignored"}

    text = update.message.get("text")
    chat_id = update.message.get("chat", {}).get("id")

    # 3 & 4. Two-Stage Gemini Classification and Parsing
    module, parsed_result = await run_in_threadpool(analyze_user_text_two_stage, text)

    notion_client = NotionAssistantClient()
    telegram_client = TelegramAssistantClient()

    # 5. Route module and parsed result to appropriate action
    try:
        if module == "TASKS":
            parsed_task: TaskAnalysis = parsed_result
            intent = parsed_task.intent

            if intent == "CREATE_TASK":
                await run_in_threadpool(
                    notion_client.create_task,
                    title=parsed_task.title or text,
                    priority=parsed_task.priority,
                    tag=parsed_task.tag,
                    due_date=parsed_task.due_date,
                    description=parsed_task.description,
                )
                reply_text = f"✅ Task created: *{parsed_task.title or text}*"
                if parsed_task.due_date:
                    reply_text += f"\n📅 Due: {parsed_task.due_date}"
                if parsed_task.priority:
                    reply_text += f"\n⚡ Priority: {parsed_task.priority}"
                if parsed_task.tag:
                    reply_text += f"\n🏷 Tag: {parsed_task.tag}"

            elif intent == "UPDATE_TASK":
                target_status = parsed_task.target_status or "In progress"
                title_query = parsed_task.title or text
                success, matched_title, _ = await run_in_threadpool(
                    notion_client.update_task_status,
                    title_query=title_query,
                    status_name=target_status,
                    new_due_date=parsed_task.new_due_date or parsed_task.due_date,
                )
                if success:
                    reply_text = f"✅ Updated *{matched_title}*\n🔄 Status: *{target_status}*"
                    if parsed_task.new_due_date or parsed_task.due_date:
                        reply_text += f"\n📅 Due: {parsed_task.new_due_date or parsed_task.due_date}"
                else:
                    reply_text = f"⚠️ Could not find an active task matching: *{title_query}*"

            elif intent == "QUERY_TODAY":
                today_items = await run_in_threadpool(notion_client.get_today_tasks)
                if today_items:
                    tasks_str = "\n".join(
                        [f"• {item.title}" + (f" (Due: {item.due_date})" if item.due_date else "") for item in today_items]
                    )
                    reply_text = f"📅 *Today's Tasks ({len(today_items)}):*\n{tasks_str}"
                else:
                    reply_text = "🎉 No tasks due today!"

            elif intent == "QUERY_PENDING":
                pending_items = await run_in_threadpool(notion_client.get_pending, limit=5)
                if pending_items:
                    tasks_str = "\n".join(
                        [f"• {item.title}" + (f" (Due: {item.due_date})" if item.due_date else "") for item in pending_items]
                    )
                    reply_text = f"📋 *Pending Tasks ({len(pending_items)}):*\n{tasks_str}"
                else:
                    reply_text = "🎉 No pending tasks found!"

            else:  # DAILY_LOG or fallback
                log_content = parsed_task.log_content or text
                reply_text = f"📝 *Daily Log Recorded:*\n{log_content}"

        elif module == "MIND":
            mind_entry: MindEntry = parsed_result
            entry_title = mind_entry.title or (text[:50] if text else "Untitled Entry")
            entry_content = mind_entry.content or text
            entry_sub_intent = mind_entry.sub_intent or "DAILY_LOG"

            await run_in_threadpool(
                notion_client.create_mind_entry,
                entry_type=entry_sub_intent,
                title=entry_title,
                content=entry_content,
                core_thesis=mind_entry.core_thesis,
                tags=mind_entry.tags,
            )

            if entry_sub_intent == "DRAFT_SUBSTACK":
                reply_text = f"✍️ *Substack Draft Created (Idea):* *{entry_title}*"
            elif entry_sub_intent == "RAMBLING":
                reply_text = f"💭 *Rambling Recorded:* *{entry_title}*"
            else:
                reply_text = f"📝 *Daily Log Recorded:* *{entry_title}*"

            if mind_entry.core_thesis:
                reply_text += f"\n💡 *Core Thesis:* {mind_entry.core_thesis}"
            elif mind_entry.summary:
                reply_text += f"\n📌 *Summary:* {mind_entry.summary}"
            if mind_entry.tags:
                reply_text += f"\n🏷 Tags: {', '.join(mind_entry.tags)}"

        elif module == "LEARNING":
            learning_req: LearningRequest = parsed_result
            reply_text = f"📚 *Learning Topic Added:* *{learning_req.topic or text}*"
            if learning_req.category:
                reply_text += f"\n📂 Category: {learning_req.category}"
            if learning_req.proficiency_level:
                reply_text += f"\n🎯 Level: {learning_req.proficiency_level}"
            if learning_req.goal:
                reply_text += f"\n🎯 Goal: {learning_req.goal}"
            if learning_req.resources_requested:
                reply_text += f"\n📖 Resources: {learning_req.resources_requested}"

        elif module == "LEETCODE":
            lc_req: LeetcodeReviewRequest = parsed_result
            num_part = f"#{lc_req.problem_number} " if lc_req.problem_number else ""
            reply_text = f"💻 *LeetCode Review Logged:* *{num_part}{lc_req.problem_name or text}*"
            if lc_req.difficulty:
                reply_text += f"\n⚡ Difficulty: {lc_req.difficulty}"
            if lc_req.patterns:
                reply_text += f"\n🧩 Patterns: {', '.join(lc_req.patterns)}"
            if lc_req.status:
                reply_text += f"\n📊 Status: {lc_req.status}"
            if lc_req.review_notes:
                reply_text += f"\n📝 Notes: {lc_req.review_notes}"

        else:
            reply_text = f"📝 *Recorded:*\n{text}"

        # 6. POST reply back to Telegram sendMessage endpoint via threadpool
        if chat_id:
            try:
                await run_in_threadpool(
                    telegram_client.send_message,
                    text=reply_text,
                    chat_id=str(chat_id)
                )
            except Exception as tg_err:
                logger.error("Failed to send message to Telegram chat_id=%s: %s", chat_id, tg_err)

        # 7. Log request completion
        logger.info(
            "Webhook request processed successfully: chat_id=%s, module=%s, status=success",
            chat_id,
            module,
        )
        return {"status": "ok"}

    except Exception as exc:
        # Log failure
        logger.error(
            "Webhook request failed: chat_id=%s, module=%s, status=failure, error=%s",
            chat_id,
            module,
            exc,
        )
        return {"status": "error", "message": str(exc)}

