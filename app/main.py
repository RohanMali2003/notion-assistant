import logging
import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, Header, HTTPException
from starlette.concurrency import run_in_threadpool
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

#check imports, it seems there's some conflict.

from app.config import settings
from app.notion_client import NotionAssistantClient
from app.schemas import TaskAnalysis, TelegramWebhookUpdate, WebhookResponse
from app.telegram_client import TelegramAssistantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

app = FastAPI(
    title="Notion Assistant Webhook API",
    description="FastAPI service handling Notion and Telegram webhook events.",
    version="1.0.0",
)


def analyze_user_text_with_gemini(text: str) -> TaskAnalysis:
    """Call Gemini to classify user text intent and extract task structure.

    Wrap call in try/except. On any parsing or API failure, fall back to DAILY_LOG.
    """
    try:
        if genai is None or types is None:
            raise RuntimeError("google-genai library is not installed or available")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
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
            raise ValueError("Empty response from Gemini model")
    except Exception as exc:
        logger.warning("Gemini parsing failed (%s). Falling back to intent=DAILY_LOG.", exc)
        return TaskAnalysis(
            intent="DAILY_LOG",
            title=text[:50] if text else "Daily Log",
            log_content=text,
        )


@app.get("/health", response_model=WebhookResponse)
def health_check():
    """Health check endpoint for uptime pings and Render health checks."""
    return WebhookResponse(status="ok")


@app.post("/webhook")
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

    # 3 & 4. Call Gemini via run_in_threadpool
    parsed_task: TaskAnalysis = await run_in_threadpool(analyze_user_text_with_gemini, text)
    intent = parsed_task.intent

    notion_client = NotionAssistantClient()
    telegram_client = TelegramAssistantClient()

    # 5. Route intent to appropriate action
    try:
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

        # 8. Log request completion
        logger.info(
            "Webhook request processed successfully: chat_id=%s, parsed_intent=%s, status=success",
            chat_id,
            intent,
        )
        return {"status": "ok"}

    except Exception as exc:
        # 8. Log failure
        logger.error(
            "Webhook request failed: chat_id=%s, parsed_intent=%s, status=failure, error=%s",
            chat_id,
            intent,
            exc,
        )
        return {"status": "error", "message": str(exc)}
