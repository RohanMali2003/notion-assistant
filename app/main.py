"""FastAPI Application Entrypoint for Notion Assistant & Ocean Motion.

Handles webhook verification and ingress dispatching for WhatsApp, Telegram,
and registers the Motion Strategic Mentorship API router.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

# Re-exports for complete backward compatibility with existing tests & external imports
try:
    from google import genai
    from google.genai import types
except (ImportError, AttributeError):
    class _MockGenaiModule:
        class Client:
            pass
    class _MockTypesModule:
        class GenerateContentConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
    genai = _MockGenaiModule()
    types = _MockTypesModule()

from app.ai import get_gemini_client, get_gemini_model
from app.config import settings
from app.dispatcher import (
    execute_module_action,
    handle_pending_menu_reply,
    process_incoming_text_message,
)
from app.graph_memory_service import graph_memory
from app.learning_service import execute_learning_background_pipeline
from app.leetcode_service import execute_leetcode_background_pipeline
from app.media_service import execute_media_pipeline
from app.memory import conversation_memory
from app.motion import motion_api_router, persona_router
from app.notion_client import NotionAssistantClient
from app.parsers import (
    analyze_user_text_two_stage,
    analyze_user_text_with_gemini,
    classify_module_stage1,
    parse_batch_task_action_stage2,
    parse_document_append_stage2,
    parse_learning_stage2,
    parse_leetcode_stage2,
    parse_memory_control_stage2,
    parse_mind_stage2,
    parse_rollback_stage2,
    parse_search_stage2,
    parse_task_action_stage2,
    parse_tasks_stage2,
)
from app.rollback_service import execute_rollback
from app.schemas import (
    BatchTaskActionAnalysis,
    DocumentAppendAnalysis,
    LearningRequest,
    LeetcodeReviewRequest,
    MemoryGovernanceAnalysis,
    MindEntry,
    ModuleClassification,
    RollbackAnalysis,
    SearchQueryAnalysis,
    TaskActionAnalysis,
    TaskAnalysis,
    TelegramWebhookUpdate,
    WebhookResponse,
    WorkspaceEntryItem,
)
from app.search_service import execute_second_brain_search_pipeline
from app.task_action_service import execute_batch_task_action, execute_task_action
from app.telegram_client import TelegramAssistantClient
from app.url_digest_service import execute_url_digest_background_pipeline
from app.weekly_digest_service import execute_weekly_digest_pipeline
from app.whatsapp_client import WhatsAppAssistantClient
from app.workspace_service import add_entries_to_workspace_target

# Alias for backwards compatibility
_handle_module_action = execute_module_action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

app = FastAPI(
    title="Notion Assistant Webhook API",
    description="FastAPI service handling Notion, WhatsApp, Telegram, and Motion API events.",
    version="1.0.0",
)

# Mount Ocean Motion Subsystem API Router
app.include_router(motion_api_router)


# ==========================================
# --- Health & Webhook Handshake Routes ---
# ==========================================

@app.get("/health", response_model=WebhookResponse)
def health_check():
    """Health check endpoint for uptime monitors and container probes."""
    return WebhookResponse(status="ok")


@app.get("/webhook")
def whatsapp_webhook_handshake(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    """WhatsApp Cloud API webhook verification handshake."""
    expected_verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN") or getattr(settings, "WHATSAPP_VERIFY_TOKEN", "")
    if hub_mode == "subscribe" and hub_verify_token and hub_verify_token == expected_verify_token:
        logger.info("WhatsApp webhook handshake verified successfully.")
        return PlainTextResponse(content=hub_challenge or "", status_code=200)

    logger.warning("WhatsApp webhook handshake verification failed.")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


# ==========================================
# --- Payload Extraction Helpers ---
# ==========================================

def _extract_whatsapp_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sender, message type, and content from WhatsApp Cloud API webhook payload."""
    try:
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                for msg in change.get("value", {}).get("messages", []):
                    sender = str(msg.get("from")) if msg.get("from") else None
                    msg_type = msg.get("type", "text")

                    if msg_type == "text" or "text" in msg:
                        text = msg.get("text", {}).get("body", "") if isinstance(msg.get("text"), dict) else str(msg.get("text", ""))
                        return {"sender": sender, "type": "text", "text": text}

                    if msg_type == "image":
                        img = msg.get("image", {})
                        caption = img.get("caption", "")
                        return {
                            "sender": sender,
                            "type": "image",
                            "media_id": img.get("id"),
                            "caption": caption,
                            "mime_type": img.get("mime_type", "image/jpeg"),
                            "text": caption,
                        }
    except Exception as exc:
        logger.debug("Error extracting WhatsApp event: %s", exc)
    return {"sender": None, "type": None, "text": None}


def _extract_whatsapp_message(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract sender phone and text from WhatsApp webhook payload."""
    ev = _extract_whatsapp_event(payload)
    return ev.get("sender"), ev.get("text")



# ==========================================
# --- Ingress Webhook Handlers ---
# ==========================================

@app.post("/webhook")
async def whatsapp_webhook(
    background_tasks: BackgroundTasks,
    request: Request = None,
):
    """WhatsApp Cloud API incoming message webhook."""
    payload: Dict[str, Any] = {}
    if request is not None:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

    event = _extract_whatsapp_event(payload)
    sender_phone = event.get("sender")
    msg_type = event.get("type")

    if not sender_phone or not msg_type:
        logger.info("WhatsApp webhook received non-message or empty event.")
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    whatsapp_client = WhatsAppAssistantClient()
    notion_client = NotionAssistantClient()

    # 1. Image / Media Handling
    if msg_type == "image":
        media_id = event.get("media_id")
        caption = event.get("caption", "")
        if media_id:
            try:
                await run_in_threadpool(
                    whatsapp_client.send_message,
                    to=sender_phone,
                    text="Analyzing your image with Gemini Vision...",
                )
                conversation_memory.add_assistant_message(sender_phone, "Analyzing your image with Gemini Vision...", module="MIND")
                image_bytes, mime_type = await run_in_threadpool(
                    whatsapp_client.download_media_bytes,
                    media_id=media_id,
                )
                background_tasks.add_task(
                    execute_media_pipeline,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    caption=caption,
                    to_phone=sender_phone,
                )
            except Exception as media_err:
                logger.error("Failed to process WhatsApp media message: %s", media_err)
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    # 2. Text Message Handling
    text = event.get("text", "")
    if not text:
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    try:
        await process_incoming_text_message(
            text=text,
            sender_id=sender_phone,
            background_tasks=background_tasks,
            notion_client=notion_client,
            to_phone=sender_phone,
            whatsapp_client=whatsapp_client,
        )
    except Exception as exc:
        logger.error("Failed to process WhatsApp message: %s", exc)

    return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)


@app.post("/webhook/telegram")
async def telegram_webhook(
    update: TelegramWebhookUpdate,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None, alias="X-Telegram-Bot-Api-Secret-Token"),
):
    """Telegram Bot API incoming update webhook."""
    secret_in_config = (
        os.getenv("TELEGRAM_WEBHOOK_SECRET")
        or os.getenv("WEBHOOK_SECRET")
        or getattr(settings, "TELEGRAM_WEBHOOK_SECRET", "")
        or getattr(settings, "WEBHOOK_SECRET", "")
    )
    if secret_in_config and x_telegram_bot_api_secret_token != secret_in_config:
        logger.warning("Unauthorized Telegram webhook request: secret token mismatch.")
        raise HTTPException(status_code=401, detail="Invalid secret token")

    telegram_client = TelegramAssistantClient()
    notion_client = NotionAssistantClient()

    msg = update.message or {}
    chat_id = msg.get("chat", {}).get("id")
    sender_id = str(chat_id) if chat_id else "unknown_tg"

    # 1. Photo / Document Handling
    photos = msg.get("photo", [])
    document = msg.get("document", {})
    caption = msg.get("caption", "")

    file_id = None
    if photos and isinstance(photos, list):
        file_id = photos[-1].get("file_id")
    elif document and isinstance(document, dict) and document.get("mime_type", "").startswith("image/"):
        file_id = document.get("file_id")

    if file_id and chat_id:
        try:
            await run_in_threadpool(
                telegram_client.send_message,
                text="Analyzing your image with Gemini Vision...",
                chat_id=str(chat_id),
            )
            conversation_memory.add_assistant_message(sender_id, "Analyzing your image with Gemini Vision...", module="MIND")
            image_bytes, mime_type = await run_in_threadpool(
                telegram_client.download_file_bytes,
                file_id=file_id,
            )
            background_tasks.add_task(
                execute_media_pipeline,
                image_bytes=image_bytes,
                mime_type=mime_type,
                caption=caption,
                chat_id=str(chat_id),
            )
        except Exception as media_err:
            logger.error("Failed to process Telegram media message: %s", media_err)
        return {"status": "ok"}

    # 2. Text Message Handling
    text = msg.get("text")
    if not text:
        logger.info("Telegram webhook ignored: no message text present.")
        return {"status": "ignored"}

    try:
        await process_incoming_text_message(
            text=text,
            sender_id=sender_id,
            background_tasks=background_tasks,
            notion_client=notion_client,
            chat_id=str(chat_id) if chat_id else None,
            telegram_client=telegram_client,
        )
        return {"status": "ok"}
    except Exception as exc:
        logger.error("Telegram webhook request failed: chat_id=%s, error=%s", chat_id, exc)
        return {"status": "error", "message": str(exc)}

