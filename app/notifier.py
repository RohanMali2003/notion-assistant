"""Unified Multichannel Notification Dispatcher.

Handles delivery of messages to WhatsApp and/or Telegram with centralized
logging and robust error handling to eliminate boilerplate across background services.
"""

import logging
from typing import Any, Dict, Optional

from app.telegram_client import TelegramAssistantClient
from app.whatsapp_client import WhatsAppAssistantClient

logger = logging.getLogger("notion-assistant.notifier")


def send_notification(
    text: str,
    to_phone: Optional[str] = None,
    chat_id: Optional[str] = None,
    target: Optional[str] = None,
    preview_url: bool = False,
    whatsapp_client: Optional[Any] = None,
    telegram_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Send notification message to WhatsApp and/or Telegram channels safely.

    Supports explicit `to_phone` / `chat_id` or flexible single `target` parameter.
    """
    import sys
    app_main = sys.modules.get("app.main")
    result: Dict[str, Any] = {}

    phone = to_phone
    tg_id = chat_id

    # Auto-resolve target if provided
    if target and not phone and not tg_id:
        target_str = str(target).strip()
        if target_str.startswith("+") or (target_str.isdigit() and len(target_str) >= 10):
            phone = target_str
        else:
            tg_id = target_str

    if phone:
        try:
            if whatsapp_client is not None:
                wa = whatsapp_client
            elif app_main and hasattr(app_main, "WhatsAppAssistantClient"):
                wa = app_main.WhatsAppAssistantClient()
            else:
                wa = WhatsAppAssistantClient()
            if preview_url:
                wa.send_message(to=phone, text=text, preview_url=True)
            else:
                wa.send_message(to=phone, text=text)

            logger.info("Notification sent via WhatsApp to %s", phone)
            result["whatsapp"] = "success"
        except Exception as wa_err:
            logger.error("Failed to send notification via WhatsApp to %s: %s", phone, wa_err)
            result["whatsapp"] = f"error: {wa_err}"

    if tg_id:
        try:
            if telegram_client is not None:
                tg = telegram_client
            elif app_main and hasattr(app_main, "TelegramAssistantClient"):
                tg = app_main.TelegramAssistantClient()
            else:
                tg = TelegramAssistantClient()
            tg.send_message(text=text, chat_id=str(tg_id))
            logger.info("Notification sent via Telegram to chat_id=%s", tg_id)
            result["telegram"] = "success"
        except Exception as tg_err:
            logger.error("Failed to send notification via Telegram to chat_id=%s: %s", tg_id, tg_err)
            result["telegram"] = f"error: {tg_err}"

    return result


