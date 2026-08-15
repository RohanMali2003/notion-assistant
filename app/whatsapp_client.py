import logging
import os
import re
from typing import Any, Dict, Optional
import httpx

logger = logging.getLogger(__name__)


class WhatsAppAssistantClient:
    """Thin wrapper around WhatsApp Cloud API HTTP endpoints."""

    def __init__(
        self,
        token: Optional[str] = None,
        phone_number_id: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        from app.config import settings

        if token is not None:
            self.token = token
        else:
            self.token = os.getenv("WHATSAPP_TOKEN") or getattr(settings, "WHATSAPP_TOKEN", "")

        if phone_number_id is not None:
            self.phone_number_id = phone_number_id
        else:
            self.phone_number_id = (
                os.getenv("WHATSAPP_PHONE_NUMBER_ID")
                or getattr(settings, "WHATSAPP_PHONE_NUMBER_ID", "")
            )

        if api_url is not None:
            base_api = api_url
        else:
            base_api = (
                os.getenv("WHATSAPP_API_URL")
                or getattr(settings, "WHATSAPP_API_URL", "https://graph.facebook.com/v20.0")
            )
        self.api_url = base_api.rstrip("/")
        self.messages_url = f"{self.api_url}/{self.phone_number_id}/messages"

    def send_message(
        self,
        to: str,
        text: str,
        preview_url: bool = False,
    ) -> Dict[str, Any]:
        """Send a text message via WhatsApp Cloud API."""
        if not self.token or not self.phone_number_id:
            logger.warning(
                "WhatsApp credentials missing (token=%s, phone_number_id=%s). Message not sent.",
                bool(self.token),
                bool(self.phone_number_id),
            )
            return {"status": "skipped", "reason": "missing_credentials"}

        # Normalize phone number (keep digits only)
        clean_to = re.sub(r"[^\d]", "", to.strip())

        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_to,
            "type": "text",
            "text": {
                "preview_url": preview_url,
                "body": text,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.post(self.messages_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
