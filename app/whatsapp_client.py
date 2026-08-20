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

    def send_template_message(
        self,
        to: str,
        template_name: str = "ocean_notification",
        body_text: str = "",
        language_code: str = "en_US",
    ) -> Dict[str, Any]:
        """Send a pre-approved utility template message via WhatsApp Cloud API.

        Bypasses the 24-hour customer service window restrictions for $0 within
        the 1,000 free monthly conversation quota.
        """
        if not self.token or not self.phone_number_id:
            logger.warning("WhatsApp credentials missing. Template not sent.")
            return {"status": "skipped", "reason": "missing_credentials"}

        clean_to = re.sub(r"[^\d]", "", to.strip())

        # Build template payload with dynamic body parameter {{1}}
        template_components = []
        if body_text:
            template_components.append({
                "type": "body",
                "parameters": [
                    {
                        "type": "text",
                        "text": body_text[:1024],  # WhatsApp template parameter character limit
                    }
                ],
            })

        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {
                    "code": language_code,
                },
            },
        }
        if template_components:
            payload["template"]["components"] = template_components

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.post(self.messages_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    def send_message(
        self,
        to: str,
        text: str,
        preview_url: bool = False,
        use_template_fallback: bool = True,
        template_name: str = "ocean_notification",
        language_code: str = "en_US",
    ) -> Dict[str, Any]:
        """Send a text message via WhatsApp Cloud API with automatic template fallback."""
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

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(self.messages_url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            # Check if error is due to 24-hour window restriction (code 131047 or 400)
            is_window_expired = False
            try:
                err_json = exc.response.json()
                err_data = err_json.get("error", {})
                err_code = err_data.get("code")
                err_subcode = err_data.get("error_subcode")
                err_msg = err_data.get("message", "").lower()
                if (
                    err_code in (131047, 131026, 131000)
                    or err_subcode == 131047
                    or "24 hours" in err_msg
                    or "template" in err_msg
                ):
                    is_window_expired = True
            except Exception:
                is_window_expired = True

            if use_template_fallback and is_window_expired:
                logger.info(
                    "WhatsApp 24-hour window expired. Falling back to template '%s' for %s",
                    template_name,
                    clean_to,
                )
                try:
                    return self.send_template_message(
                        to=clean_to,
                        template_name=template_name,
                        body_text=text,
                        language_code=language_code,
                    )
                except Exception as t_err:
                    logger.error("WhatsApp template fallback also failed: %s", t_err)
                    raise exc
            raise exc

    def download_media_bytes(self, media_id: str, timeout: float = 15.0) -> tuple[bytes, str]:
        """Fetch media URL from WhatsApp Cloud API and download raw binary bytes.

        Returns (media_bytes, mime_type).
        """
        if not self.token:
            raise ValueError("WhatsApp token is missing.")

        media_meta_url = f"{self.api_url}/{media_id}"
        headers = {"Authorization": f"Bearer {self.token}"}

        with httpx.Client(timeout=timeout) as client:
            # 1. Query media metadata
            meta_resp = client.get(media_meta_url, headers=headers)
            meta_resp.raise_for_status()
            meta_data = meta_resp.json()

            download_url = meta_data.get("url")
            mime_type = meta_data.get("mime_type", "image/jpeg")

            if not download_url:
                raise ValueError(f"No download URL returned for WhatsApp media ID {media_id}")

            # 2. Download binary stream with auth header
            download_resp = client.get(download_url, headers=headers)
            download_resp.raise_for_status()
            return download_resp.content, mime_type
