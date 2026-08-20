from typing import Any, Dict, Optional
import httpx


class TelegramAssistantClient:
    """Thin wrapper around Telegram Bot API HTTP endpoints."""

    def __init__(self, bot_token: Optional[str] = None, default_chat_id: Optional[str] = None):
        if bot_token is None or default_chat_id is None:
            from app.config import settings
            bot_token = bot_token or settings.TELEGRAM_BOT_TOKEN
            default_chat_id = default_chat_id or settings.TELEGRAM_CHAT_ID

        self.bot_token = bot_token
        self.default_chat_id = default_chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = "Markdown"
    ) -> Dict[str, Any]:
        """Send a text message via Telegram Bot API."""
        target_chat_id = chat_id or self.default_chat_id
        url = f"{self.base_url}/sendMessage"
        payload: Dict[str, Any] = {
            "chat_id": target_chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload)
            if response.status_code != 200 and parse_mode:
                # Fallback to plain text if Markdown parsing fails on Telegram's side
                payload.pop("parse_mode", None)
                response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    def send_reminder_notification(
        self,
        title: str,
        due_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a formatted reminder notification message to Telegram."""
        msg = f"🔔 *Reminder Alert*\n\n📌 *Task:* {title}"
        if due_date:
            msg += f"\n📅 *Due:* {due_date}"
        return self.send_message(msg)

    def download_file_bytes(self, file_id: str, timeout: float = 15.0) -> tuple[bytes, str]:
        """Fetch file path from Telegram getFile API and download binary bytes.

        Returns (file_bytes, mime_type).
        """
        if not self.bot_token:
            raise ValueError("Telegram bot token is missing.")

        get_file_url = f"{self.base_url}/getFile"
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(get_file_url, json={"file_id": file_id})
            resp.raise_for_status()
            file_data = resp.json()

            file_path = file_data.get("result", {}).get("file_path")
            if not file_path:
                raise ValueError(f"No file path returned for Telegram file ID {file_id}")

            download_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
            dl_resp = client.get(download_url)
            dl_resp.raise_for_status()

            # Infer mime type from file path
            mime = "image/jpeg"
            if file_path.lower().endswith(".png"):
                mime = "image/png"
            elif file_path.lower().endswith(".webp"):
                mime = "image/webp"
            elif file_path.lower().endswith(".pdf"):
                mime = "application/pdf"

            return dl_resp.content, mime
