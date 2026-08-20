"""Standalone Cron Script for Sunday Life & Study Velocity Executive Digest.

Aggregates Notion workspace activity over the past 7 days, evaluates velocity via Gemini,
creates a structured Notion Weekly Review page, and delivers the executive digest to Telegram/WhatsApp.
"""

import logging
import os
from pathlib import Path
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.notion_client import NotionAssistantClient
from app.telegram_client import TelegramAssistantClient
from app.whatsapp_client import WhatsAppAssistantClient
from app.weekly_digest_service import execute_weekly_digest_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cron.weekly_digest")


def main() -> None:
    """Run Sunday Velocity Digest pipeline."""
    print("=" * 60)
    print("⚡ Starting Sunday Life & Study Velocity Digest...")
    print("=" * 60)

    notion_client = NotionAssistantClient()
    telegram_client = TelegramAssistantClient()
    whatsapp_client = WhatsAppAssistantClient()

    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    whatsapp_phone = os.getenv("WHATSAPP_PHONE_NUMBER") or os.getenv("DEFAULT_RECIPIENT_PHONE")

    result = execute_weekly_digest_pipeline(
        to_phone=whatsapp_phone,
        chat_id=telegram_chat_id,
        days=7,
        notion_client=notion_client,
        whatsapp_client=whatsapp_client,
        telegram_client=telegram_client,
    )

    print("\n📊 Velocity Summary:")
    print(f" • Velocity Score: {result.get('velocity_score')}/100")
    print(f" • Verdict       : {result.get('verdict')}")
    print(f" • Headline      : {result.get('headline')}")
    if result.get("notion_page_url"):
        print(f" • Notion Page   : {result.get('notion_page_url')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
