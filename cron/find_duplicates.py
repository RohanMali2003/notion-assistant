"""Standalone cron script to scan Notion for duplicates and publish a Cleanup Review report."""

import logging
import os
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is in python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Load .env file
env_file = repo_root / ".env"
if env_file.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)
    except ImportError:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("find_duplicates_cron")


def run_duplicate_audit() -> int:
    """Execute duplicate scan, update Notion cleanup review page, and print results."""
    from app.cleanup_reporter import NotionCleanupReporter
    from app.telegram_client import TelegramAssistantClient
    from app.whatsapp_client import WhatsAppAssistantClient

    print("=" * 60)
    print("🧹 Starting Notion Duplicate & Cleanup Audit...")
    print("=" * 60)

    reporter = NotionCleanupReporter()
    summary = reporter.update_cleanup_report_page()

    page_url = summary.get("page_url", "")
    total_clusters = summary.get("total_duplicate_clusters", 0)
    subj_count = summary.get("duplicate_subjects", 0)
    task_count = summary.get("duplicate_tasks", 0)
    res_count = summary.get("duplicate_resources", 0)

    print("\n📊 Audit Summary:")
    print(f" • Duplicate Subject Clusters : {subj_count}")
    print(f" • Duplicate Task Clusters    : {task_count}")
    print(f" • Duplicate Resource Clusters: {res_count}")
    print(f" • Total Potential Duplicates : {total_clusters}")
    print(f"\n🔗 Notion Review Page: {page_url}")
    print("=" * 60)

    # Optional: Send notification if duplicates were found
    if total_clusters > 0:
        notif_text = (
            f"🧹 *Notion Duplicate Audit Complete*\n\n"
            f"Found *{total_clusters}* duplicate cluster(s):\n"
            f"• 🏛️ Subjects: {subj_count}\n"
            f"• 📋 Tasks: {task_count}\n"
            f"• 📚 Resources: {res_count}\n\n"
            f"Review items & clean up here:\n🔗 {page_url}"
        )

        # Telegram notification
        try:
            tg = TelegramAssistantClient()
            if tg.token and tg.default_chat_id:
                tg.send_message(text=notif_text)
                logger.info("Sent audit summary to Telegram")
        except Exception as tg_err:
            logger.debug("Could not send Telegram notification: %s", tg_err)

    return 0


if __name__ == "__main__":
    sys.exit(run_duplicate_audit())
