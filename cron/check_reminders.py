"""Standalone script to check Notion reminders and send Telegram alerts.

Reads environment variables directly, queries Notion for reminder candidates,
and dispatches Telegram alerts via direct HTTP POST requests without FastAPI dependencies.
"""

import os
import sys
import requests
from app.notion_client import NotionAssistantClient


def check_reminders() -> None:
    """Fetch reminder candidates and send Telegram notification."""
    notion_api_key = os.environ.get("NOTION_API_KEY") or os.environ.get("NOTION_TOKEN")
    notion_tasks_db_id = os.environ.get("NOTION_TASKS_DB_ID") or os.environ.get("NOTION_DATABASE_ID")
    telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    missing_vars = []
    if not notion_api_key:
        missing_vars.append("NOTION_API_KEY")
    if not notion_tasks_db_id:
        missing_vars.append("NOTION_TASKS_DB_ID")
    if not telegram_bot_token:
        missing_vars.append("TELEGRAM_BOT_TOKEN")
    if not telegram_chat_id:
        missing_vars.append("TELEGRAM_CHAT_ID")

    if missing_vars:
        missing_str = ", ".join(missing_vars)
        raise ValueError(
            f"Missing required environment variable(s): [{missing_str}]"
        )

    notion_client = NotionAssistantClient(
        token=notion_api_key, database_id=notion_tasks_db_id
    )

    due_soon, high_priority_no_due = notion_client.get_reminder_candidates()

    if not due_soon and not high_priority_no_due:
        print("No reminders due soon or high priority without due date.")
        sys.exit(0)

    lines = []
    lines.append("📋 *Due soon*")
    if due_soon:
        for task in due_soon:
            title = task.get("title", "Untitled")
            due_date = task.get("due_date")
            if due_date:
                lines.append(f"• {title} (Due: {due_date})")
            else:
                lines.append(f"• {title}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("🔥 *High priority, no due date*")
    if high_priority_no_due:
        for task in high_priority_no_due:
            title = task.get("title", "Untitled")
            lines.append(f"• {title}")
    else:
        lines.append("None")

    message = "\n".join(lines)

    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": telegram_chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            payload.pop("parse_mode", None)
            response = requests.post(url, json=payload, timeout=10)

        if response.status_code != 200:
            print(
                f"Telegram send failed with HTTP status {response.status_code}: {response.text}",
                file=sys.stderr,
            )
            sys.exit(1)

        result = response.json()
        if not result.get("ok"):
            print(
                f"Telegram API returned non-ok response: {result}",
                file=sys.stderr,
            )
            sys.exit(1)

        print("Telegram notification sent successfully.")
    except Exception as err:
        print(f"Error executing Telegram request: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    from pathlib import Path
    env_file = Path(".env")
    if env_file.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_file)
        except ImportError:
            pass
    check_reminders()
