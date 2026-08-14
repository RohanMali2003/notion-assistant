import os
from pathlib import Path

# Use python-dotenv for local dev only if .env file exists
env_file = Path(".env")
if env_file.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)
    except ImportError:
        pass

REQUIRED_ENV_VARS = [
    "NOTION_TOKEN",
    "NOTION_DATABASE_ID",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
]

# Check for missing required environment variables at startup
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

if missing_vars:
    missing_str = ", ".join(missing_vars)
    raise RuntimeError(
        f"Missing required environment variable(s): [{missing_str}]. "
        "Please specify these variables in your environment or local .env file."
    )


class Settings:
    """Application settings loaded from environment variables."""

    NOTION_TOKEN: str = os.environ["NOTION_TOKEN"]
    NOTION_DATABASE_ID: str = os.environ["NOTION_DATABASE_ID"]
    TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID: str = os.environ["TELEGRAM_CHAT_ID"]
    APP_ENV: str = os.getenv("APP_ENV", "production")
    PORT: int = int(os.getenv("PORT", "8000"))
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")
    TELEGRAM_WEBHOOK_SECRET: str = os.getenv("TELEGRAM_WEBHOOK_SECRET", os.getenv("WEBHOOK_SECRET", ""))
    WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    WHATSAPP_TOKEN: str = os.getenv("WHATSAPP_TOKEN", "")
    WHATSAPP_PHONE_NUMBER_ID: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")


settings = Settings()

