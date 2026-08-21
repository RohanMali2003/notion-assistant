import os

# Set fallback environment variables for test execution before app modules import
os.environ.setdefault("NOTION_TOKEN", "test_notion_token")
os.environ.setdefault("NOTION_DATABASE_ID", "test_notion_database_id")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test_telegram_bot_token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "123456789")
os.environ.setdefault("TELEGRAM_WEBHOOK_SECRET", "test_webhook_secret")
os.environ.setdefault("GEMINI_API_KEY", "test_gemini_api_key")
os.environ.setdefault("MOTION_DATA_DIR", "data/motion_test")
