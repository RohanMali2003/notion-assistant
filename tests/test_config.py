import os
import sys
from unittest.mock import patch
import pytest


def test_config_fail_fast_missing_vars():
    """Verify that importing config without required env vars raises a RuntimeError listing missing vars."""
    env_clean = {
        "NOTION_TOKEN": "",
        "NOTION_DATABASE_ID": "",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
    }
    with patch.dict(os.environ, env_clean, clear=True):
        if "app.config" in sys.modules:
            del sys.modules["app.config"]
        
        with pytest.raises(RuntimeError) as exc_info:
            import app.config  # noqa: F401

        err_msg = str(exc_info.value)
        assert "NOTION_TOKEN" in err_msg
        assert "NOTION_DATABASE_ID" in err_msg
        assert "TELEGRAM_BOT_TOKEN" in err_msg
        assert "TELEGRAM_CHAT_ID" in err_msg


def test_config_success_when_vars_present():
    """Verify that config imports successfully when all required env vars are present."""
    env_valid = {
        "NOTION_TOKEN": "test_notion_token",
        "NOTION_DATABASE_ID": "test_db_id",
        "TELEGRAM_BOT_TOKEN": "test_bot_token",
        "TELEGRAM_CHAT_ID": "12345",
    }
    with patch.dict(os.environ, env_valid, clear=True):
        if "app.config" in sys.modules:
            del sys.modules["app.config"]
        
        import app.config
        assert app.config.settings.NOTION_TOKEN == "test_notion_token"
        assert app.config.settings.NOTION_DATABASE_ID == "test_db_id"
        assert app.config.settings.TELEGRAM_BOT_TOKEN == "test_bot_token"
        assert app.config.settings.TELEGRAM_CHAT_ID == "12345"


def test_config_mind_db_ids():
    """Verify that MIND module database IDs are loaded into settings."""
    env_valid = {
        "NOTION_TOKEN": "test_notion_token",
        "NOTION_DATABASE_ID": "test_db_id",
        "TELEGRAM_BOT_TOKEN": "test_bot_token",
        "TELEGRAM_CHAT_ID": "12345",
        "NOTION_SUBSTACK_ID": "substack_123",
        "NOTION_RAMBLINGS_ID": "ramblings_456",
        "NOTION_DAILY_LOGS_ID": "daily_logs_789",
    }
    with patch.dict(os.environ, env_valid, clear=True):
        if "app.config" in sys.modules:
            del sys.modules["app.config"]

        import app.config
        assert app.config.settings.NOTION_SUBSTACK_ID == "substack_123"
        assert app.config.settings.NOTION_RAMBLINGS_ID == "ramblings_456"
        assert app.config.settings.NOTION_DAILY_LOGS_ID == "daily_logs_789"


def test_config_aliases_support():
    """Verify that NOTION_API_KEY and NOTION_TASKS_DB_ID are accepted as valid aliases."""
    env_valid = {
        "NOTION_API_KEY": "test_alias_token",
        "NOTION_TASKS_DB_ID": "test_alias_db_id",
        "TELEGRAM_BOT_TOKEN": "test_bot_token",
        "TELEGRAM_CHAT_ID": "12345",
    }
    with patch("pathlib.Path.is_file", return_value=False), patch.dict(os.environ, env_valid, clear=True):
        if "app.config" in sys.modules:
            del sys.modules["app.config"]

        import app.config
        assert app.config.settings.NOTION_TOKEN == "test_alias_token"
        assert app.config.settings.NOTION_DATABASE_ID == "test_alias_db_id"
        assert app.config.settings.NOTION_TASKS_DB_ID == "test_alias_db_id"
