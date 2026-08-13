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
