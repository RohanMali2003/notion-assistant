from unittest.mock import MagicMock, patch
import pytest
from app.whatsapp_client import WhatsAppAssistantClient


@pytest.fixture
def env_setup(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "fake_whatsapp_token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "100200300")
    monkeypatch.setenv("WHATSAPP_API_URL", "https://graph.facebook.com/v20.0")


def test_whatsapp_client_init(env_setup):
    client = WhatsAppAssistantClient()
    assert client.token == "fake_whatsapp_token"
    assert client.phone_number_id == "100200300"
    assert "https://graph.facebook.com/v20.0/100200300/messages" in client.messages_url


@patch("httpx.Client.post")
def test_whatsapp_send_message_success(mock_post, env_setup):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"messages": [{"id": "wamid.123"}]}
    mock_post.return_value = mock_resp

    client = WhatsAppAssistantClient()
    res = client.send_message(to="+1 (234) 567-890", text="Hello from assistant!")

    assert res == {"messages": [{"id": "wamid.123"}]}
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["json"]["to"] == "1234567890"
    assert call_kwargs["json"]["text"]["body"] == "Hello from assistant!"


def test_whatsapp_send_message_missing_credentials(monkeypatch):
    monkeypatch.setenv("WHATSAPP_TOKEN", "")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "")

    client = WhatsAppAssistantClient(token="", phone_number_id="")
    res = client.send_message(to="123456", text="Hello")
    assert res["status"] == "skipped"
    assert res["reason"] == "missing_credentials"
