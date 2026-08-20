"""Unit tests for Multimodal Vision & Media Processing Service."""

from unittest.mock import MagicMock, patch
import pytest
from app.media_service import (
    analyze_image_with_gemini,
    execute_media_pipeline,
)
from app.telegram_client import TelegramAssistantClient
from app.whatsapp_client import WhatsAppAssistantClient


@patch("app.media_service.get_gemini_client")
def test_analyze_image_with_gemini_whiteboard(mock_gemini_cls):
    mock_gemini = MagicMock()
    mock_gemini_cls.return_value = mock_gemini
    mock_resp = MagicMock()
    mock_resp.text = """
TITLE: Distributed KV-Store Architecture
CATEGORY: WHITEBOARD_DIAGRAM
DOMAIN: System Design
SYNTHESIS:
The whiteboard depicts a distributed key-value store architecture consisting of:
- Consistent hashing ring with virtual nodes.
- Read/Write quorum replication (W + R > N).
- Gossip protocol for node discovery and failure detection.
ACTION_ITEMS:
- Implement virtual node mapping in Python.
- Test quorum replication under network partitions.
"""
    mock_gemini.models.generate_content.return_value = mock_resp

    res = analyze_image_with_gemini(b"fake_image_bytes", "image/png", caption="System sketch")
    assert res["title"] == "Distributed KV-Store Architecture"
    assert res["category"] == "WHITEBOARD_DIAGRAM"
    assert res["domain_tag"] == "System Design"
    assert len(res["action_items"]) == 2
    assert "Consistent hashing ring" in res["synthesis"]


@patch("app.media_service.analyze_image_with_gemini")
def test_execute_media_pipeline(mock_analyze):
    mock_analyze.return_value = {
        "title": "Dining Onboarding Form",
        "category": "DOCUMENT_RECEIPT",
        "domain_tag": "UMass Admin",
        "synthesis": "Employment form requiring submission of I-9 and direct deposit info.",
        "action_items": ["Submit I-9 form by Friday", "Upload direct deposit form"],
        "full_text": "...",
    }

    mock_notion = MagicMock()
    mock_notion.create_task.return_value = {
        "id": "task-doc-123",
        "url": "https://notion.so/task-doc-123",
    }
    mock_wa = MagicMock()
    mock_tg = MagicMock()

    res = execute_media_pipeline(
        image_bytes=b"doc_bytes",
        mime_type="image/jpeg",
        caption="Form to fill",
        to_phone="15551234567",
        chat_id="888",
        notion_client=mock_notion,
        whatsapp_client=mock_wa,
        telegram_client=mock_tg,
    )

    assert res["status"] == "ok"
    assert res["title"] == "Dining Onboarding Form"
    assert res["domain_tag"] == "UMass Admin"
    mock_notion.create_task.assert_called_once()
    mock_wa.send_message.assert_called_once()
    assert "https://notion.so/task-doc-123" in mock_wa.send_message.call_args[1]["text"]
    mock_tg.send_message.assert_called_once()


@patch("app.whatsapp_client.httpx.Client")
def test_whatsapp_download_media_bytes(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client

    # Mock metadata call
    meta_resp = MagicMock()
    meta_resp.json.return_value = {"url": "https://lookaside.fbsbx.com/123", "mime_type": "image/jpeg"}
    # Mock download call
    dl_resp = MagicMock()
    dl_resp.content = b"image_binary_data"

    mock_client.get.side_effect = [meta_resp, dl_resp]

    wa = WhatsAppAssistantClient(token="mock_token", phone_number_id="123")
    data, mime = wa.download_media_bytes("media-123")

    assert data == b"image_binary_data"
    assert mime == "image/jpeg"


@patch("app.telegram_client.httpx.Client")
def test_telegram_download_file_bytes(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client

    # Mock getFile call
    meta_resp = MagicMock()
    meta_resp.json.return_value = {"result": {"file_path": "photos/file_1.jpg"}}
    # Mock download call
    dl_resp = MagicMock()
    dl_resp.content = b"telegram_photo_data"

    mock_client.post.return_value = meta_resp
    mock_client.get.return_value = dl_resp

    tg = TelegramAssistantClient(bot_token="mock_bot_token", default_chat_id="123")
    data, mime = tg.download_file_bytes("file-123")

    assert data == b"telegram_photo_data"
    assert mime == "image/jpeg"
