"""Unit tests for 1-Tap Drop & Digest URL Ingestion Service."""

from unittest.mock import MagicMock, patch
import pytest
from app.url_digest_service import (
    execute_url_digest_background_pipeline,
    extract_urls,
    fetch_url_content,
    is_url_dominant_message,
    summarize_and_log_url,
)


def test_extract_urls():
    text1 = "Check this paper https://arxiv.org/abs/2312.11805!"
    assert extract_urls(text1) == ["https://arxiv.org/abs/2312.11805"]

    text2 = "Read https://github.com/google/gemma_pytorch and https://blog.google/technology/ai/"
    assert extract_urls(text2) == [
        "https://github.com/google/gemma_pytorch",
        "https://blog.google/technology/ai/",
    ]

    assert extract_urls("No links here") == []


def test_is_url_dominant_message():
    assert is_url_dominant_message("https://arxiv.org/abs/2312.11805") is True
    assert is_url_dominant_message("Check this out: https://arxiv.org/abs/2312.11805") is True
    assert is_url_dominant_message("I was thinking today that we should check out https://arxiv.org/abs/2312.11805 and also write a long task about it and plan our entire week") is False


@patch("app.url_digest_service.httpx.Client")
def test_fetch_url_content_arxiv(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = """
    <html>
        <h1 class="title mathjax"><span class="descriptor">Title:</span>Gemini: A Family of Highly Capable Multimodal Models</h1>
        <div class="authors"><span class="descriptor">Authors:</span>Gemini Team, Google</div>
        <blockquote class="abstract mathjax"><span class="descriptor">Abstract:</span>We present Gemini, a new family of multimodal models...</blockquote>
    </html>
    """
    mock_client.get.return_value = mock_resp

    data = fetch_url_content("https://arxiv.org/abs/2312.11805")
    assert data["source"] == "ArXiv"
    assert "Gemini" in data["title"]
    assert "Gemini Team" in data["authors"]
    assert data["inferred_format"] == "Paper"


@patch("app.url_digest_service.get_gemini_client")
@patch("app.url_digest_service.fetch_url_content")
def test_summarize_and_log_url(mock_fetch, mock_gemini_cls):
    mock_fetch.return_value = {
        "url": "https://arxiv.org/abs/2312.11805",
        "title": "Gemini: A Family of Highly Capable Multimodal Models",
        "raw_text": "Gemini models architecture and evaluation across benchmarks",
        "inferred_format": "Paper",
    }

    mock_gemini = MagicMock()
    mock_gemini_cls.return_value = mock_gemini
    mock_model_resp = MagicMock()
    mock_model_resp.text = """
TITLE: Gemini: A Family of Highly Capable Multimodal Models
FORMAT: Paper
DOMAIN: AI Research
ESSENCE: Introduces Google's flagship natively multimodal foundation models spanning Ultra, Pro, and Nano scales.
KEY_TAKEAWAYS:
- Built from the ground up for native multimodality across text, vision, audio, and code.
- Outperforms human experts on MMLU benchmark.
- Incorporates efficient cross-attention and inference optimizations.
PRACTICAL_IMPLICATIONS: Enables zero-shot cross-modal reasoning and high-efficiency mobile deployment.
"""
    mock_gemini.models.generate_content.return_value = mock_model_resp

    mock_notion = MagicMock()
    mock_notion.create_resource_row.return_value = {
        "id": "res-123",
        "url": "https://notion.so/res-123",
    }

    result = summarize_and_log_url(
        url="https://arxiv.org/abs/2312.11805",
        user_comment="Check this paper",
        notion_client=mock_notion,
    )

    assert result["status"] == "ok"
    assert result["title"] == "Gemini: A Family of Highly Capable Multimodal Models"
    assert result["domain_tag"] == "AI Research"
    assert result["format"] == "Paper"
    assert len(result["takeaways"]) == 3
    assert result["notion_url"] == "https://notion.so/res-123"
    assert "https://notion.so/res-123" in result["reply_text"]
    assert "AI Research" in result["reply_text"]


@patch("app.url_digest_service.summarize_and_log_url")
def test_execute_url_digest_background_pipeline(mock_summarize):
    mock_summarize.return_value = {
        "status": "ok",
        "url": "https://arxiv.org/abs/123",
        "title": "Test Paper",
        "reply_text": "🔗 *Digested Material:* Test Paper",
        "notion_url": "https://notion.so/123",
    }

    mock_wa = MagicMock()
    mock_tg = MagicMock()

    res = execute_url_digest_background_pipeline(
        url="https://arxiv.org/abs/123",
        to_phone="15551234567",
        chat_id="999",
        whatsapp_client=mock_wa,
        telegram_client=mock_tg,
    )

    assert res["status"] == "ok"
    mock_wa.send_message.assert_called_once_with(
        to="15551234567",
        text="🔗 *Digested Material:* Test Paper",
        preview_url=True,
    )
    mock_tg.send_message.assert_called_once_with(
        text="🔗 *Digested Material:* Test Paper",
        chat_id="999",
    )
