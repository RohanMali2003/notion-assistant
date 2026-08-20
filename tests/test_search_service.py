"""Unit tests for Semantic Second-Brain Search & Knowledge Retrieval."""

from unittest.mock import MagicMock, patch
import pytest
from app.schemas import SearchResultItem
from app.search_service import (
    answer_second_brain_question,
    execute_second_brain_search_pipeline,
    search_workspace_knowledge,
)


def test_search_workspace_knowledge():
    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client
    mock_notion.subjects_db_id = "subj-db"
    mock_notion.resources_db_id = "res-db"
    mock_notion.tasks_db_id = "tasks-db"

    # Search API result
    global_page = {
        "id": "page-moe-1",
        "url": "https://notion.so/moe-page",
        "properties": {
            "Subject name": {"type": "title", "title": [{"plain_text": "Mixture of Experts (MoE) Architecture"}]},
        },
        "parent": {"type": "database_id", "database_id": "subj-db"},
    }
    mock_notion._request_with_retry.return_value = {"results": [global_page]}

    results = search_workspace_knowledge("Mixture of Experts", notion_client=mock_notion)
    assert len(results) >= 1
    assert "Mixture of Experts" in results[0].title
    assert results[0].category == "Subject"


@patch("app.search_service.get_gemini_client")
def test_answer_second_brain_question(mock_gemini_cls):
    mock_gemini = MagicMock()
    mock_gemini_cls.return_value = mock_gemini
    mock_resp = MagicMock()
    mock_resp.text = (
        "Based on your Notion notes, Mixture of Experts (MoE) uses conditional computation via sparsely-gated layers "
        "where only top-k experts (typically k=2) are active per token, enabling massive parameter scaling without proportional compute increase."
    )
    mock_gemini.models.generate_content.return_value = mock_resp

    items = [
        SearchResultItem(
            title="Mixture of Experts (MoE) Architecture",
            url="https://notion.so/moe",
            category="Subject",
            snippet="Conditional computation and gating mechanism for scaling parameters",
        )
    ]

    res = answer_second_brain_question("How does MoE work?", items)
    assert res["status"] == "ok"
    assert "conditionally" in res["answer"].lower() or "computation" in res["answer"].lower()
    assert "https://notion.so/moe" in res["reply_text"]
    assert "Sources in your Notion:" in res["reply_text"]


@patch("app.search_service.search_workspace_knowledge")
@patch("app.search_service.answer_second_brain_question")
def test_execute_second_brain_search_pipeline(mock_answer, mock_search):
    mock_search.return_value = [
        SearchResultItem(
            title="Consistent Hashing in Distributed Systems",
            url="https://notion.so/ch",
            category="Subject",
        )
    ]
    mock_answer.return_value = {
        "status": "ok",
        "query": "What is consistent hashing?",
        "answer": "Consistent hashing minimizes key remapping when nodes join or leave.",
        "reply_text": "🧠 *Second-Brain Answer*\n\nConsistent hashing minimizes remapping.\n\n📚 *Sources in your Notion:*\n• *Consistent Hashing* (Subject)\n  🔗 https://notion.so/ch",
    }

    mock_wa = MagicMock()
    mock_tg = MagicMock()

    result = execute_second_brain_search_pipeline(
        query="What is consistent hashing?",
        to_phone="15551234567",
        chat_id="555",
        whatsapp_client=mock_wa,
        telegram_client=mock_tg,
    )

    assert result["status"] == "ok"
    mock_wa.send_message.assert_called_once()
    mock_tg.send_message.assert_called_once()
