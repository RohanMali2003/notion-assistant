"""Unit tests for Ocean v3.2: Background Entity Extraction, Notion Graph Sync, and Disambiguation Menus."""

import pytest
from unittest.mock import MagicMock, patch

from app.graph_memory_service import GraphMemoryService
from app.matcher import entity_resolver
from app.memory import conversation_memory
from app.schemas import TaskActionAnalysis


def test_ambiguous_menu_detection():
    candidates = [
        {"title": "Pay Berkshire Dining Hall bill"},
        {"title": "Pay Berkshire Housing deposit"},
    ]

    matched, tier, score = entity_resolver.resolve_entity(
        query="Pay Berkshire",
        candidates=candidates,
    )
    assert tier == "ambiguous_menu"
    assert isinstance(matched, list)
    assert len(matched) == 2


def test_pending_menu_memory_lifecycle():
    sender_id = "test_user_v32"

    conversation_memory.set_pending_menu(sender_id, {
        "module": "TASK_ACTION",
        "candidates": [{"title": "Task A"}, {"title": "Task B"}],
    })

    pending = conversation_memory.get_pending_menu(sender_id)
    assert pending is not None
    assert len(pending["candidates"]) == 2

    conversation_memory.clear_pending_menu(sender_id)
    assert conversation_memory.get_pending_menu(sender_id) is None


def test_extract_and_index_entities_mock():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_ext.db")
        service = GraphMemoryService(db_path=db_path)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"entities": [{"name": "UMass Course", "entity_type": "PROJECT", "summary": "CS690"}]}'
        mock_client.models.generate_content.return_value = mock_response

        with patch("app.graph_memory_service.get_gemini_client", return_value=mock_client):
            res = service.extract_and_index_entities_from_text("Studying CS690 at UMass Course", source_module="LEARNING")
            assert res["extracted_count"] == 1
            assert "UMass Course" in res["nodes"]

            node = service.get_node("UMass Course")
            assert node is not None
            assert node["entity_type"] == "PROJECT"
