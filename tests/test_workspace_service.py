"""Unit tests for Dynamic Workspace Hierarchy, Page Explorer, and Block Content Inspection."""

from unittest.mock import MagicMock, patch
import pytest
from app.schemas import SearchQueryAnalysis, WorkspacePageNode
from app.search_service import execute_second_brain_search_pipeline
from app.workspace_service import (
    build_workspace_hierarchy_graph,
    explore_container,
    extract_page_block_contents,
    inspect_page_content,
    suggest_page_archival,
)


def test_build_workspace_hierarchy_graph():
    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client

    # Search returns Home, Notes, and Substack
    search_results = [
        {
            "id": "page-home",
            "url": "https://notion.so/home",
            "properties": {"title": {"type": "title", "title": [{"plain_text": "Home"}]}},
            "parent": {"type": "workspace"},
        },
        {
            "id": "page-notes",
            "url": "https://notion.so/notes",
            "properties": {"title": {"type": "title", "title": [{"plain_text": "Notes"}]}},
            "parent": {"type": "page_id", "page_id": "page-home"},
        },
    ]
    def mock_request(func, *args, **kwargs):
        block_id = kwargs.get("block_id")
        if block_id == "page-home":
            return {"results": [{"type": "child_page", "id": "page-notes", "child_page": {"title": "Notes"}}]}
        if block_id == "page-notes":
            return {"results": [{"type": "child_page", "id": "page-budget", "child_page": {"title": "year one budget"}}]}
        return {"results": search_results}

    mock_notion._request_with_retry.side_effect = mock_request

    graph = build_workspace_hierarchy_graph(notion_client=mock_notion, force_refresh=True)
    assert len(graph) >= 2
    assert "page-notes" in graph
    assert "Home > Notes" in graph["page-notes"].breadcrumb
    assert graph["page-notes"].is_container is True


def test_explore_container():
    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client

    # Mock build_workspace_hierarchy_graph
    with patch("app.workspace_service.build_workspace_hierarchy_graph") as mock_graph:
        notes_node = WorkspacePageNode(
            id="page-notes",
            title="Notes",
            url="https://notion.so/notes",
            breadcrumb="Home > Notes",
            is_container=True,
            children_pages=[
                {"id": "page-budget", "title": "year one budget", "url": "https://notion.so/budget", "type": "page"},
                {"id": "page-ideas", "title": "ideas for projects", "url": "https://notion.so/ideas", "type": "page"},
            ],
        )
        mock_graph.return_value = {"page-notes": notes_node}

        res = explore_container("Notes", notion_client=mock_notion)
        assert res.status == "ok"
        assert res.container_title == "Notes"
        assert len(res.subpages) == 2
        assert "year one budget" in res.reply_text
        assert "ideas for projects" in res.reply_text
        assert "https://notion.so/budget" in res.reply_text


def test_extract_page_block_contents():
    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client

    blocks = [
        {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Fall 2026 Budget"}]}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Estimated tuition: $15,000"}]}},
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "Housing: $1,200/mo"}]}},
        {"type": "to_do", "to_do": {"checked": True, "rich_text": [{"plain_text": "Submit visa documents"}]}},
    ]
    mock_notion._request_with_retry.return_value = {"results": blocks}

    lines, count = extract_page_block_contents("page-budget", notion_client=mock_notion)
    assert count == 4
    assert "# Fall 2026 Budget" in lines[0]
    assert "Estimated tuition: $15,000" in lines[1]
    assert "• Housing: $1,200/mo" in lines[2]
    assert "[✓] Submit visa documents" in lines[3]


@patch("app.workspace_service.get_gemini_client")
def test_inspect_page_content(mock_gemini_cls):
    mock_gemini = MagicMock()
    mock_gemini_cls.return_value = mock_gemini
    mock_resp = MagicMock()
    mock_resp.text = "The year one budget outlines estimated tuition of $15,000 and housing expenses of $1,200 per month."
    mock_gemini.models.generate_content.return_value = mock_resp

    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client

    with patch("app.workspace_service.build_workspace_hierarchy_graph") as mock_graph, \
         patch("app.workspace_service.extract_page_block_contents") as mock_extract:
        
        budget_node = WorkspacePageNode(
            id="page-budget",
            title="year one budget",
            url="https://notion.so/budget",
            breadcrumb="Home > Notes > year one budget",
        )
        mock_graph.return_value = {"page-budget": budget_node}
        mock_extract.return_value = (["# Budget", "Tuition: $15,000"], 2)

        res = inspect_page_content("year one budget", user_question="what is in year one budget", notion_client=mock_notion)
        assert res.status == "ok"
        assert res.page_title == "year one budget"
        assert "Home > Notes > year one budget" in res.breadcrumb
        assert "15,000" in res.reply_text
        assert "https://notion.so/budget" in res.reply_text


def test_suggest_page_archival():
    mock_notion = MagicMock()
    mock_client = MagicMock()
    mock_notion.client = mock_client

    with patch("app.workspace_service.inspect_page_content") as mock_inspect:
        mock_inspect.return_value = MagicMock(
            status="ok",
            page_title="year one budget",
            page_url="https://notion.so/budget",
            breadcrumb="Home > Notes > year one budget",
        )

        res = suggest_page_archival("year one budget", notion_client=mock_notion)
        assert res["status"] == "ok"
        assert "Archive Suggestion" in res["reply_text"]
        assert "year one budget" in res["reply_text"]
        assert "https://notion.so/budget" in res["reply_text"]


@patch("app.search_service.explore_container")
def test_execute_second_brain_search_folder_explore(mock_explore):
    mock_explore.return_value = MagicMock(
        status="ok",
        container_title="Notes",
        subpages=[{"title": "year one budget"}],
        reply_text="📂 *Home > Notes*\n• 📄 *year one budget*",
    )

    mock_wa = MagicMock()
    mock_tg = MagicMock()

    analysis = SearchQueryAnalysis(
        query="what's in my notes?",
        search_type="FOLDER_EXPLORE",
        container_name="Notes",
    )

    res = execute_second_brain_search_pipeline(
        query="what's in my notes?",
        search_analysis=analysis,
        to_phone="15551234567",
        chat_id="777",
        whatsapp_client=mock_wa,
        telegram_client=mock_tg,
    )

    assert res["type"] == "FOLDER_EXPLORE"
    mock_wa.send_message.assert_called_once()
    mock_tg.send_message.assert_called_once()
