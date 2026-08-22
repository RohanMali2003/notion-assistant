"""Unit tests for notion_schema_engine.py (Dynamic Schema Introspection & Polymorphic Property Adapters)."""

from unittest.mock import MagicMock
import pytest

from app.notion_schema_engine import (
    CheckboxPropertyAdapter,
    DatePropertyAdapter,
    MultiSelectPropertyAdapter,
    NotionDatabaseSchema,
    NotionSchemaEngine,
    NumberPropertyAdapter,
    RichTextPropertyAdapter,
    SelectPropertyAdapter,
    StatusPropertyAdapter,
    TitlePropertyAdapter,
    UrlPropertyAdapter,
    schema_engine,
)


def test_property_adapters_formatting():
    # 1. Title
    title_adapter = TitlePropertyAdapter("Name", "title", {})
    t_res = title_adapter.format_value("The Pragmatic Programmer")
    assert t_res == {"title": [{"type": "text", "text": {"content": "The Pragmatic Programmer"}}]}

    # 2. Rich Text & Chunking
    rt_adapter = RichTextPropertyAdapter("Notes", "rich_text", {})
    rt_res = rt_adapter.format_value("Great classic software engineering book.")
    assert rt_res == {"rich_text": [{"type": "text", "text": {"content": "Great classic software engineering book."}}]}

    # 3. Select with case-insensitive option matching
    sel_adapter = SelectPropertyAdapter(
        "Priority",
        "select",
        {"select": {"options": [{"name": "High"}, {"name": "Medium"}, {"name": "Low"}]}},
    )
    sel_res = sel_adapter.format_value("high")
    assert sel_res == {"select": {"name": "High"}}

    # 4. Status with options matching
    status_adapter = StatusPropertyAdapter(
        "Status",
        "status",
        {"status": {"options": [{"name": "Want to Read"}, {"name": "In Progress"}, {"name": "Read"}]}},
    )
    status_res = status_adapter.format_value("want to read")
    assert status_res == {"status": {"name": "Want to Read"}}

    # 5. MultiSelect (List or comma string)
    ms_adapter = MultiSelectPropertyAdapter(
        "Tags",
        "multi_select",
        {"multi_select": {"options": [{"name": "Software"}, {"name": "AI"}]}},
    )
    ms_res_list = ms_adapter.format_value(["software", "Career"])
    assert ms_res_list == {"multi_select": [{"name": "Software"}, {"name": "Career"}]}

    ms_res_str = ms_adapter.format_value("software, AI, Reading")
    assert ms_res_str == {"multi_select": [{"name": "Software"}, {"name": "AI"}, {"name": "Reading"}]}

    # 6. Date resolution
    date_adapter = DatePropertyAdapter("Due Date", "date", {})
    date_res = date_adapter.format_value("2026-08-30")
    assert date_res == {"date": {"start": "2026-08-30"}}

    # 7. Number casting
    num_adapter = NumberPropertyAdapter("Rating", "number", {})
    assert num_adapter.format_value(5) == {"number": 5}
    assert num_adapter.format_value("4.5 stars") == {"number": 4.5}

    # 8. URL formatting
    url_adapter = UrlPropertyAdapter("Link", "url", {})
    assert url_adapter.format_value("github.com/test") == {"url": "https://github.com/test"}

    # 9. Checkbox
    cb_adapter = CheckboxPropertyAdapter("Done", "checkbox", {})
    assert cb_adapter.format_value(True) == {"checkbox": True}
    assert cb_adapter.format_value("yes") == {"checkbox": True}
    assert cb_adapter.format_value("no") == {"checkbox": False}


def test_database_schema_introspection_and_payload_building():
    raw_schema = {
        "properties": {
            "Book Title": {"type": "title", "title": {}},
            "Author": {"type": "rich_text", "rich_text": {}},
            "Genre": {"type": "multi_select", "multi_select": {"options": [{"name": "Tech"}, {"name": "Sci-Fi"}]}},
            "Status": {"type": "status", "status": {"options": [{"name": "Want to Read"}, {"name": "Read"}]}},
            "Rating": {"type": "number", "number": {}},
            "URL": {"type": "url", "url": {}},
        }
    }

    schema = NotionDatabaseSchema("db_test_123", raw_schema, title="Reading List")
    assert len(schema.adapters) == 6
    assert schema.title_property_name == "Book Title"

    item_fields = {
        "title": "A Philosophy of Software Design",
        "author": "John Ousterhout",
        "tags": ["tech", "Architecture"],
        "rating": 5,
        "url": "https://web.stanford.edu/~ouster/cgi-bin/book.php",
    }

    built = schema.build_page_properties(item_fields, default_status="Want to Read")

    assert "Book Title" in built
    assert built["Book Title"]["title"][0]["text"]["content"] == "A Philosophy of Software Design"

    assert "Author" in built
    assert built["Author"]["rich_text"][0]["text"]["content"] == "John Ousterhout"

    assert "Genre" in built
    assert built["Genre"]["multi_select"] == [{"name": "Tech"}, {"name": "Architecture"}]

    assert "Status" in built
    assert built["Status"]["status"]["name"] == "Want to Read"

    assert "Rating" in built
    assert built["Rating"]["number"] == 5

    assert "URL" in built
    assert built["URL"]["url"] == "https://web.stanford.edu/~ouster/cgi-bin/book.php"


def test_schema_engine_caching():
    mock_notion = MagicMock()
    mock_notion._request_with_retry.side_effect = lambda f, *a, **kw: f(*a, **kw)
    mock_notion.client.databases.retrieve.return_value = {
        "title": [{"plain_text": "Projects"}],
        "properties": {
            "Name": {"type": "title", "title": {}},
            "Tags": {"type": "multi_select", "multi_select": {}},
        }
    }

    schema_engine.clear_cache()
    s1 = schema_engine.get_schema("db_projects_1", mock_notion)
    assert s1 is not None
    assert s1.title == "Projects"

    # Second call should hit cache
    s2 = schema_engine.get_schema("db_projects_1", mock_notion)
    assert s2 is s1
    assert mock_notion.client.databases.retrieve.call_count == 1
