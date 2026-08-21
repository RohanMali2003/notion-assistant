"""Centralized Notion Property Extraction & Block Builder Utilities.

Provides shared, robust helpers to inspect Notion page objects and build
Notion block structures without duplication across services.
"""

from typing import Any, Dict, List, Optional


# ==========================================
# --- Notion Property Extraction Helpers ---
# ==========================================

def extract_page_title(page: Dict[str, Any], default: str = "") -> str:
    """Extract plain-text title from a Notion page or database object."""
    if not isinstance(page, dict):
        return default

    props = page.get("properties", {})
    if isinstance(props, dict):
        # 1. Search for explicit title type property
        for val in props.values():
            if isinstance(val, dict) and val.get("type") == "title":
                title_list = val.get("title", [])
                if title_list:
                    extracted = "".join(t.get("plain_text", "") for t in title_list).strip()
                    if extracted:
                        return extracted

        # 2. Check common title property keys
        for key in ("Name", "Title", "Task name", "Topic", "Log", "Entry"):
            if key in props and isinstance(props[key], dict):
                title_list = props[key].get("title", [])
                if title_list:
                    extracted = "".join(t.get("plain_text", "") for t in title_list).strip()
                    if extracted:
                        return extracted

    # 3. Direct title attribute on database objects
    if "title" in page and isinstance(page["title"], list):
        extracted = "".join(t.get("plain_text", "") for t in page["title"]).strip()
        if extracted:
            return extracted

    # 4. Child page block
    if "child_page" in page and isinstance(page["child_page"], dict):
        extracted = page["child_page"].get("title", "").strip()
        if extracted:
            return extracted

    return default


def extract_page_url(page: Dict[str, Any]) -> str:
    """Extract or construct full web URL for a Notion page."""
    if not isinstance(page, dict):
        return ""

    url = page.get("url")
    if url:
        return url

    page_id = page.get("id", "")
    if page_id:
        clean_id = str(page_id).replace("-", "")
        return f"https://www.notion.so/{clean_id}"
    return ""


def extract_page_status(page: Dict[str, Any]) -> Optional[str]:
    """Extract Status property value from a Notion page."""
    if not isinstance(page, dict):
        return None

    props = page.get("properties", {})
    for prop_name, prop_val in props.items():
        if isinstance(prop_val, dict):
            if prop_val.get("type") == "status":
                status_obj = prop_val.get("status")
                if isinstance(status_obj, dict):
                    return status_obj.get("name")
            elif prop_name.lower() == "status" and prop_val.get("type") == "select":
                select_obj = prop_val.get("select")
                if isinstance(select_obj, dict):
                    return select_obj.get("name")
    return None


def extract_page_due_date(page: Dict[str, Any]) -> Optional[str]:
    """Extract Due date (YYYY-MM-DD) from a Notion page."""
    if not isinstance(page, dict):
        return None

    props = page.get("properties", {})
    for key in ("Due date", "Due Date", "Due", "Date", "due_date"):
        if key in props and isinstance(props[key], dict):
            date_obj = props[key].get("date")
            if date_obj and isinstance(date_obj, dict):
                return date_obj.get("start")

    # Fallback search for any date property
    for prop_val in props.values():
        if isinstance(prop_val, dict) and prop_val.get("type") == "date":
            date_obj = prop_val.get("date")
            if date_obj and isinstance(date_obj, dict):
                return date_obj.get("start")
    return None


def extract_page_tags(page: Dict[str, Any]) -> List[str]:
    """Extract list of tag strings from multi_select or select properties."""
    if not isinstance(page, dict):
        return []

    props = page.get("properties", {})
    for tag_key in ("Tags", "Tag", "tags", "tag", "Category", "Topics"):
        if tag_key in props and isinstance(props[tag_key], dict):
            val = props[tag_key]
            if "multi_select" in val and isinstance(val["multi_select"], list):
                return [item["name"] for item in val["multi_select"] if isinstance(item, dict) and "name" in item]
            elif "select" in val:
                sel = val.get("select")
                if isinstance(sel, dict) and "name" in sel:
                    return [sel["name"]]

    for val in props.values():
        if isinstance(val, dict):
            if "multi_select" in val and isinstance(val["multi_select"], list):
                res = [item["name"] for item in val["multi_select"] if isinstance(item, dict) and "name" in item]
                if res:
                    return res
    return []



def extract_page_priority(page: Dict[str, Any]) -> Optional[str]:
    """Extract Priority select value from a Notion page."""
    if not isinstance(page, dict):
        return None

    props = page.get("properties", {})
    for key in ("Priority", "priority"):
        if key in props and isinstance(props[key], dict):
            select_obj = props[key].get("select")
            if select_obj and isinstance(select_obj, dict):
                return select_obj.get("name")
    return None


# ==========================================
# --- Notion Block Construction Helpers ---
# ==========================================

def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """Chunk text to adhere to Notion's 2000 character limit per rich text object."""
    if not text:
        return [""]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def create_paragraph_block(text: str) -> Dict[str, Any]:
    """Create a standard Notion paragraph block."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": chunk}} for chunk in chunk_text(text)]
        },
    }


def create_heading_block(text: str, level: int = 2) -> Dict[str, Any]:
    """Create a Notion heading block (level 1, 2, or 3)."""
    h_type = f"heading_{max(1, min(3, level))}"
    return {
        "object": "block",
        "type": h_type,
        h_type: {
            "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
        },
    }


def create_bullet_block(text: str) -> Dict[str, Any]:
    """Create a Notion bulleted list item block."""
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"type": "text", "text": {"content": chunk}} for chunk in chunk_text(text)]
        },
    }


def create_todo_block(text: str, checked: bool = False) -> Dict[str, Any]:
    """Create a Notion to_do checkbox block."""
    return {
        "object": "block",
        "type": "to_do",
        "to_do": {
            "rich_text": [{"type": "text", "text": {"content": chunk}} for chunk in chunk_text(text)],
            "checked": checked,
        },
    }


def create_callout_block(text: str, emoji: str = "💡") -> Dict[str, Any]:
    """Create a Notion callout block with custom emoji icon."""
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": chunk}} for chunk in chunk_text(text)],
            "icon": {"type": "emoji", "emoji": emoji},
        },
    }


def create_code_block(code: str, language: str = "python") -> Dict[str, Any]:
    """Create a Notion code block."""
    return {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": [{"type": "text", "text": {"content": chunk}} for chunk in chunk_text(code)],
            "language": language.lower(),
        },
    }


def create_divider_block() -> Dict[str, Any]:
    """Create a Notion divider block."""
    return {
        "object": "block",
        "type": "divider",
        "divider": {},
    }


def create_numbered_block(text: str) -> Dict[str, Any]:
    """Create a Notion numbered list item block."""
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {
            "rich_text": [{"type": "text", "text": {"content": chunk}} for chunk in chunk_text(text)]
        },
    }


def create_quote_block(text: str) -> Dict[str, Any]:
    """Create a Notion quote block."""
    return {
        "object": "block",
        "type": "quote",
        "quote": {
            "rich_text": [{"type": "text", "text": {"content": chunk}} for chunk in chunk_text(text)]
        },
    }


def create_rich_text(
    content: str,
    url: Optional[str] = None,
    bold: bool = False,
    italic: bool = False,
    code: bool = False,
) -> Dict[str, Any]:
    """Create a single Notion rich_text object with annotations and optional URL link."""
    obj: Dict[str, Any] = {
        "type": "text",
        "text": {"content": content[:2000]},
    }
    if url:
        obj["text"]["link"] = {"url": url}
    annotations = {}
    if bold:
        annotations["bold"] = True
    if italic:
        annotations["italic"] = True
    if code:
        annotations["code"] = True
    if annotations:
        obj["annotations"] = annotations
    return obj


# ==========================================
# --- Notion Property Construction Helpers ---
# ==========================================

def format_title_property(title: str) -> Dict[str, Any]:
    """Format title property object for page creation."""
    return {
        "title": [{"type": "text", "text": {"content": (title or "Untitled")[:2000]}}]
    }


def format_select_property(name: str) -> Dict[str, Any]:
    """Format select property object."""
    return {"select": {"name": name}}


def format_multi_select_property(names: List[str]) -> Dict[str, Any]:
    """Format multi_select property object."""
    return {"multi_select": [{"name": n} for n in names if n]}


def format_date_property(start_date: str) -> Dict[str, Any]:
    """Format date property object (YYYY-MM-DD)."""
    return {"date": {"start": start_date}}


def format_relation_property(page_ids: List[str]) -> Dict[str, Any]:
    """Format relation property object with target page IDs."""
    return {"relation": [{"id": pid} for pid in page_ids if pid]}


def format_url_property(url: str) -> Dict[str, Any]:
    """Format URL property object."""
    return {"url": url or None}


def format_rich_text_property(text: str) -> Dict[str, Any]:
    """Format rich_text property object chunked to Notion limit."""
    return {
        "rich_text": [{"type": "text", "text": {"content": chunk}} for chunk in chunk_text(text)]
    }


def build_notion_block(content: str, block_type: str = "bulleted_list_item") -> Dict[str, Any]:
    """Build a Notion block dict by string type name."""
    b_type = (block_type or "bulleted_list_item").lower()
    if b_type in ("to_do", "todo", "task"):
        return create_todo_block(content)
    elif b_type in ("paragraph", "text"):
        return create_paragraph_block(content)
    elif b_type in ("callout", "note"):
        return create_callout_block(content)
    elif b_type in ("heading_1", "h1"):
        return create_heading_block(content, level=1)
    elif b_type in ("heading_2", "h2", "heading"):
        return create_heading_block(content, level=2)
    elif b_type in ("heading_3", "h3"):
        return create_heading_block(content, level=3)
    elif b_type in ("numbered_list_item", "numbered", "num"):
        return create_numbered_block(content)
    elif b_type in ("quote", "blockquote"):
        return create_quote_block(content)
    elif b_type in ("code", "code_block"):
        return create_code_block(content)
    elif b_type in ("divider", "hr"):
        return create_divider_block()
    else:
        return create_bullet_block(content)

