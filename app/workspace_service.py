"""Workspace Hierarchy, Dynamic Page Explorer, and Block Content Inspection Service.

Ocean v2.3 Engine:
1. Crawls and maintains a dynamic Workspace Hierarchy Graph without hardcoded page IDs.
2. Explores folders and containers (e.g. "what's in my notes?", "what's in miscellaneous?").
3. Performs deep block inspection & grounded QA on arbitrary Notion documents.
4. Locates documents across the workspace and confirms existence with breadcrumbs.
5. Provides safe non-destructive page archival guidance.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from app.config import settings
from app.notion_client import NotionAssistantClient, clean_math_and_markdown
from app.schemas import (
    FolderExploreResult,
    PageInspectResult,
    SearchResultItem,
    WorkspacePageNode,
)

logger = logging.getLogger("notion-assistant.workspace")

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def get_gemini_client():
    """Create and return a google-genai Client instance."""
    if genai is None:
        raise RuntimeError("google-genai library is not installed or available")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def get_gemini_model() -> str:
    """Return configured Gemini model name."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

# --- In-Memory Dynamic Workspace Cache ---
_WORKSPACE_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "ttl_seconds": 300.0,  # 5-minute cache
    "nodes_by_id": {},
    "nodes_by_title": {},
}


def _extract_page_title(page: Dict[str, Any]) -> str:
    """Extract page plain text title from Notion properties or child_page block."""
    props = page.get("properties", {})
    for key, val in props.items():
        if isinstance(val, dict) and val.get("type") == "title":
            title_objs = val.get("title", [])
            title_text = "".join(t.get("plain_text", "") for t in title_objs).strip()
            if title_text:
                return title_text
    # Fallback to direct title field if database or block
    if "title" in page and isinstance(page["title"], list):
        return "".join(t.get("plain_text", "") for t in page["title"]).strip()
    if "child_page" in page and isinstance(page["child_page"], dict):
        return page["child_page"].get("title", "").strip()
    return ""


def _extract_page_url(page: Dict[str, Any]) -> str:
    """Extract public/workspace URL for a Notion page."""
    url = page.get("url", "")
    if url:
        return url
    page_id = page.get("id", "").replace("-", "")
    return f"https://app.notion.com/p/{page_id}" if page_id else ""


def build_workspace_hierarchy_graph(
    notion_client: Optional[NotionAssistantClient] = None,
    force_refresh: bool = False,
) -> Dict[str, WorkspacePageNode]:
    """Crawl and build a dynamic hierarchy graph of all accessible Notion pages and databases."""
    global _WORKSPACE_CACHE
    now = time.time()
    if not force_refresh and (now - _WORKSPACE_CACHE["timestamp"] < _WORKSPACE_CACHE["ttl_seconds"]) and _WORKSPACE_CACHE["nodes_by_id"]:
        return _WORKSPACE_CACHE["nodes_by_id"]

    notion = notion_client or NotionAssistantClient()
    if not notion.client:
        return {}

    nodes_by_id: Dict[str, WorkspacePageNode] = {}
    nodes_by_title: Dict[str, str] = {}

    try:
        # 1. Search for all accessible pages and databases
        search_res = notion._request_with_retry(
            notion.client.search,
            page_size=100,
        )
        results = search_res.get("results", [])

        for item in results:
            item_id = item.get("id")
            if not item_id:
                continue
            title = _extract_page_title(item)
            if not title:
                continue
            url = _extract_page_url(item)
            parent = item.get("parent", {})
            p_type = parent.get("type", "workspace")
            p_id = parent.get(f"{p_type}_id", parent.get("page_id", parent.get("database_id", parent.get("block_id"))))
            last_edited = item.get("last_edited_time")

            node = WorkspacePageNode(
                id=item_id,
                title=title,
                url=url,
                parent_type=p_type,
                parent_id=p_id,
                last_edited_time=last_edited,
            )
            nodes_by_id[item_id] = node
            nodes_by_title[title.lower()] = item_id

        # 2. Discover child pages under standalone/container pages
        for node_id, node in list(nodes_by_id.items()):
            if node.parent_type in ("workspace", "page_id", "block_id"):
                try:
                    blocks_res = notion._request_with_retry(
                        notion.client.blocks.children.list,
                        block_id=node_id,
                        page_size=50,
                    )
                    child_blocks = blocks_res.get("results", [])
                    has_children = False
                    for b in child_blocks:
                        b_type = b.get("type")
                        b_id = b.get("id")
                        if b_type == "child_page":
                            child_title = b.get("child_page", {}).get("title", "")
                            if child_title:
                                has_children = True
                                child_url = f"https://app.notion.com/p/{b_id.replace('-', '')}" if b_id else ""
                                node.children_pages.append({
                                    "id": b_id,
                                    "title": child_title,
                                    "url": child_url,
                                    "type": "page",
                                })
                                # Register child node if not already present
                                if b_id and b_id not in nodes_by_id:
                                    child_node = WorkspacePageNode(
                                        id=b_id,
                                        title=child_title,
                                        url=child_url,
                                        parent_type="page_id",
                                        parent_id=node_id,
                                    )
                                    nodes_by_id[b_id] = child_node
                                    nodes_by_title[child_title.lower()] = b_id
                        elif b_type == "child_database":
                            child_title = b.get("child_database", {}).get("title", "")
                            if child_title:
                                has_children = True
                                node.children_pages.append({
                                    "id": b_id,
                                    "title": child_title,
                                    "url": f"https://app.notion.com/p/{b_id.replace('-', '')}" if b_id else "",
                                    "type": "database",
                                })
                    node.is_container = has_children
                except Exception as b_err:
                    logger.debug("Could not list blocks for node %s: %s", node_id, b_err)

        # 3. Resolve Breadcrumbs for all nodes
        for node in nodes_by_id.values():
            crumb_parts = [node.title]
            curr_parent_id = node.parent_id
            visited = {node.id}
            while curr_parent_id and curr_parent_id in nodes_by_id and curr_parent_id not in visited:
                visited.add(curr_parent_id)
                parent_node = nodes_by_id[curr_parent_id]
                crumb_parts.insert(0, parent_node.title)
                curr_parent_id = parent_node.parent_id
            node.breadcrumb = " > ".join(crumb_parts)

        # Update cache
        _WORKSPACE_CACHE["timestamp"] = now
        _WORKSPACE_CACHE["nodes_by_id"] = nodes_by_id
        _WORKSPACE_CACHE["nodes_by_title"] = nodes_by_title

        logger.info("Successfully indexed %d Notion workspace nodes", len(nodes_by_id))
        return nodes_by_id

    except Exception as exc:
        logger.error("Failed to build workspace hierarchy graph: %s", exc)
        return nodes_by_id


def explore_container(
    container_query: str,
    notion_client: Optional[NotionAssistantClient] = None,
) -> FolderExploreResult:
    """Explore a container/folder page in Notion and list its child documents/subpages."""
    notion = notion_client or NotionAssistantClient()
    nodes = build_workspace_hierarchy_graph(notion_client=notion)

    clean_q = container_query.strip().lower()
    target_node: Optional[WorkspacePageNode] = None

    # 1. Exact or substring match in hierarchy nodes
    for node in nodes.values():
        title_lower = node.title.lower()
        if clean_q == title_lower or (len(clean_q) >= 3 and clean_q in title_lower):
            target_node = node
            break

    # 2. Fallback: Search directly via Notion search
    if not target_node and notion.client:
        try:
            res = notion._request_with_retry(
                notion.client.search,
                query=container_query,
                filter={"property": "object", "value": "page"},
                page_size=5,
            )
            for page in res.get("results", []):
                p_id = page.get("id")
                p_title = _extract_page_title(page)
                if p_title:
                    target_node = WorkspacePageNode(
                        id=p_id,
                        title=p_title,
                        url=_extract_page_url(page),
                        breadcrumb=p_title,
                    )
                    break
        except Exception as exc:
            logger.warning("Container search fallback failed: %s", exc)

    if not target_node:
        return FolderExploreResult(
            status="not_found",
            container_title=container_query,
            reply_text=f"📂 *Workspace Search*\n\nCould not find a container or folder named *'{container_query}'* in your Notion workspace.",
        )

    # 3. Fetch real-time child blocks if container node children are empty
    subpages = list(target_node.children_pages)
    if not subpages and notion.client:
        try:
            b_res = notion._request_with_retry(
                notion.client.blocks.children.list,
                block_id=target_node.id,
                page_size=100,
            )
            for b in b_res.get("results", []):
                b_type = b.get("type")
                b_id = b.get("id")
                if b_type == "child_page":
                    ctitle = b.get("child_page", {}).get("title", "")
                    if ctitle:
                        subpages.append({
                            "id": b_id,
                            "title": ctitle,
                            "url": f"https://app.notion.com/p/{b_id.replace('-', '')}" if b_id else "",
                            "type": "page",
                        })
                elif b_type == "child_database":
                    ctitle = b.get("child_database", {}).get("title", "")
                    if ctitle:
                        subpages.append({
                            "id": b_id,
                            "title": ctitle,
                            "url": f"https://app.notion.com/p/{b_id.replace('-', '')}" if b_id else "",
                            "type": "database",
                        })
        except Exception as err:
            logger.error("Failed to list children for container %s: %s", target_node.id, err)

    # 4. If container is Archive or contains an Archive Index database, also query the database rows
    if "archive" in target_node.title.lower():
        try:
            # Look for child database or search Archive Index
            archive_db_id = None
            for p in subpages:
                if p.get("type") == "database" and "archive" in p.get("title", "").lower():
                    archive_db_id = p.get("id")
                    break
            if not archive_db_id:
                # Search database directly
                db_search = notion._request_with_retry(
                    notion.client.search,
                    query="Archive Index",
                    filter={"property": "object", "value": "database"},
                    page_size=1,
                )
                if db_search.get("results"):
                    archive_db_id = db_search["results"][0].get("id")

            if archive_db_id:
                db_rows = notion._query_database(database_id=archive_db_id, page_size=50)
                for r in db_rows.get("results", []):
                    rtitle = _extract_page_title(r)
                    if rtitle:
                        subpages.append({
                            "id": r.get("id"),
                            "title": rtitle,
                            "url": _extract_page_url(r),
                            "type": "archived_doc",
                        })
        except Exception as db_err:
            logger.debug("Failed to query Archive Index DB in explore_container: %s", db_err)

    # 5. Format reply
    breadcrumb = target_node.breadcrumb or target_node.title
    if subpages:
        items_text = []
        for item in subpages:
            itype = item.get("type")
            icon = "📊" if itype == "database" else ("📦" if itype == "archived_doc" else "📄")
            item_url = item.get("url", "")
            items_text.append(f"• {icon} *{item['title']}*\n  🔗 {item_url}")
        
        reply = (
            f"📂 *{breadcrumb}* ({len(subpages)} items)\n"
            f"🔗 {target_node.url}\n\n"
            + "\n".join(items_text)
        )
    else:
        reply = (
            f"📂 *{breadcrumb}*\n"
            f"🔗 {target_node.url}\n\n"
            f"*(No subpages or child documents currently inside this page.)*"
        )

    return FolderExploreResult(
        status="ok",
        container_title=target_node.title,
        container_url=target_node.url,
        breadcrumb=breadcrumb,
        subpages=subpages,
        reply_text=clean_math_and_markdown(reply),
    )


def extract_page_block_contents(
    page_id: str,
    notion_client: Optional[NotionAssistantClient] = None,
    max_blocks: int = 50,
) -> Tuple[List[str], int]:
    """Recursively extract plain text lines from child blocks of a Notion page."""
    notion = notion_client or NotionAssistantClient()
    if not notion.client or not page_id:
        return [], 0

    lines: List[str] = []
    total_blocks = 0

    try:
        blocks_res = notion._request_with_retry(
            notion.client.blocks.children.list,
            block_id=page_id,
            page_size=max_blocks,
        )
        raw_blocks = blocks_res.get("results", [])
        total_blocks = len(raw_blocks)

        for b in raw_blocks:
            b_type = b.get("type", "")
            if b_type in ("heading_1", "heading_2", "heading_3"):
                txt = "".join(t.get("plain_text", "") for t in b.get(b_type, {}).get("rich_text", []))
                if txt:
                    prefix = "###" if b_type == "heading_3" else ("##" if b_type == "heading_2" else "#")
                    lines.append(f"{prefix} {txt}")
            elif b_type in ("paragraph", "quote", "callout"):
                txt = "".join(t.get("plain_text", "") for t in b.get(b_type, {}).get("rich_text", []))
                if txt:
                    lines.append(txt)
            elif b_type in ("bulleted_list_item", "numbered_list_item"):
                txt = "".join(t.get("plain_text", "") for t in b.get(b_type, {}).get("rich_text", []))
                if txt:
                    lines.append(f"• {txt}")
            elif b_type == "to_do":
                checked = "✓" if b.get("to_do", {}).get("checked") else "○"
                txt = "".join(t.get("plain_text", "") for t in b.get("to_do", {}).get("rich_text", []))
                if txt:
                    lines.append(f"[{checked}] {txt}")
            elif b_type == "child_page":
                ctitle = b.get("child_page", {}).get("title", "")
                if ctitle:
                    lines.append(f"[Subpage: {ctitle}]")
            elif b_type == "child_database":
                dtitle = b.get("child_database", {}).get("title", "")
                if dtitle:
                    lines.append(f"[Database: {dtitle}]")
            elif b_type == "table":
                # Tables have child table_rows
                try:
                    table_rows = notion._request_with_retry(
                        notion.client.blocks.children.list,
                        block_id=b.get("id"),
                        page_size=20,
                    )
                    for r in table_rows.get("results", []):
                        cells = r.get("table_row", {}).get("cells", [])
                        row_txts = ["".join(t.get("plain_text", "") for t in c) for c in cells]
                        lines.append(" | ".join(row_txts))
                except Exception:
                    pass
    except Exception as exc:
        logger.error("Failed to extract block contents for page %s: %s", page_id, exc)

    return lines, total_blocks


def inspect_page_content(
    page_query: str,
    user_question: Optional[str] = None,
    notion_client: Optional[NotionAssistantClient] = None,
) -> PageInspectResult:
    """Locate a Notion document, extract its block contents, and synthesize a Gemini answer."""
    notion = notion_client or NotionAssistantClient()
    nodes = build_workspace_hierarchy_graph(notion_client=notion)

    clean_q = page_query.strip().lower()
    target_node: Optional[WorkspacePageNode] = None

    # 1. Match from hierarchy nodes
    for node in nodes.values():
        title_lower = node.title.lower()
        if clean_q == title_lower or (len(clean_q) >= 3 and clean_q in title_lower):
            target_node = node
            break

    # 2. Fallback: Search Notion API
    if not target_node and notion.client:
        try:
            res = notion._request_with_retry(
                notion.client.search,
                query=page_query,
                filter={"property": "object", "value": "page"},
                page_size=5,
            )
            for page in res.get("results", []):
                p_id = page.get("id")
                p_title = _extract_page_title(page)
                if p_title:
                    target_node = WorkspacePageNode(
                        id=p_id,
                        title=p_title,
                        url=_extract_page_url(page),
                        breadcrumb=p_title,
                    )
                    break
        except Exception as exc:
            logger.warning("Page search fallback failed: %s", exc)

    if not target_node:
        return PageInspectResult(
            status="not_found",
            page_title=page_query,
            reply_text=f"📄 *Workspace Search*\n\nCould not find a page or document titled *'{page_query}'* in your Notion workspace.",
        )

    # 3. Extract Block Contents
    block_lines, block_count = extract_page_block_contents(target_node.id, notion_client=notion)
    extracted_text = "\n".join(block_lines)

    breadcrumb = target_node.breadcrumb or target_node.title

    # 4. Synthesize with Gemini
    if extracted_text.strip():
        try:
            client = get_gemini_client()
            model_name = get_gemini_model()
            question_prompt = user_question or f"Summarize the key contents, numbers, and takeaways from this page."
            sys_inst = (
                "You are an intelligent second-brain assistant answering questions about a user's Notion page.\n"
                "Explain the key contents accurately based on the extracted text.\n"
                "Highlight key figures, budget items, bullet points, or project ideas precisely.\n"
                "Do NOT use LaTeX dollar signs or double asterisks in markdown."
            )
            content_prompt = (
                f"Page Title: {target_node.title}\n"
                f"Location / Breadcrumb: {breadcrumb}\n"
                f"User Question: {question_prompt}\n\n"
                f"Page Extracted Content:\n{extracted_text}"
            )
            resp = client.models.generate_content(
                model=model_name,
                contents=content_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_inst,
                ),
            )
            synthesis = resp.text or extracted_text[:500]
        except Exception as gemini_err:
            logger.warning("Gemini synthesis of page content failed (%s). Using raw text snippet.", gemini_err)
            synthesis = "\n".join(block_lines[:15])
    else:
        synthesis = "This page is currently empty or contains no readable text blocks."

    reply = (
        f"📄 *{target_node.title}*\n"
        f"📍 Location: *{breadcrumb}*\n"
        f"🔗 {target_node.url}\n\n"
        f"{synthesis}"
    )

    return PageInspectResult(
        status="ok",
        page_title=target_node.title,
        page_url=target_node.url,
        breadcrumb=breadcrumb,
        block_count=block_count,
        extracted_text=extracted_text,
        synthesis=synthesis,
        reply_text=clean_math_and_markdown(reply),
    )


def archive_page_to_archive_index(
    page_query: str,
    notion_client: Optional[NotionAssistantClient] = None,
) -> Dict[str, Any]:
    """Relocate a Notion page/document into the Archive Index database."""
    from datetime import datetime, timezone
    notion = notion_client or NotionAssistantClient()
    inspect_res = inspect_page_content(page_query, notion_client=notion)

    if inspect_res.status != "ok":
        return {
            "status": "not_found",
            "reply_text": f"📦 *Archive Action*\n\nCould not find a page matching *'{page_query}'* to archive.",
        }

    # Locate Archive Index Database
    archive_db_id = None
    try:
        db_search = notion._request_with_retry(
            notion.client.search,
            query="Archive Index",
            filter={"property": "object", "value": "database"},
            page_size=1,
        )
        if db_search.get("results"):
            archive_db_id = db_search["results"][0].get("id")
    except Exception as search_err:
        logger.error("Failed to search Archive Index DB: %s", search_err)

    if not archive_db_id:
        # Fallback to general guidance if Archive Index database is inaccessible
        return suggest_page_archival(page_query, notion_client=notion)

    # 1. Fetch raw blocks from source page
    source_blocks = []
    source_page_id = None
    nodes = build_workspace_hierarchy_graph(notion_client=notion)
    for n in nodes.values():
        if n.title.lower() == inspect_res.page_title.lower():
            source_page_id = n.id
            break

    # 2. Build new row in Archive Index database
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_page_properties = {
        "Name": {
            "title": [
                {
                    "type": "text",
                    "text": {"content": inspect_res.page_title},
                }
            ]
        },
        "Type": {
            "select": {"name": "Doc"}
        },
        "Archived month": {
            "date": {"start": today_str}
        },
    }

    # Add child blocks if available
    child_blocks_payload = []
    if inspect_res.extracted_text:
        for line in inspect_res.extracted_text.split("\n")[:30]:
            if line.strip():
                child_blocks_payload.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": line[:2000]}}]
                    }
                })

    try:
        new_page = notion._request_with_retry(
            notion.client.pages.create,
            parent={"database_id": archive_db_id},
            properties=new_page_properties,
            children=child_blocks_payload[:100] if child_blocks_payload else None,
        )
        new_page_url = _extract_page_url(new_page)

        # Mark source page as archived in Notion
        if source_page_id and notion.client:
            try:
                notion._request_with_retry(
                    notion.client.pages.update,
                    page_id=source_page_id,
                    archived=True,
                )
            except Exception as arch_err:
                logger.warning("Could not mark source page %s archived: %s", source_page_id, arch_err)

        reply = (
            f"📦 *Page Archived Successfully!*\n\n"
            f"📄 *{inspect_res.page_title}* has been moved from *{inspect_res.breadcrumb}* into your **Archive Index**.\n"
            f"🔗 {new_page_url}\n\n"
            f"*(Archived on {today_str})*"
        )

        return {
            "status": "ok",
            "page_title": inspect_res.page_title,
            "page_url": new_page_url,
            "breadcrumb": "Archive > Archive Index",
            "reply_text": clean_math_and_markdown(reply),
        }

    except Exception as create_err:
        logger.error("Failed to create page in Archive Index DB: %s", create_err)
        return suggest_page_archival(page_query, notion_client=notion)


def suggest_page_archival(
    page_query: str,
    notion_client: Optional[NotionAssistantClient] = None,
) -> Dict[str, Any]:
    """Provide safe guidance to archive a Notion document."""
    notion = notion_client or NotionAssistantClient()
    inspect_res = inspect_page_content(page_query, notion_client=notion)

    if inspect_res.status != "ok":
        return {
            "status": "not_found",
            "reply_text": f"📦 *Archive Guidance*\n\nCould not find a page matching *'{page_query}'* to archive.",
        }

    reply = (
        f"📦 *Archive Suggestion*\n\n"
        f"Found document: *{inspect_res.page_title}*\n"
        f"📍 Current Location: *{inspect_res.breadcrumb}*\n"
        f"🔗 {inspect_res.page_url}\n\n"
        f"💡 *To archive this page:*\n"
        f"• Tell Ocean *'archive {inspect_res.page_title}'* to automatically move it to your Archive Index, or click the link above and select **Delete / Archive** in Notion."
    )

    return {
        "status": "ok",
        "page_title": inspect_res.page_title,
        "page_url": inspect_res.page_url,
        "breadcrumb": inspect_res.breadcrumb,
        "reply_text": clean_math_and_markdown(reply),
    }
