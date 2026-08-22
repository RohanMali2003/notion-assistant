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

from app.ai import generate_text, get_gemini_client, get_gemini_model
from app.config import settings
from app.matcher import entity_resolver
from app.memory import conversation_memory
from app.notion_client import NotionAssistantClient, clean_math_and_markdown
from app.notion_utils import (
    build_notion_block as _build_notion_block,
    create_bullet_block as _create_bullet_block,
    create_paragraph_block as _create_paragraph_block,
    create_todo_block as _create_todo_block,
    extract_page_title as _extract_page_title,
    extract_page_url as _extract_page_url,
)
from app.schemas import (
    DocumentAppendAnalysis,
    FolderExploreResult,
    PageInspectResult,
    SearchResultItem,
    WorkspaceEntryItem,
    WorkspaceIngestAnalysis,
    WorkspacePageNode,
)

logger = logging.getLogger("notion-assistant.workspace")

# --- In-Memory Dynamic Workspace Cache ---
_WORKSPACE_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "ttl_seconds": 300.0,  # 5-minute cache
    "nodes_by_id": {},
    "nodes_by_title": {},
}


def _index_initial_search_nodes(results: List[Dict[str, Any]]) -> Tuple[Dict[str, WorkspacePageNode], Dict[str, str]]:
    """Index top-level search results into workspace node mapping."""
    nodes_by_id: Dict[str, WorkspacePageNode] = {}
    nodes_by_title: Dict[str, str] = {}
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
    return nodes_by_id, nodes_by_title


def _discover_child_container_nodes(notion: NotionAssistantClient, nodes_by_id: Dict[str, WorkspacePageNode], nodes_by_title: Dict[str, str]) -> None:
    """Discover child pages and databases under container pages."""
    for node_id, node in list(nodes_by_id.items()):
        if node.parent_type not in ("workspace", "page_id", "block_id"):
            continue
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
                        db_url = f"https://app.notion.com/p/{b_id.replace('-', '')}" if b_id else ""
                        node.children_pages.append({
                            "id": b_id,
                            "title": child_title,
                            "url": db_url,
                            "type": "database",
                        })
                        db_node = nodes_by_id.get(b_id)
                        if not db_node:
                            db_node = WorkspacePageNode(
                                id=b_id,
                                title=child_title,
                                url=db_url,
                                parent_type="page_id",
                                parent_id=node_id,
                                is_container=True,
                            )
                            nodes_by_id[b_id] = db_node
                            nodes_by_title[child_title.lower()] = b_id

                        try:
                            db_rows = notion._query_database(database_id=b_id, page_size=50)
                            for r in db_rows.get("results", []):
                                r_id = r.get("id")
                                r_title = _extract_page_title(r)
                                r_url = _extract_page_url(r)
                                if r_id and r_title:
                                    db_node.children_pages.append({
                                        "id": r_id,
                                        "title": r_title,
                                        "url": r_url,
                                        "type": "doc",
                                    })
                                    if r_id not in nodes_by_id:
                                        nodes_by_id[r_id] = WorkspacePageNode(
                                            id=r_id,
                                            title=r_title,
                                            url=r_url,
                                            parent_type="database_id",
                                            parent_id=b_id,
                                            last_edited_time=r.get("last_edited_time"),
                                        )
                                        nodes_by_title[r_title.lower()] = r_id
                        except Exception as dbr_err:
                            logger.debug("Failed to query rows for child_database %s (%s): %s", child_title, b_id, dbr_err)
            node.is_container = has_children
        except Exception as b_err:
            logger.debug("Could not list blocks for node %s: %s", node_id, b_err)


def _compute_graph_breadcrumbs(nodes_by_id: Dict[str, WorkspacePageNode]) -> None:
    """Compute and attach hierarchical breadcrumbs for all indexed workspace nodes."""
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

    try:
        search_res = notion._request_with_retry(
            notion.client.search,
            page_size=100,
        )
        results = search_res.get("results", [])

        nodes_by_id, nodes_by_title = _index_initial_search_nodes(results)
        _discover_child_container_nodes(notion, nodes_by_id, nodes_by_title)
        _compute_graph_breadcrumbs(nodes_by_id)

        _WORKSPACE_CACHE["timestamp"] = now
        _WORKSPACE_CACHE["nodes_by_id"] = nodes_by_id
        _WORKSPACE_CACHE["nodes_by_title"] = nodes_by_title

        logger.info("Successfully indexed %d Notion workspace nodes", len(nodes_by_id))
        return nodes_by_id

    except Exception as exc:
        logger.error("Failed to build workspace hierarchy graph: %s", exc)
        return _WORKSPACE_CACHE.get("nodes_by_id", {})



def _normalize_title_text(text: str) -> str:
    """Normalize digits, punctuation, and common variations for fuzzy matching."""
    s = text.lower().strip()
    s = re.sub(r"\b1\b", "one", s)
    s = re.sub(r"\b2\b", "two", s)
    s = re.sub(r"\b3\b", "three", s)
    s = re.sub(r"\b4\b", "four", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _find_best_node_match(
    query: str,
    nodes: Dict[str, WorkspacePageNode],
    is_container_preferred: bool = False,
) -> Optional[WorkspacePageNode]:
    """Find the best matching WorkspacePageNode using 3-tier Entity Resolution Cascade."""
    if not query or not nodes:
        return None

    candidate_list = list(nodes.values())
    if is_container_preferred:
        filtered = [n for n in candidate_list if n.is_container]
        if filtered:
            candidate_list = filtered

    best_node, tier, score = entity_resolver.resolve_entity(
        query=query,
        candidates=candidate_list,
        key_fn=lambda n: n.title,
    )
    return best_node


def explore_container(
    container_query: str,
    notion_client: Optional[NotionAssistantClient] = None,
) -> FolderExploreResult:
    """Explore a container/folder page in Notion and list its child documents/subpages."""
    notion = notion_client or NotionAssistantClient()
    nodes = build_workspace_hierarchy_graph(notion_client=notion)

    target_node = _find_best_node_match(container_query, nodes, is_container_preferred=True)

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

    target_node = _find_best_node_match(page_query, nodes, is_container_preferred=False)

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
        question_prompt = user_question or "Summarize the key contents, numbers, and takeaways from this page."
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
        synthesis = generate_text(
            prompt=content_prompt,
            system_instruction=sys_inst,
            fallback_default="\n".join(block_lines[:15]),
        )
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
                child_blocks_payload.append(_create_paragraph_block(line.strip()))


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


# --- Ocean v3.0: In-Place Document Appending ---

def find_page_node_in_workspace(
    query: str,
    notion_client: Optional[NotionAssistantClient] = None,
    is_container_preferred: bool = False,
) -> Optional[WorkspacePageNode]:
    """Find the best matching WorkspacePageNode across the hierarchy graph with search API fallback."""
    notion = notion_client or NotionAssistantClient()
    nodes = build_workspace_hierarchy_graph(notion_client=notion)

    target_node = _find_best_node_match(query, nodes, is_container_preferred=is_container_preferred)

    if not target_node and notion.client:
        try:
            res = notion._request_with_retry(
                notion.client.search,
                query=query,
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

    return target_node


def _build_notion_block(content: str, block_type: str = "bulleted_list_item") -> Dict[str, Any]:
    """Construct Notion block JSON payload based on block_type."""
    clean_content = content.strip()

    if block_type == "to_do":
        return {
            "object": "block",
            "type": "to_do",
            "to_do": {
                "rich_text": [{"type": "text", "text": {"content": clean_content}}],
                "checked": False,
            },
        }
    elif block_type == "callout":
        return {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": clean_content}}],
                "icon": {"emoji": "💡"},
            },
        }
    elif block_type == "paragraph":
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": clean_content}}],
            },
        }
    else:
        # Default: bulleted_list_item
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": clean_content}}],
            },
        }


def add_entries_to_workspace_target(
    target_query: str,
    items: List[WorkspaceEntryItem],
    default_status: Optional[str] = None,
    block_type: str = "bulleted_list_item",
    notion_client: Optional[NotionAssistantClient] = None,
    sender_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Dynamically adds items/books to a Notion database (e.g. Reading List) or appends blocks to a Page."""
    notion = notion_client or NotionAssistantClient()

    if notion.client is None:
        return {
            "status": "error",
            "message": "Notion client not initialized.",
            "reply_text": "❌ Notion integration is not configured or connected.",
        }

    if not items:
        return {
            "status": "error",
            "message": "No items provided to add.",
            "reply_text": "⚠️ No items were provided to add to your workspace target.",
        }

    # 1. Resolve target node from workspace graph
    nodes = build_workspace_hierarchy_graph(notion_client=notion)
    target_node = find_page_node_in_workspace(target_query, notion_client=notion)

    # 2. Check if target is or contains a Database
    database_id = None
    target_title = target_query
    target_url = ""
    breadcrumb = target_query

    # Direct database search fallback if target_query mentions common databases like Reading List
    if not target_node or "reading" in target_query.lower() or "book" in target_query.lower():
        try:
            db_search = notion._request_with_retry(
                notion.client.search,
                query=target_query,
                filter={"property": "object", "value": "database"},
                page_size=3,
            )
            for db in db_search.get("results", []):
                db_title = _extract_page_title(db)
                if db_title and ("reading" in db_title.lower() or target_query.lower() in db_title.lower()):
                    database_id = db.get("id")
                    target_title = db_title
                    target_url = _extract_page_url(db)
                    breadcrumb = f"Media > {db_title}"
                    break
        except Exception as dbe:
            logger.debug("Database search fallback failed: %s", dbe)

    if target_node:
        target_title = target_node.title
        target_url = target_node.url
        breadcrumb = target_node.breadcrumb or target_node.title

        if not database_id:
            for child in target_node.children_pages:
                if child.get("type") == "database":
                    database_id = child.get("id")
                    break
            if not database_id and target_node.parent_type == "database_id":
                database_id = target_node.id

    # If still no database_id and target_node is None, try generic search fallback
    if not target_node and not database_id:
        try:
            res = notion._request_with_retry(
                notion.client.search,
                query=target_query,
                page_size=5,
            )
            for r in res.get("results", []):
                r_obj = r.get("object")
                r_title = _extract_page_title(r)
                if r_obj == "database":
                    database_id = r.get("id")
                    target_title = r_title or target_query
                    target_url = _extract_page_url(r)
                    breadcrumb = target_title
                    break
                elif r_obj == "page" and not target_node:
                    target_node = WorkspacePageNode(
                        id=r.get("id"),
                        title=r_title or target_query,
                        url=_extract_page_url(r),
                        breadcrumb=r_title or target_query,
                    )
        except Exception as exc:
            logger.warning("Target search fallback failed: %s", exc)

    if not target_node and not database_id:
        return {
            "status": "not_found",
            "message": f"Target '{target_query}' not found in workspace hierarchy.",
            "reply_text": f"❓ Could not locate document **'{target_query}'** in your Notion workspace graph.",
        }

    affected_items = []

    # CASE A: TARGET IS A DATABASE (e.g. Reading List, Project Ideas, etc.)
    if database_id:
        try:
            from app.notion_schema_engine import schema_engine
            schema = schema_engine.get_schema(database_id=database_id, notion_client=notion)

            for item in items:
                item_dict = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else getattr(item, "__dict__", {}))

                if schema:
                    props_payload = schema.build_page_properties(item_dict, default_status=default_status or "Want to Read")
                else:
                    props_payload = {
                        "Name": {
                            "title": [{"type": "text", "text": {"content": item.title}}]
                        }
                    }

                created_page = notion._request_with_retry(
                    notion.client.pages.create,
                    parent={"database_id": database_id},
                    properties=props_payload,
                )
                page_id = created_page.get("id", "")
                page_url = _extract_page_url(created_page)

                affected_items.append({
                    "id": page_id,
                    "title": item.title,
                    "url": page_url,
                    "type": "database_row",
                })

            if sender_id:
                conversation_memory.record_mutation(
                    sender_id=sender_id,
                    action_type="WORKSPACE_INGEST",
                    target_title=target_title,
                    affected_items=affected_items,
                    rollback_data={"database_id": database_id, "target_type": "database"},
                    summary=f"Added {len(affected_items)} item(s) to {target_title}",
                )

            icon_emoji = "📖" if "reading" in target_title.lower() or "book" in target_title.lower() else "✨"
            lines = []
            for it in affected_items:
                status_str = f" (Status: *{it['status']}*)" if it.get("status") else ""
                lines.append(f"• {icon_emoji} **[{it['title']}]({it['url']})**{status_str}")

            header_icon = "📚" if "reading" in target_title.lower() or "book" in target_title.lower() else "📂"
            reply = (
                f"{header_icon} *Added to {target_title}!*\n\n"
                + "\n".join(lines)
                + f"\n\n📍 Location: *{breadcrumb}*"
            )

            return {
                "status": "ok",
                "target_type": "database",
                "target_title": target_title,
                "target_url": target_url,
                "breadcrumb": breadcrumb,
                "affected_items": affected_items,
                "reply_text": clean_math_and_markdown(reply),
            }

        except Exception as exc:
            logger.error("Failed to add rows to database %s: %s", database_id, exc, exc_info=True)
            return {
                "status": "error",
                "message": str(exc),
                "reply_text": f"❌ Failed to add entries to **'{target_title}'**: {exc}",
            }

    # CASE B: TARGET IS A STANDARD DOCUMENT PAGE
    else:
        page_id = target_node.id
        page_title = target_node.title
        page_url = target_node.url or f"https://notion.so/{page_id.replace('-', '')}"
        breadcrumb = target_node.breadcrumb or page_title

        blocks_payload = [_build_notion_block(item.title, block_type=block_type) for item in items]
        try:
            notion._request_with_retry(
                notion.client.blocks.children.append,
                block_id=page_id,
                children=blocks_payload,
            )

            for item in items:
                affected_items.append({
                    "title": item.title,
                    "type": "block",
                })

            if sender_id:
                conversation_memory.record_mutation(
                    sender_id=sender_id,
                    action_type="WORKSPACE_INGEST",
                    target_title=page_title,
                    affected_items=affected_items,
                    rollback_data={"page_id": page_id, "target_type": "page"},
                    summary=f"Appended {len(items)} block(s) to {page_title}",
                )

            icon_emoji = "•" if block_type == "bulleted_list_item" else ("☑️" if block_type == "to_do" else "📝")
            item_lines = [f"{icon_emoji} {it['title']}" for it in affected_items]
            reply = (
                f"📝 *Appended to Note!*\n\n"
                f"📌 **[{page_title}]({page_url})**\n"
                f"📍 Location: *{breadcrumb}*\n\n"
                f"✨ *Added item(s):*\n"
                + "\n".join(item_lines)
            )

            return {
                "status": "ok",
                "target_type": "page",
                "target_title": page_title,
                "target_url": page_url,
                "breadcrumb": breadcrumb,
                "affected_items": affected_items,
                "reply_text": clean_math_and_markdown(reply),
            }
        except Exception as exc:
            logger.error("Failed to append blocks to page %s: %s", page_title, exc, exc_info=True)
            return {
                "status": "error",
                "message": str(exc),
                "reply_text": f"❌ Failed to append content to **'{page_title}'**: {exc}",
            }


def append_blocks_to_document(
    target_title_or_id: str,
    content: str,
    block_type: str = "bulleted_list_item",
    notion_client: Optional[NotionAssistantClient] = None,
    sender_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Appends new block (or adds database rows) directly into an existing Notion document or database."""
    lines = [line.strip().lstrip("1234567890.-•* ").strip() for line in content.split("\n") if line.strip()]
    items = [WorkspaceEntryItem(title=l) for l in lines] if lines else [WorkspaceEntryItem(title=content.strip())]
    res = add_entries_to_workspace_target(
        target_query=target_title_or_id,
        items=items,
        block_type=block_type,
        notion_client=notion_client,
        sender_id=sender_id,
    )
    if res.get("status") == "ok":
        res["page_title"] = res.get("target_title", target_title_or_id)
        res["page_url"] = res.get("target_url", "")
        res["content_appended"] = content.strip()
        res["block_type"] = block_type
    return res

