"""Semantic Second-Brain Search & Knowledge Retrieval Service for Ocean."""

import difflib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from app.notion_client import NotionAssistantClient, clean_math_and_markdown
from app.schemas import SearchQueryAnalysis, SearchResultItem
from app.telegram_client import TelegramAssistantClient
from app.whatsapp_client import WhatsAppAssistantClient
from app.workspace_service import (
    build_workspace_hierarchy_graph,
    explore_container,
    inspect_page_content,
    suggest_page_archival,
)

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def get_gemini_client():
    """Create and return a google-genai Client instance."""
    if genai is None:
        raise RuntimeError("google-genai library is not installed or available")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


SECOND_BRAIN_ANSWER_SYSTEM_INSTRUCTION = (
    "You are Ocean's elite second-brain knowledge synthesis engine.\n"
    "Your objective is to answer user inquiries accurately and insightfully based on their personal Notion workspace knowledge (Subjects, Research Resources, Tasks, Whiteboard notes, and Daily Logs).\n\n"
    "Strict Requirements:\n"
    "1. ACCURACY & GROUNDING: Ground your response directly in the provided workspace pages. Cite the exact titles and URLs of the pages.\n"
    "2. UNCONSTRAINED ESSENCE & DEPTH: Provide a thorough, direct explanation that answers what the user is asking without superficial brevity.\n"
    "3. NOTION CITATIONS: Include clickable Notion links in the format [Page Title](URL) or list them under a '📚 Sources in your Notion:' section.\n"
    "4. FORMATTING RULES:\n"
    "   - DO NOT use LaTeX math notation or dollar signs (write O(N), 10^9, N directly, NEVER $O(N)$ or $10^9$).\n"
    "   - DO NOT use double asterisks (**). Use clean standard markdown formatting.\n"
)


def _extract_page_title(page: Dict[str, Any]) -> str:
    """Extract plain text title from a Notion page."""
    props = page.get("properties", {})
    for _, prop_val in props.items():
        if isinstance(prop_val, dict) and prop_val.get("type") == "title":
            title_list = prop_val.get("title", [])
            if title_list:
                return "".join(t.get("plain_text", "") for t in title_list).strip()
    return "Untitled Page"


def _extract_page_url(page: Dict[str, Any]) -> str:
    """Extract clean Notion URL from page dictionary."""
    url = page.get("url", "")
    if not url and page.get("id"):
        clean_id = page["id"].replace("-", "")
        return f"https://www.notion.so/{clean_id}"
    return url


def _extract_page_snippet(page: Dict[str, Any], notion_client: NotionAssistantClient) -> str:
    """Extract summary property or first block contents from Notion page."""
    props = page.get("properties", {})
    # Check for Summary or Description property
    for prop_name, prop_val in props.items():
        if isinstance(prop_val, dict):
            prop_type = prop_val.get("type")
            if prop_type == "rich_text":
                texts = prop_val.get("rich_text", [])
                if texts:
                    return "".join(t.get("plain_text", "") for t in texts).strip()

    # If no rich_text property, fetch first 3 child blocks
    page_id = page.get("id")
    if page_id and notion_client.client:
        try:
            blocks_res = notion_client._request_with_retry(
                notion_client.client.blocks.children.list,
                block_id=page_id,
                page_size=5,
            )
            snippets = []
            for b in blocks_res.get("results", []):
                b_type = b.get("type", "")
                if b_type in ("paragraph", "callout", "heading_1", "heading_2", "heading_3", "bulleted_list_item"):
                    rich_texts = b.get(b_type, {}).get("rich_text", [])
                    t = "".join(r.get("plain_text", "") for r in rich_texts).strip()
                    if t:
                        snippets.append(t)
            if snippets:
                return " | ".join(snippets[:3])
        except Exception:
            pass

    return ""


def search_workspace_knowledge(
    query: str,
    domain_filter: Optional[str] = None,
    notion_client: Optional[NotionAssistantClient] = None,
    limit: int = 6,
) -> List[SearchResultItem]:
    """Search Notion across workspace search, Subjects, Resources, and Tasks."""
    notion = notion_client or NotionAssistantClient()
    client = notion.client
    if client is None:
        raise RuntimeError("Notion client is not initialized")

    results: List[SearchResultItem] = []
    seen_ids = set()

    # 1. Full Workspace Global Search via Notion Search API
    try:
        search_res = notion._request_with_retry(
            client.search,
            query=query,
            page_size=12,
        )
        for page in search_res.get("results", []):
            page_id = page.get("id")
            if not page_id or page_id in seen_ids:
                continue
            seen_ids.add(page_id)

            title = _extract_page_title(page)
            url = _extract_page_url(page)
            snippet = _extract_page_snippet(page, notion)

            # Determine category
            parent_type = page.get("parent", {}).get("type", "")
            cat = "Page"
            if parent_type == "database_id":
                db_id = page.get("parent", {}).get("database_id", "").replace("-", "")
                if notion.subjects_db_id and db_id == notion.subjects_db_id.replace("-", ""):
                    cat = "Subject"
                elif notion.resources_db_id and db_id == notion.resources_db_id.replace("-", ""):
                    cat = "Resource"
                elif notion.tasks_db_id and db_id == notion.tasks_db_id.replace("-", ""):
                    cat = "Task"

            results.append(SearchResultItem(
                title=title,
                url=url,
                category=cat,
                snippet=snippet,
                last_edited_time=page.get("last_edited_time"),
            ))
    except Exception as exc:
        logger.error("Notion global workspace search failed (%s)", exc)

    # 2. Query Subjects Database for high-relevance matches
    if notion.subjects_db_id and len(results) < limit:
        try:
            subj_res = notion._query_database(
                database_id=notion.subjects_db_id,
                page_size=20,
            )
            query_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower()))
            for page in subj_res.get("results", []):
                page_id = page.get("id")
                if not page_id or page_id in seen_ids:
                    continue
                title = _extract_page_title(page)
                title_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", title.lower()))
                if query_words.intersection(title_words):
                    seen_ids.add(page_id)
                    results.append(SearchResultItem(
                        title=title,
                        url=_extract_page_url(page),
                        category="Subject",
                        snippet=_extract_page_snippet(page, notion),
                        last_edited_time=page.get("last_edited_time"),
                    ))
        except Exception as exc:
            logger.error("Notion subjects database search failed (%s)", exc)

    # 3. Query Resources Database
    if notion.resources_db_id and len(results) < limit:
        try:
            res_res = notion._query_database(
                database_id=notion.resources_db_id,
                page_size=20,
            )
            query_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower()))
            for page in res_res.get("results", []):
                page_id = page.get("id")
                if not page_id or page_id in seen_ids:
                    continue
                title = _extract_page_title(page)
                title_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", title.lower()))
                if query_words.intersection(title_words):
                    seen_ids.add(page_id)
                    results.append(SearchResultItem(
                        title=title,
                        url=_extract_page_url(page),
                        category="Resource",
                        snippet=_extract_page_snippet(page, notion),
                        last_edited_time=page.get("last_edited_time"),
                    ))
        except Exception as exc:
            logger.error("Notion resources database search failed (%s)", exc)

    return results[:limit]


def answer_second_brain_question(
    query: str,
    search_results: List[SearchResultItem],
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Synthesize a grounded answer using retrieved Notion workspace items."""
    client = get_gemini_client()

    context_lines = []
    for idx, item in enumerate(search_results, 1):
        context_lines.append(
            f"[{idx}] {item.category}: {item.title}\n"
            f"    URL: {item.url}\n"
            f"    Snippet: {item.snippet or 'No additional preview'}"
        )

    retrieved_text = "\n\n".join(context_lines) if context_lines else "No directly matching Notion pages found."

    prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved Notion Workspace Context:\n{retrieved_text}\n\n"
        "Please provide a comprehensive, well-structured, grounded answer citing the matching Notion sources."
    )

    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_SEARCH_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SECOND_BRAIN_ANSWER_SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
        answer_text = response.text or ""
    except Exception as exc:
        logger.error("Gemini Second-Brain QA synthesis failed (%s)", exc)
        answer_text = ""

    # Clean LaTeX math and markdown asterisks
    clean_ans = clean_math_and_markdown(answer_text, for_whatsapp=True)

    # Format delivery message
    delivery_lines = [
        f"🧠 *Second-Brain Answer*\n",
        clean_ans,
    ]

    if search_results:
        delivery_lines.append("\n📚 *Sources in your Notion:*")
        for res in search_results[:4]:
            delivery_lines.append(f"• *{res.title}* ({res.category})\n  🔗 {res.url}")

    final_reply = "\n".join(delivery_lines).strip()

    return {
        "status": "ok",
        "query": query,
        "answer": clean_ans,
        "sources": [r.model_dump() for r in search_results],
        "reply_text": final_reply,
    }


def execute_second_brain_search_pipeline(
    query: str,
    search_analysis: Optional[SearchQueryAnalysis] = None,
    to_phone: Optional[str] = None,
    chat_id: Optional[str] = None,
    notion_client: Optional[NotionAssistantClient] = None,
    whatsapp_client: Optional[WhatsAppAssistantClient] = None,
    telegram_client: Optional[TelegramAssistantClient] = None,
) -> Dict[str, Any]:
    """Execute Second-Brain search, folder exploration, page inspection, or QA generation and deliver result."""
    notion = notion_client or NotionAssistantClient()
    whatsapp = whatsapp_client or WhatsAppAssistantClient()
    telegram = telegram_client or TelegramAssistantClient()

    search_type = search_analysis.search_type if search_analysis else "QUESTION"
    container_name = search_analysis.container_name if search_analysis else None
    page_name = search_analysis.page_name if search_analysis else None

    # Detect heuristic intent from raw query if search_type is generic
    q_lower = query.strip().lower()
    folder_keywords = ("what's in my ", "what is in my ", "what's in ", "what is in ", "list pages in ", "show pages in ", "notes in ")
    archive_keywords = ("archive ", "send down to archive", "move to archive", "send to archive", "send it down to archive")

    if search_type == "FOLDER_EXPLORE" or (any(kw in q_lower for kw in folder_keywords) and not page_name):
        target_folder = container_name
        if not target_folder:
            for kw in folder_keywords:
                if kw in q_lower:
                    target_folder = q_lower.split(kw, 1)[1].strip("? .")
                    break
        target_folder = target_folder or query
        explore_res = explore_container(target_folder, notion_client=notion)
        reply_text = explore_res.reply_text
        result_payload = {
            "status": explore_res.status,
            "type": "FOLDER_EXPLORE",
            "container": explore_res.container_title,
            "subpages": explore_res.subpages,
            "reply_text": reply_text,
        }

    elif search_type == "ARCHIVE_SUGGEST" or any(kw in q_lower for kw in archive_keywords):
        target_doc = page_name
        if not target_doc:
            for kw in ("archive ", "archive the ", "send down to archive ", "move to archive "):
                if kw in q_lower:
                    target_doc = q_lower.split(kw, 1)[1].strip("? .")
                    break
        target_doc = target_doc or query
        arch_res = suggest_page_archival(target_doc, notion_client=notion)
        reply_text = arch_res.get("reply_text", "")
        result_payload = {
            "status": arch_res.get("status", "ok"),
            "type": "ARCHIVE_SUGGEST",
            "reply_text": reply_text,
        }

    elif search_type == "PAGE_INSPECT" or ("budget" in q_lower or "finances" in q_lower or "what's in that " in q_lower or "tell me what's in " in q_lower):
        target_doc = page_name or query
        inspect_res = inspect_page_content(target_doc, user_question=query, notion_client=notion)
        reply_text = inspect_res.reply_text
        result_payload = {
            "status": inspect_res.status,
            "type": "PAGE_INSPECT",
            "page_title": inspect_res.page_title,
            "reply_text": reply_text,
        }

    else:
        # Standard Second-Brain Search & Grounded QA
        search_results = search_workspace_knowledge(query, notion_client=notion)
        synthesis = answer_second_brain_question(query, search_results)
        reply_text = synthesis.get("reply_text", "")
        result_payload = synthesis

    # Deliver via WhatsApp or Telegram
    if to_phone and reply_text:
        try:
            whatsapp.send_message(to=to_phone, text=reply_text, preview_url=True)
            logger.info("Sent WhatsApp Second-Brain response to %s", to_phone)
        except Exception as wa_err:
            logger.error("Failed to send WhatsApp Second-Brain response: %s", wa_err)

    if chat_id and reply_text:
        try:
            telegram.send_message(text=reply_text, chat_id=str(chat_id))
            logger.info("Sent Telegram Second-Brain response to %s", chat_id)
        except Exception as tg_err:
            logger.error("Failed to send Telegram Second-Brain response: %s", tg_err)

    return result_payload
