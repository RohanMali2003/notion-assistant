"""Reporter that scans Notion databases for duplicates and writes a structured review page in Notion."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
from app.duplicate_detector import DuplicateCluster, DuplicateItem, find_duplicate_clusters
from app.notion_client import NotionAssistantClient
from app.tag_directory import find_tag_reclassification_suggestions

logger = logging.getLogger(__name__)

CLEANUP_PAGE_TITLE = "🧹 Notion Cleanup & Duplicate Review"


def _extract_page_title(page: Dict[str, Any]) -> str:
    """Extract title from a Notion page properties object."""
    props = page.get("properties", {})
    for _, prop_val in props.items():
        if isinstance(prop_val, dict) and prop_val.get("type") == "title":
            title_list = prop_val.get("title", [])
            if title_list:
                return "".join(t.get("plain_text", "") for t in title_list).strip()
    return "Untitled Page"


def _extract_page_url(page: Dict[str, Any]) -> str:
    """Extract or construct full web URL for a Notion page."""
    url = page.get("url")
    if url:
        return url
    page_id = page.get("id", "")
    clean_id = str(page_id).replace("-", "")
    return f"https://www.notion.so/{clean_id}"


def _extract_status(page: Dict[str, Any]) -> Optional[str]:
    """Extract Status property from a page."""
    props = page.get("properties", {})
    for _, prop_val in props.items():
        if isinstance(prop_val, dict) and prop_val.get("type") == "status":
            status_obj = prop_val.get("status")
            if isinstance(status_obj, dict):
                return status_obj.get("name")
    return None


def _extract_due_date(page: Dict[str, Any]) -> Optional[str]:
    """Extract Due date property from a page."""
    props = page.get("properties", {})
    for _, prop_val in props.items():
        if isinstance(prop_val, dict) and prop_val.get("type") == "date":
            date_obj = prop_val.get("date")
            if isinstance(date_obj, dict):
                return date_obj.get("start")
    return None


def _extract_tags(page: Dict[str, Any]) -> List[str]:
    """Extract Tags multi-select property from a page."""
    props = page.get("properties", {})
    for _, prop_val in props.items():
        if isinstance(prop_val, dict) and prop_val.get("type") == "multi_select":
            ms_list = prop_val.get("multi_select", [])
            return [item.get("name", "") for item in ms_list if isinstance(item, dict) and item.get("name")]
    return []


class NotionCleanupReporter:
    """Orchestrates multi-database duplicate scanning and Notion cleanup page generation."""

    def __init__(self, notion_client: Optional[NotionAssistantClient] = None):
        self.notion = notion_client or NotionAssistantClient()

    def fetch_all_subjects(self) -> List[DuplicateItem]:
        """Fetch all subjects from Notion Subjects database."""
        db_id = self.notion.subjects_db_id
        if not db_id:
            return []

        items: List[DuplicateItem] = []
        try:
            res = self.notion._query_database(database_id=db_id, page_size=100)
            for page in res.get("results", []):
                items.append(
                    DuplicateItem(
                        id=page.get("id", ""),
                        title=_extract_page_title(page),
                        url=_extract_page_url(page),
                        created_time=page.get("created_time"),
                        raw_props=page.get("properties", {}),
                    )
                )
        except Exception as exc:
            logger.error("Failed to fetch subjects for duplicate scan: %s", exc)

        return items

    def fetch_all_tasks(self) -> List[DuplicateItem]:
        """Fetch active/all tasks from Notion Tasks database."""
        db_id = self.notion.tasks_db_id
        if not db_id:
            return []

        items: List[DuplicateItem] = []
        try:
            res = self.notion._query_database(database_id=db_id, page_size=100)
            for page in res.get("results", []):
                items.append(
                    DuplicateItem(
                        id=page.get("id", ""),
                        title=_extract_page_title(page),
                        url=_extract_page_url(page),
                        created_time=page.get("created_time"),
                        status=_extract_status(page),
                        due_date=_extract_due_date(page),
                        tags=_extract_tags(page),
                        raw_props=page.get("properties", {}),
                    )
                )
        except Exception as exc:
            logger.error("Failed to fetch tasks for duplicate scan: %s", exc)

        return items

    def fetch_all_resources(self) -> List[DuplicateItem]:
        """Fetch all resources from Notion Resources database."""
        db_id = self.notion.resources_db_id
        if not db_id:
            return []

        items: List[DuplicateItem] = []
        try:
            res = self.notion._query_database(database_id=db_id, page_size=100)
            for page in res.get("results", []):
                props = page.get("properties", {})
                url_val = props.get("URL", {}).get("url") or ""
                items.append(
                    DuplicateItem(
                        id=page.get("id", ""),
                        title=_extract_page_title(page),
                        url=url_val or _extract_page_url(page),
                        created_time=page.get("created_time"),
                        raw_props=props,
                    )
                )
        except Exception as exc:
            logger.error("Failed to fetch resources for duplicate scan: %s", exc)

        return items

    def find_all_duplicates(self) -> Dict[str, Any]:
        """Run duplicate detection and tag optimization audit across Notion databases."""
        subjects = self.fetch_all_subjects()
        tasks = self.fetch_all_tasks()
        resources = self.fetch_all_resources()

        subject_clusters = find_duplicate_clusters(subjects, category="Subject", threshold=0.68)
        task_clusters = find_duplicate_clusters(tasks, category="Task", threshold=0.75)
        resource_clusters = find_duplicate_clusters(resources, category="Resource", threshold=0.80)

        # Audit items tagged 'Miscellaneous' or generic tags for re-tagging suggestions
        all_items_dicts = []
        for item in tasks + resources:
            current_tag = item.tags[0] if item.tags else "Miscellaneous"
            all_items_dicts.append({
                "id": item.id,
                "title": item.title,
                "url": item.url,
                "current_tag": current_tag,
                "text": f"{item.title} {item.status or ''}",
            })
        tag_suggestions = find_tag_reclassification_suggestions(all_items_dicts)

        return {
            "subjects": subject_clusters,
            "tasks": task_clusters,
            "resources": resource_clusters,
            "tag_suggestions": tag_suggestions,
        }

    def find_or_create_cleanup_page(self) -> Tuple[str, str]:
        """Locate existing cleanup page or create a new one. Returns (page_id, page_url)."""
        client = self.notion.client
        if client is None:
            raise RuntimeError("Notion client is not initialized")

        # 1. Search for existing page titled CLEANUP_PAGE_TITLE
        try:
            search_res = self.notion._request_with_retry(
                client.search,
                query=CLEANUP_PAGE_TITLE,
                filter={"value": "page", "property": "object"},
            )
            for result in search_res.get("results", []):
                page_title = _extract_page_title(result)
                if CLEANUP_PAGE_TITLE in page_title:
                    page_id = result.get("id", "")
                    page_url = _extract_page_url(result)
                    logger.info("Found existing cleanup page '%s' (%s)", CLEANUP_PAGE_TITLE, page_id)
                    return page_id, page_url
        except Exception as exc:
            logger.warning("Search for cleanup page failed (%s). Creating new page.", exc)

        # 2. If not found, create new page under parent database / workspace
        parent_id = self.notion.tasks_db_id or self.notion.subjects_db_id
        parent_obj = {"database_id": parent_id} if parent_id else {"page_id": parent_id}

        # First try creating as a page with parent page or database
        try:
            new_page = self.notion._request_with_retry(
                client.pages.create,
                parent={"database_id": self.notion.tasks_db_id},
                properties={
                    "Task name": {"title": [{"text": {"content": CLEANUP_PAGE_TITLE}}]},
                    "Tags": {"multi_select": [{"name": "Miscellaneous"}]},
                },
            )
            page_id = new_page.get("id", "")
            page_url = _extract_page_url(new_page)
            logger.info("Created new cleanup page '%s' in Tasks DB (%s)", CLEANUP_PAGE_TITLE, page_id)
            return page_id, page_url
        except Exception as create_err:
            logger.warning("Could not create in Tasks DB (%s). Attempting generic page create.", create_err)
            new_page = self.notion._request_with_retry(
                client.pages.create,
                parent=parent_obj,
                properties={
                    "title": [{"text": {"content": CLEANUP_PAGE_TITLE}}]
                },
            )
            page_id = new_page.get("id", "")
            page_url = _extract_page_url(new_page)
            return page_id, page_url

    def build_report_blocks(self, audit_results: Dict[str, List[DuplicateCluster]]) -> List[Dict[str, Any]]:
        """Construct rich Notion block layout for the duplicate audit report."""
        blocks: List[Dict[str, Any]] = []

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subj_clusters = audit_results.get("subjects", [])
        task_clusters = audit_results.get("tasks", [])
        res_clusters = audit_results.get("resources", [])

        total_clusters = len(subj_clusters) + len(task_clusters) + len(res_clusters)

        # 1. Header Callout
        if total_clusters == 0:
            callout_text = f"🎉 Scan Completed at {now_str}\nZero duplicates found! Your Notion workspace is clean and organized."
            callout_emoji = "✨"
        else:
            callout_text = (
                f"🕒 Last Scanned: {now_str}\n"
                f"Found {total_clusters} potential duplicate cluster(s) across your databases.\n"
                f"• {len(subj_clusters)} Duplicate Subject group(s)\n"
                f"• {len(task_clusters)} Duplicate Task group(s)\n"
                f"• {len(res_clusters)} Duplicate Resource group(s)\n\n"
                "💡 Note: Nothing has been deleted. Click the links below to review each item and decide whether to delete or keep."
            )
            callout_emoji = "🧹"

        blocks.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "icon": {"type": "emoji", "emoji": callout_emoji},
                "rich_text": [{"type": "text", "text": {"content": callout_text}}],
            },
        })

        if total_clusters == 0:
            return blocks

        # 2. Duplicate Subjects Section
        if subj_clusters:
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"🏛️ Duplicate Subjects ({len(subj_clusters)})"}}]},
            })

            for idx, cluster in enumerate(subj_clusters, 1):
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": f"Cluster #{idx}: {cluster.match_reason}"}}],
                    },
                })
                for item in cluster.items:
                    created_date = (item.created_time or "")[:10]
                    blocks.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                {"type": "text", "text": {"content": item.title, "link": {"url": item.url}}, "annotations": {"bold": True}},
                                {"type": "text", "text": {"content": f" (Created: {created_date})" if created_date else ""}},
                            ],
                        },
                    })
                if cluster.recommended_action:
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": "👉 Recommended Action: "}, "annotations": {"italic": True}},
                                {"type": "text", "text": {"content": cluster.recommended_action}},
                            ],
                        },
                    })

        # 3. Duplicate Tasks Section
        if task_clusters:
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"📋 Duplicate Tasks ({len(task_clusters)})"}}]},
            })

            for idx, cluster in enumerate(task_clusters, 1):
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": f"Task Cluster #{idx}: {cluster.match_reason}"}}],
                    },
                })
                for item in cluster.items:
                    status_str = f" | Status: {item.status}" if item.status else ""
                    due_str = f" | Due: {item.due_date}" if item.due_date else ""
                    created_date = (item.created_time or "")[:10]
                    blocks.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                {"type": "text", "text": {"content": item.title, "link": {"url": item.url}}, "annotations": {"bold": True}},
                                {"type": "text", "text": {"content": f"{status_str}{due_str} (Created: {created_date})"}},
                            ],
                        },
                    })
                if cluster.recommended_action:
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": "👉 Recommended Action: "}, "annotations": {"italic": True}},
                                {"type": "text", "text": {"content": cluster.recommended_action}},
                            ],
                        },
                    })

        # 4. Duplicate Resources Section
        if res_clusters:
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"📚 Duplicate Resources ({len(res_clusters)})"}}]},
            })

            for idx, cluster in enumerate(res_clusters, 1):
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": f"Resource Cluster #{idx}: {cluster.match_reason}"}}],
                    },
                })
                for item in cluster.items:
                    blocks.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                {"type": "text", "text": {"content": item.title, "link": {"url": item.url}}, "annotations": {"bold": True}},
                                {"type": "text", "text": {"content": f" — {item.url}" if item.url else ""}},
                            ],
                        },
                    })

        # 5. Tag Review & Optimization Suggestions
        tag_suggestions = audit_results.get("tag_suggestions", [])
        if tag_suggestions:
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"🏷️ Tag Optimization & Suggestions ({len(tag_suggestions)})"}}]},
            })
            for sug in tag_suggestions[:10]:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {"type": "text", "text": {"content": sug["title"], "link": {"url": sug["url"]}}, "annotations": {"bold": True}},
                            {"type": "text", "text": {"content": f" — Suggest moving from {sug['current_tag']} ➔ "}},
                            {"type": "text", "text": {"content": sug["suggested_tag"]}, "annotations": {"bold": True, "code": True}},
                            {"type": "text", "text": {"content": f" ({sug['reason']})"}, "annotations": {"italic": True}},
                        ],
                    },
                })

        return blocks

    def update_cleanup_report_page(self) -> Dict[str, Any]:
        """Scan Notion databases, identify duplicates, and write/refresh the Notion Cleanup page."""
        audit_results = self.find_all_duplicates()
        page_id, page_url = self.find_or_create_cleanup_page()

        client = self.notion.client
        if client is None:
            raise RuntimeError("Notion client is not initialized")

        # 1. Clear existing block children on the page
        try:
            children_res = self.notion._request_with_retry(
                client.blocks.children.list,
                block_id=page_id,
                page_size=100,
            )
            for child in children_res.get("results", []):
                child_id = child.get("id")
                if child_id:
                    try:
                        self.notion._request_with_retry(client.blocks.delete, block_id=child_id)
                    except Exception as del_err:
                        logger.debug("Could not delete child block %s: %s", child_id, del_err)
        except Exception as list_err:
            logger.warning("Could not list children of cleanup page: %s", list_err)

        # 2. Append new report blocks
        blocks = self.build_report_blocks(audit_results)
        try:
            self.notion._request_with_retry(
                client.blocks.children.append,
                block_id=page_id,
                children=blocks,
            )
            logger.info("Successfully updated cleanup page '%s' with %d blocks", CLEANUP_PAGE_TITLE, len(blocks))
        except Exception as append_err:
            logger.error("Failed to append report blocks to cleanup page: %s", append_err)

        total_subj = len(audit_results.get("subjects", []))
        total_tasks = len(audit_results.get("tasks", []))
        total_res = len(audit_results.get("resources", []))

        return {
            "status": "ok",
            "page_id": page_id,
            "page_url": page_url,
            "duplicate_subjects": total_subj,
            "duplicate_tasks": total_tasks,
            "duplicate_resources": total_res,
            "total_duplicate_clusters": total_subj + total_tasks + total_res,
            "audit_results": audit_results,
        }
