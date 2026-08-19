import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from notion_client import Client, APIResponseError
except ImportError:
    Client = None
    APIResponseError = None

from app.schemas import LEARNING_TAG, ReminderItem

logger = logging.getLogger(__name__)


class NotionValidationError(ValueError):
    """Raised when Notion API returns a 400 validation error."""

    def __init__(self, message: str, property_name: Optional[str] = None):
        super().__init__(message)
        self.property_name = property_name


class TaskDict(dict):
    """Dictionary representing a task, supporting both dict indexing and attribute access."""

    def __getattr__(self, item: str) -> Any:
        if item in self:
            return self[item]
        raise AttributeError(f"'TaskDict' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class NotionAssistantClient:
    """Thin wrapper around notion-client for Notion API interactions."""

    def __init__(
        self,
        token: Optional[str] = None,
        database_id: Optional[str] = None,
        tasks_db_id: Optional[str] = None,
        substack_db_id: Optional[str] = None,
        ramblings_db_id: Optional[str] = None,
        daily_logs_db_id: Optional[str] = None,
        subjects_db_id: Optional[str] = None,
        resources_db_id: Optional[str] = None,
        leetcode_log_db_id: Optional[str] = None,
    ):
        try:
            from app.config import settings
        except Exception:
            settings = None

        self.token = (
            token
            or (getattr(settings, "NOTION_TOKEN", None) if settings else None)
            or os.getenv("NOTION_TOKEN")
            or os.getenv("NOTION_API_KEY")
        )
        if tasks_db_id is not None:
            tasks_id = tasks_db_id
        elif database_id is not None:
            tasks_id = database_id
        else:
            tasks_id = (
                (getattr(settings, "NOTION_TASKS_DB_ID", None) if settings else None)
                or os.getenv("NOTION_TASKS_DB_ID")
                or (getattr(settings, "NOTION_DATABASE_ID", None) if settings else None)
                or os.getenv("NOTION_DATABASE_ID")
            )

        self.database_id = tasks_id
        self.tasks_db_id = tasks_id

        if substack_db_id is not None:
            self.substack_db_id = substack_db_id
        else:
            self.substack_db_id = getattr(settings, "NOTION_SUBSTACK_ID", None) or os.getenv("NOTION_SUBSTACK_ID", "")

        if ramblings_db_id is not None:
            self.ramblings_db_id = ramblings_db_id
        else:
            self.ramblings_db_id = getattr(settings, "NOTION_RAMBLINGS_ID", None) or os.getenv("NOTION_RAMBLINGS_ID", "")

        if daily_logs_db_id is not None:
            self.daily_logs_db_id = daily_logs_db_id
        else:
            self.daily_logs_db_id = getattr(settings, "NOTION_DAILY_LOGS_ID", None) or os.getenv("NOTION_DAILY_LOGS_ID", "")

        if subjects_db_id is not None:
            self.subjects_db_id = subjects_db_id
        else:
            self.subjects_db_id = getattr(settings, "NOTION_SUBJECTS_DB_ID", None) or os.getenv("NOTION_SUBJECTS_DB_ID", "")

        if resources_db_id is not None:
            self.resources_db_id = resources_db_id
        else:
            self.resources_db_id = getattr(settings, "NOTION_RESOURCES_DB_ID", None) or os.getenv("NOTION_RESOURCES_DB_ID", "")

        if leetcode_log_db_id is not None:
            self.leetcode_log_db_id = leetcode_log_db_id
        else:
            self.leetcode_log_db_id = (
                getattr(settings, "NOTION_LEETCODE_LOG_DB_ID", None)
                or os.getenv("NOTION_LEETCODE_LOG_DB_ID")
                or os.getenv("NOTION_LEETCODE_DB_ID", "")
            )

        self._db_props_cache: Dict[str, Dict[str, str]] = {}

        if Client is not None and self.token:
            self.client = Client(auth=self.token, notion_version="2022-06-28")
        else:
            self.client = None

    def _extract_offending_property(self, exc: Exception) -> Optional[str]:
        raw_msgs = []
        if hasattr(exc, "body") and isinstance(exc.body, dict):
            raw_msgs.append(str(exc.body.get("message", "")))
            if "notes" in exc.body:
                raw_msgs.append(str(exc.body.get("notes")))
        if hasattr(exc, "message"):
            raw_msgs.append(str(exc.message))
        raw_msgs.append(str(exc))

        full_text = " ".join(raw_msgs)

        match = re.search(
            r"properties[\.\[\'\"]+(.*?)(?:[\'\"\]]|(?=\.(?:date|title|rich_text|select|status|number|checkbox|multi_select|type))|\s+should|\s+is|\s+was|\s+failed|$)",
            full_text,
            re.IGNORECASE,
        )
        if match:
            prop = match.group(1).strip(" .['\"]")
            if prop and prop.lower() not in ("type", "date", "title", "rich_text", "select", "status", "number", "checkbox"):
                return prop

        match = re.search(r"property [\'\"]([A-Za-z0-9_\s]+?)[\'\"]", full_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        match = re.search(r"property with name or id:\s*([A-Za-z0-9_\s]+)", full_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        for known_prop in ["Due date", "Due Date", "Description", "Priority", "Tag", "Status", "Name", "Title"]:
            if known_prop in full_text:
                return known_prop

        return None

    def _request_with_retry(self, func, *args, **kwargs):
        if self.client is None:
            return {}

        max_attempts = 3
        base_delay = 0.1

        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                status_code = getattr(exc, "status", None) or getattr(exc, "code", None) or getattr(exc, "status_code", None)
                code_str = str(getattr(exc, "code", "")).lower()
                msg_str = str(exc).lower()

                is_429 = (
                    status_code == 429
                    or "429" in msg_str
                    or "rate_limited" in code_str
                    or "rate limit" in msg_str
                )

                if is_429:
                    if attempt < max_attempts:
                        sleep_time = base_delay * (2 ** (attempt - 1))
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise exc

                is_400 = (
                    status_code == 400
                    or "400" in msg_str
                    or "validation_error" in code_str
                    or "invalid" in msg_str
                )

                if is_400:
                    offending_prop = self._extract_offending_property(exc)
                    if offending_prop:
                        err_msg = f"Notion API 400 Validation Error for property '{offending_prop}': {exc}"
                        raise NotionValidationError(err_msg, property_name=offending_prop) from exc
                    else:
                        err_msg = f"Notion API 400 Validation Error: {exc}"
                        raise NotionValidationError(err_msg) from exc

                raise exc

    def _get_db_properties_schema(self, database_id: Optional[str] = None) -> Dict[str, str]:
        """Returns dict of property_name -> property_type for target database."""
        target_db = database_id or self.database_id
        if not target_db:
            return {}
        if not hasattr(self, "_db_props_cache") or self._db_props_cache is None:
            self._db_props_cache = {}
        if target_db in self._db_props_cache:
            return self._db_props_cache[target_db]
        if self.client is None:
            return {}
        try:
            db = self.client.databases.retrieve(target_db)
            if not isinstance(db, dict):
                return {}
            props = db.get("properties")
            if not isinstance(props, dict):
                return {}
            schema = {name: info.get("type") for name, info in props.items() if isinstance(info, dict)}
            self._db_props_cache[target_db] = schema
            return schema
        except Exception:
            return {}

    def _build_mind_blocks(self, core_thesis: Optional[str], content: str) -> List[Dict[str, Any]]:
        """Constructs child paragraph blocks: 1st block is the core thesis, followed by full text paragraphs."""
        blocks: List[Dict[str, Any]] = []

        def make_paragraph_block(text_content: str) -> Dict[str, Any]:
            return {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": text_content}
                        }
                    ]
                }
            }

        # 1. First block: One-sentence core thesis
        if core_thesis and core_thesis.strip():
            thesis_clean = core_thesis.strip()
            for i in range(0, len(thesis_clean), 2000):
                blocks.append(make_paragraph_block(thesis_clean[i:i + 2000]))

        # 2. Following blocks: Full text content
        if content and content.strip():
            lines = content.strip().split("\n")
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    blocks.append(make_paragraph_block(""))
                    continue
                for i in range(0, len(line_str), 2000):
                    blocks.append(make_paragraph_block(line_str[i:i + 2000]))

        if not blocks:
            blocks.append(make_paragraph_block(""))

        return blocks

    def create_mind_entry(
        self,
        entry_type: str,
        title: str,
        content: str,
        core_thesis: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Creates a new row in the appropriate MIND Notion database (Substack, Ramblings, or Daily Logs)

        and appends child blocks with the core thesis and full body text.
        """
        normalized_type = entry_type.upper()
        if normalized_type in ("SUBSTACK_DRAFT", "DRAFT_SUBSTACK", "SUBSTACK"):
            target_db_id = self.substack_db_id
            destination_type = "DRAFT_SUBSTACK"
        elif normalized_type in ("RAMBLING", "RAMBLINGS", "THOUGHT", "BRAIN_DUMP"):
            target_db_id = self.ramblings_db_id
            destination_type = "RAMBLING"
        elif normalized_type in ("DAILY_LOG", "DAILY_LOGS", "LOG", "REFLECTION"):
            target_db_id = self.daily_logs_db_id
            destination_type = "DAILY_LOG"
        else:
            target_db_id = self.daily_logs_db_id
            destination_type = "DAILY_LOG"

        if not target_db_id:
            raise ValueError(
                f"Notion database ID for MIND entry type '{destination_type}' is not configured."
            )

        if self.tasks_db_id and target_db_id == self.tasks_db_id:
            raise ValueError(
                f"Destination database for MIND entry type '{destination_type}' matches NOTION_TASKS_DB_ID. "
                "Mind entries must NOT be written to the tasks database."
            )

        schema = self._get_db_properties_schema(target_db_id)

        # 1. Determine Title property key
        title_key = "Name"
        for k, v in schema.items():
            if v == "title":
                title_key = k
                break
        else:
            for candidate in ["Title", "Name", "Topic", "Log", "Entry"]:
                if candidate in schema:
                    title_key = candidate
                    break

        properties: Dict[str, Any] = {
            title_key: {
                "title": [
                    {"text": {"content": title or "Untitled"}}
                ]
            }
        }

        # 2. Destination-specific properties
        if destination_type == "DRAFT_SUBSTACK":
            # Set Status to "Idea"
            if "Status" in schema:
                status_type = schema.get("Status")
                if status_type == "select":
                    properties["Status"] = {"select": {"name": "Idea"}}
                else:
                    properties["Status"] = {"status": {"name": "Idea"}}
            else:
                properties["Status"] = {"status": {"name": "Idea"}}

        elif destination_type == "DAILY_LOG":
            # Set Date property to today's date (YYYY-MM-DD)
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            date_key = "Date"
            for k, v in schema.items():
                if v == "date":
                    date_key = k
                    break
            else:
                for candidate in ["Date", "Log Date", "Created Date", "Date of Log", "Due date", "Due Date"]:
                    if candidate in schema:
                        date_key = candidate
                        break
            properties[date_key] = {"date": {"start": today_str}}

        # Optional tags if schema supports multi_select or select
        if tags:
            for tag_candidate in ["Tags", "Tag", "Category", "Topics"]:
                if tag_candidate in schema:
                    prop_type = schema[tag_candidate]
                    if prop_type == "multi_select":
                        properties[tag_candidate] = {"multi_select": [{"name": t} for t in tags]}
                    elif prop_type == "select":
                        properties[tag_candidate] = {"select": {"name": tags[0]}}
                    break

        # 3. Construct child blocks (1st block = core thesis, following = full text)
        thesis = core_thesis
        if not thesis and content:
            thesis = content.strip().split(".")[0].strip()
            if thesis:
                thesis += "."

        children_blocks = self._build_mind_blocks(thesis, content)

        return self._request_with_retry(
            self.client.pages.create,
            parent={"database_id": target_db_id},
            properties=properties,
            children=children_blocks,
        )

    def create_task(
        self,
        title: str,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
        due_date: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a page in NOTION_TASKS_DB_ID with Status=Not started."""
        schema = self._get_db_properties_schema()

        # Determine Title property key ('Task name', 'Name', 'Title', or any title type)
        title_key = "Name"
        if "Task name" in schema:
            title_key = "Task name"
        elif "Title" in schema:
            title_key = "Title"
        elif "Name" in schema:
            title_key = "Name"
        else:
            for k, v in schema.items():
                if v == "title":
                    title_key = k
                    break

        properties: Dict[str, Any] = {
            title_key: {
                "title": [
                    {"text": {"content": title}}
                ]
            },
            "Status": {
                "status": {
                    "name": "Not started"
                }
            }
        }
        if priority:
            priority_key = "Priority" if (not schema or "Priority" in schema) else "priority"
            properties[priority_key] = {"select": {"name": priority}}
        if tag:
            if "Tags" in schema:
                tag_type = schema.get("Tags")
                if tag_type == "multi_select":
                    properties["Tags"] = {"multi_select": [{"name": tag}]}
                else:
                    properties["Tags"] = {"select": {"name": tag}}
            elif "Tag" in schema:
                tag_type = schema.get("Tag")
                if tag_type == "multi_select":
                    properties["Tag"] = {"multi_select": [{"name": tag}]}
                else:
                    properties["Tag"] = {"select": {"name": tag}}
            else:
                properties["Tag"] = {"select": {"name": tag}}

        if due_date is not None:
            due_key = "Due date" if (not schema or "Due date" in schema) else ("Due Date" if "Due Date" in schema else "Due date")
            properties[due_key] = {"date": {"start": due_date}}
        if description is not None:
            desc_key = "Description" if (not schema or "Description" in schema) else "description"
            properties[desc_key] = {
                "rich_text": [
                    {"text": {"content": description}}
                ]
            }

        return self._request_with_retry(
            self.client.pages.create,
            parent={"database_id": self.database_id},
            properties=properties
        )

    def _parse_page_to_dict(self, page: Dict[str, Any]) -> TaskDict:
        props = page.get("properties", {})

        title = "Untitled"
        for title_key in ("Name", "Title", "name", "title"):
            if title_key in props:
                title_list = props[title_key].get("title", [])
                if title_list:
                    title = "".join([t.get("plain_text", "") for t in title_list])
                    break
        if title == "Untitled":
            for val in props.values():
                if isinstance(val, dict) and val.get("type") == "title":
                    title_list = val.get("title", [])
                    if title_list:
                        title = "".join([t.get("plain_text", "") for t in title_list])
                        break

        due_date = None
        for date_key in ("Due date", "Due Date", "Due", "due_date"):
            if date_key in props and isinstance(props[date_key], dict):
                date_obj = props[date_key].get("date")
                if date_obj and isinstance(date_obj, dict):
                    due_date = date_obj.get("start")
                    break
        if due_date is None:
            for val in props.values():
                if isinstance(val, dict) and val.get("type") == "date":
                    date_obj = val.get("date")
                    if date_obj and isinstance(date_obj, dict):
                        due_date = date_obj.get("start")
                        break

        priority = None
        priority_prop = props.get("Priority") or props.get("priority")
        if priority_prop and isinstance(priority_prop, dict):
            select_obj = priority_prop.get("select")
            if select_obj and isinstance(select_obj, dict):
                priority = select_obj.get("name")

        tag = None
        tag_prop = props.get("Tag") or props.get("tag")
        if tag_prop and isinstance(tag_prop, dict):
            select_obj = tag_prop.get("select")
            if select_obj and isinstance(select_obj, dict):
                tag = select_obj.get("name")
            elif "multi_select" in tag_prop:
                ms = tag_prop.get("multi_select", [])
                if ms and isinstance(ms, list):
                    tag = ms[0].get("name")

        page_id = page.get("id")
        page_url = page.get("url")
        if not page_url and page_id:
            clean_id = page_id.replace("-", "")
            page_url = f"https://www.notion.so/{clean_id}"

        return TaskDict(
            title=title,
            due_date=due_date,
            priority=priority,
            tag=tag,
            page_id=page_id,
            url=page_url,
        )

    def _query_database(self, **kwargs) -> Dict[str, Any]:
        """Queries database using databases.query or client.request depending on notion-client version."""
        if self.client is None:
            return {}
        db_id = kwargs.pop("database_id", self.database_id)
        if hasattr(self.client, "databases") and hasattr(self.client.databases, "query"):
            return self._request_with_retry(self.client.databases.query, database_id=db_id, **kwargs)
        elif hasattr(self.client, "request"):
            return self._request_with_retry(
                self.client.request,
                path=f"databases/{db_id}/query",
                method="POST",
                body=kwargs
            )
        elif hasattr(self.client, "data_sources") and hasattr(self.client.data_sources, "query"):
            return self._request_with_retry(self.client.data_sources.query, data_source_id=db_id, **kwargs)
        else:
            raise AttributeError("Installed notion-client has no endpoint for querying database")

    def get_pending(
        self,
        limit: int = 5,
        offset: int = 0,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[TaskDict]:
        """Queries DB where Status != Done, optionally filtered by priority and tag, sorted by Due date ascending."""
        if self.client is None:
            return []

        if not priority and not tag and offset == 0:
            fetch_size = limit
        else:
            fetch_size = max(100, offset + limit)

        filter_conditions: List[Dict[str, Any]] = [
            {
                "property": "Status",
                "status": {
                    "does_not_equal": "Done"
                }
            }
        ]

        if priority:
            filter_conditions.append({
                "property": "Priority",
                "select": {
                    "equals": priority.capitalize()
                }
            })

        query_filter: Dict[str, Any]
        if len(filter_conditions) == 1:
            query_filter = filter_conditions[0]
        else:
            query_filter = {"and": filter_conditions}

        try:
            response = self._query_database(
                database_id=self.database_id,
                page_size=min(fetch_size, 100),
                filter=query_filter,
                sorts=[
                    {
                        "property": "Due date",
                        "direction": "ascending"
                    }
                ]
            )
        except Exception as exc:
            logger.warning(f"Filtered Notion query failed ({exc}), falling back to unfiltered status query")
            response = self._query_database(
                database_id=self.database_id,
                page_size=min(fetch_size, 100),
                filter={
                    "property": "Status",
                    "status": {
                        "does_not_equal": "Done"
                    }
                },
                sorts=[
                    {
                        "property": "Due date",
                        "direction": "ascending"
                    }
                ]
            )

        tasks: List[TaskDict] = []
        for page in response.get("results", []):
            task = self._parse_page_to_dict(page)
            if priority and str(task.get("priority", "")).strip().lower() != priority.strip().lower():
                continue
            if tag and str(task.get("tag", "")).strip().lower() != tag.strip().lower():
                continue
            tasks.append(task)

        tasks.sort(key=lambda x: (x["due_date"] is None, x["due_date"] or ""))
        return tasks[offset : offset + limit]

    def get_reminder_candidates(self) -> Tuple[List[TaskDict], List[TaskDict]]:
        """Returns two lists of task dicts with Status != Done:

        (a) tasks with Due date within the next 2 days
        (b) tasks with Priority = High and Due date is empty
        """
        if self.client is None:
            return ([], [])

        results: List[Dict[str, Any]] = []
        has_more = True
        start_cursor = None

        while has_more:
            query_kwargs: Dict[str, Any] = {
                "database_id": self.database_id,
                "filter": {
                    "property": "Status",
                    "status": {
                        "does_not_equal": "Done"
                    }
                }
            }
            if start_cursor:
                query_kwargs["start_cursor"] = start_cursor

            response = self._query_database(**query_kwargs)
            results.extend(response.get("results", []))
            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

        list_a: List[TaskDict] = []
        list_b: List[TaskDict] = []

        today = datetime.now(timezone.utc).date()
        two_days_later = today + timedelta(days=2)

        for page in results:
            task = self._parse_page_to_dict(page)
            due_date_str = task.get("due_date")
            priority_str = task.get("priority")

            if due_date_str:
                try:
                    date_part = due_date_str[:10]
                    parsed_date = datetime.strptime(date_part, "%Y-%m-%d").date()
                    if parsed_date <= two_days_later:
                        list_a.append(task)
                except ValueError:
                    pass
            else:
                if priority_str and str(priority_str).strip().lower() == "high":
                    list_b.append(task)

        return (list_a, list_b)

    def fetch_pending_reminders(self) -> List[ReminderItem]:
        """Fetch pending tasks/reminders as ReminderItem objects for backward compatibility."""
        pending_tasks = self.get_pending(limit=100)
        reminders: List[ReminderItem] = []
        for task in pending_tasks:
            reminders.append(
                ReminderItem(
                    page_id=task.get("page_id", ""),
                    title=task.get("title", "Untitled"),
                    due_date=task.get("due_date"),
                    status="Pending"
                )
            )
        return reminders

    def mark_reminder_notified(self, page_id: str) -> Dict[str, Any]:
        """Update Notion page status to Notified."""
        return self._request_with_retry(
            self.client.pages.update,
            page_id=page_id,
            properties={
                "Status": {
                    "status": {
                        "name": "Notified"
                    }
                }
            }
        )

    def get_today_tasks(
        self,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
        offset: int = 0,
        limit: int = 10,
    ) -> List[TaskDict]:
        """Fetch tasks where Due date matches today's date."""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        all_pending = self.get_pending(limit=100, offset=0, priority=priority, tag=tag)
        today_tasks: List[TaskDict] = []
        for task in all_pending:
            due = task.get("due_date")
            if due and due.startswith(today_str):
                today_tasks.append(task)
        return today_tasks[offset : offset + limit]

    def update_task_status(
        self,
        title_query: str,
        status_name: Optional[str] = None,
        new_due_date: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Finds active task by title search and updates its Status and/or Due date."""
        if self.client is None or not title_query:
            return (False, title_query, None)

        response = self._query_database()
        results = response.get("results", [])

        target_page_id = None
        matched_title = title_query
        query_norm = title_query.strip().lower()

        for page in results:
            task = self._parse_page_to_dict(page)
            task_title = task.get("title", "")
            if query_norm in task_title.lower() or task_title.lower() in query_norm:
                target_page_id = page.get("id")
                matched_title = task_title
                break

        if not target_page_id:
            return (False, title_query, None)

        schema = self._get_db_properties_schema()
        update_props: Dict[str, Any] = {}

        if status_name:
            update_props["Status"] = {"status": {"name": status_name}}

        if new_due_date:
            due_key = "Due date" if (not schema or "Due date" in schema) else ("Due Date" if "Due Date" in schema else "Due date")
            update_props[due_key] = {"date": {"start": new_due_date}}

        if not update_props:
            return (True, matched_title, None)

        updated_page = self._request_with_retry(
            self.client.pages.update,
            page_id=target_page_id,
            properties=update_props,
        )
        return (True, matched_title, updated_page)

    def create_subject_page(
        self,
        title: str,
        curriculum_topics: List[str],
        resources: Optional[List[Any]] = None,
        starter_tasks: Optional[List[str]] = None,
        overview: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates the Subject page in NOTION_SUBJECTS_DB_ID with Subject title and

        a rich children array of blocks for curriculum topics, direct clickable resource links
        (with types and summaries), and starter tasks.
        Leaves Completed tasks and % Completed untouched as read-only rollups.
        """
        target_db_id = self.subjects_db_id
        if not target_db_id:
            raise ValueError("NOTION_SUBJECTS_DB_ID is not configured.")

        schema = self._get_db_properties_schema(target_db_id)

        # Determine title property key (e.g. 'Subject', 'Name', 'Title')
        title_key = "Subject"
        if "Subject" in schema:
            title_key = "Subject"
        else:
            for k, v in schema.items():
                if v == "title":
                    title_key = k
                    break

        properties: Dict[str, Any] = {
            title_key: {
                "title": [
                    {"text": {"content": title}}
                ]
            }
        }

        # Build children array of blocks directly in payload
        children_blocks: List[Dict[str, Any]] = []

        # If rich sections (resources or starter_tasks or overview) are provided, build formatted layout
        if resources or starter_tasks or overview:
            # 1. Optional Overview Callout
            if overview:
                children_blocks.append({
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "icon": {"type": "emoji", "emoji": "💡"},
                        "rich_text": [{"type": "text", "text": {"content": overview.strip()}}]
                    }
                })

            # 2. Heading: Curriculum & Key Concepts
            if curriculum_topics:
                children_blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": "📖 Curriculum & Key Concepts"}}]
                    }
                })
                for topic in curriculum_topics:
                    topic_str = topic.strip()
                    if not topic_str:
                        continue
                    children_blocks.append({
                        "object": "block",
                        "type": "numbered_list_item",
                        "numbered_list_item": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": topic_str}
                                }
                            ]
                        }
                    })

            # 3. Heading: Core Resources & Papers (direct clickable links + summaries)
            if resources:
                children_blocks.append({
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                })
                children_blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": "📚 Core Resources & Papers"}}]
                    }
                })
                for res in resources:
                    res_url = res.get("url") if isinstance(res, dict) else getattr(res, "url", "")
                    res_name = (
                        res.get("name") or res.get("title")
                        if isinstance(res, dict)
                        else getattr(res, "name", "")
                    )
                    res_type = (
                        res.get("resource_type")
                        if isinstance(res, dict)
                        else getattr(res, "resource_type", "Article")
                    )
                    res_summary = (
                        res.get("summary")
                        if isinstance(res, dict)
                        else getattr(res, "summary", None)
                    )

                    if not res_name and res_url:
                        res_name = res_url
                    if not res_name:
                        continue

                    rich_text_elements: List[Dict[str, Any]] = []
                    if res_type:
                        rich_text_elements.append({
                            "type": "text",
                            "text": {"content": f"[{res_type}] "},
                            "annotations": {"bold": True, "color": "blue"}
                        })

                    if res_url:
                        rich_text_elements.append({
                            "type": "text",
                            "text": {"content": res_name, "link": {"url": res_url}},
                            "annotations": {"bold": True, "underline": True}
                        })
                    else:
                        rich_text_elements.append({
                            "type": "text",
                            "text": {"content": res_name},
                            "annotations": {"bold": True}
                        })

                    if res_summary:
                        clean_summary = str(res_summary).strip()
                        if clean_summary:
                            rich_text_elements.append({
                                "type": "text",
                                "text": {"content": f" — {clean_summary}"}
                            })

                    children_blocks.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": rich_text_elements
                        }
                    })

            # 4. Heading: Starter Tasks
            if starter_tasks:
                children_blocks.append({
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                })
                children_blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": "🎯 Starter Tasks"}}]
                    }
                })
                for st in starter_tasks:
                    st_str = st.strip()
                    if not st_str:
                        continue
                    children_blocks.append({
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"type": "text", "text": {"content": st_str}}],
                            "checked": False
                        }
                    })
        else:
            # Fallback / simple layout: just numbered list items
            for topic in curriculum_topics:
                topic_str = topic.strip()
                if not topic_str:
                    continue
                children_blocks.append({
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": topic_str}
                            }
                        ]
                    }
                })

        return self._request_with_retry(
            self.client.pages.create,
            parent={"database_id": target_db_id},
            properties=properties,
            children=children_blocks,
        )

    def create_resource_row(
        self,
        name: str,
        url: str,
        resource_type: str,
        subject_page_id: str,
    ) -> Dict[str, Any]:
        """Creates a row in NOTION_RESOURCES_DB_ID: Resource Name (title),

        Type (select), URL, and Subjects (relation to Subject page ID).
        """
        target_db_id = self.resources_db_id
        if not target_db_id:
            raise ValueError("NOTION_RESOURCES_DB_ID is not configured.")

        schema = self._get_db_properties_schema(target_db_id)

        title_key = "Resource Name"
        if "Resource Name" in schema:
            title_key = "Resource Name"
        elif "Name" in schema:
            title_key = "Name"
        elif "Title" in schema:
            title_key = "Title"
        else:
            for k, v in schema.items():
                if v == "title":
                    title_key = k
                    break

        rel_key = "Subjects"
        if "Subjects" in schema:
            rel_key = "Subjects"
        elif "Subject" in schema:
            rel_key = "Subject"

        type_key = "Type" if (not schema or "Type" in schema) else "Resource Type"

        properties: Dict[str, Any] = {
            title_key: {
                "title": [
                    {"text": {"content": name or url}}
                ]
            },
            type_key: {
                "select": {
                    "name": resource_type
                }
            },
            "URL": {
                "url": url
            },
            rel_key: {
                "relation": [
                    {"id": subject_page_id}
                ]
            }
        }

        return self._request_with_retry(
            self.client.pages.create,
            parent={"database_id": target_db_id},
            properties=properties,
        )

    def create_starter_task(
        self,
        title: str,
        subject_page_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a starter task in NOTION_TASKS_DB_ID with Tags=['Learning']

        and links it back to the Subject relation property.
        """
        schema = self._get_db_properties_schema(self.tasks_db_id)

        title_key = "Name"
        if "Task name" in schema:
            title_key = "Task name"
        elif "Title" in schema:
            title_key = "Title"
        elif "Name" in schema:
            title_key = "Name"
        else:
            for k, v in schema.items():
                if v == "title":
                    title_key = k
                    break

        properties: Dict[str, Any] = {
            title_key: {
                "title": [
                    {"text": {"content": title}}
                ]
            },
            "Status": {
                "status": {
                    "name": "Not started"
                }
            }
        }

        if "Tags" in schema:
            tag_type = schema.get("Tags")
            if tag_type == "multi_select":
                properties["Tags"] = {"multi_select": [{"name": LEARNING_TAG}]}
            else:
                properties["Tags"] = {"select": {"name": LEARNING_TAG}}
        elif "Tag" in schema:
            tag_type = schema.get("Tag")
            if tag_type == "multi_select":
                properties["Tag"] = {"multi_select": [{"name": LEARNING_TAG}]}
            else:
                properties["Tag"] = {"select": {"name": LEARNING_TAG}}
        else:
            properties["Tags"] = {"multi_select": [{"name": LEARNING_TAG}]}

        if subject_page_id:
            for rel_candidate in ["Subject", "Subjects", "Course", "Topic"]:
                if rel_candidate in schema:
                    properties[rel_candidate] = {"relation": [{"id": subject_page_id}]}
                    break

        created_task = self._request_with_retry(
            self.client.pages.create,
            parent={"database_id": self.tasks_db_id},
            properties=properties,
        )

        # Also link back via Subject Tasks relation if configured on Subject page
        if subject_page_id and self.subjects_db_id:
            try:
                subj_schema = self._get_db_properties_schema(self.subjects_db_id)
                if "Tasks" in subj_schema:
                    task_id = created_task.get("id")
                    if task_id:
                        self._request_with_retry(
                            self.client.pages.update,
                            page_id=subject_page_id,
                            properties={"Tasks": {"relation": [{"id": task_id}]}}
                        )
            except Exception as subj_err:
                logger.debug("Could not link task back to Subject page: %s", subj_err)

        return created_task

    def _build_leetcode_blocks(
        self,
        verdict: Optional[str] = None,
        time_complexity: Optional[str] = None,
        space_complexity: Optional[str] = None,
        is_optimal: bool = True,
        review_text: str = "",
        testing_questions: Optional[List[str]] = None,
        code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Constructs Notion child blocks for LeetCode problem review log."""
        blocks: List[Dict[str, Any]] = []

        def make_paragraph(text_content: str) -> Dict[str, Any]:
            return {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": text_content[:2000]}}
                    ]
                }
            }

        def make_heading_2(text_content: str) -> Dict[str, Any]:
            return {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {"type": "text", "text": {"content": text_content[:2000]}}
                    ]
                }
            }

        def make_bullet_item(text_content: str) -> Dict[str, Any]:
            return {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"type": "text", "text": {"content": text_content[:2000]}}
                    ]
                }
            }

        # 1. Summary Callout Block
        verdict_str = verdict or "Evaluated"
        opt_str = "Optimal approach" if is_optimal else "Suboptimal approach"
        summary_lines = [
            f"Verdict: {verdict_str} ({opt_str})",
            f"Time Complexity: {time_complexity or 'N/A'}",
            f"Space Complexity: {space_complexity or 'N/A'}",
        ]
        callout_text = "\n".join(summary_lines)
        blocks.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [
                    {"type": "text", "text": {"content": callout_text[:2000]}}
                ],
                "icon": {"emoji": "💻"}
            }
        })

        # 2. Review Breakdown
        if review_text and review_text.strip():
            blocks.append(make_heading_2("Analysis & Complexity Review"))
            for paragraph in review_text.strip().split("\n\n"):
                clean_p = paragraph.strip()
                if not clean_p:
                    continue
                for i in range(0, len(clean_p), 2000):
                    blocks.append(make_paragraph(clean_p[i:i + 2000]))

        # 3. Targeted Testing Questions
        if testing_questions:
            blocks.append(make_heading_2("Targeted Testing & Logic Questions"))
            for q in testing_questions:
                clean_q = q.strip().lstrip("•-*0123456789. ")
                if clean_q:
                    for i in range(0, len(clean_q), 2000):
                        blocks.append(make_bullet_item(clean_q[i:i + 2000]))

        # 4. Submitted Solution Code
        if code and code.strip():
            blocks.append(make_heading_2("Submitted Solution"))
            clean_code = code.strip()
            # Notion code blocks have 2000 character limit per rich_text element
            code_text_chunks = [
                {"type": "text", "text": {"content": clean_code[i:i + 2000]}}
                for i in range(0, min(len(clean_code), 4000), 2000)
            ]
            blocks.append({
                "object": "block",
                "type": "code",
                "code": {
                    "rich_text": code_text_chunks,
                    "language": "python"
                }
            })

        return blocks

    def create_leetcode_log_row(
        self,
        problem_title: str,
        difficulty: Optional[str] = None,
        verdict: Optional[str] = None,
        time_complexity: Optional[str] = None,
        space_complexity: Optional[str] = None,
        is_optimal: bool = True,
        review_text: str = "",
        testing_questions: Optional[List[str]] = None,
        code: Optional[str] = None,
        problem_url: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        database_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a row in NOTION_LEETCODE_LOG_DB_ID for a LeetCode problem review."""
        target_db_id = database_id or self.leetcode_log_db_id
        if not target_db_id:
            raise ValueError("NOTION_LEETCODE_LOG_DB_ID is not configured.")

        schema = self._get_db_properties_schema(target_db_id)

        # Title Property Key
        title_key = "Problem"
        if "Problem" in schema:
            title_key = "Problem"
        elif "Problem Name" in schema:
            title_key = "Problem Name"
        elif "Name" in schema:
            title_key = "Name"
        elif "Title" in schema:
            title_key = "Title"
        else:
            for k, v in schema.items():
                if v == "title":
                    title_key = k
                    break

        properties: Dict[str, Any] = {
            title_key: {
                "title": [
                    {"text": {"content": problem_title}}
                ]
            }
        }

        # Difficulty
        if difficulty:
            for diff_key in ["Difficulty", "difficulty"]:
                if diff_key in schema:
                    prop_type = schema[diff_key]
                    if prop_type == "select":
                        properties[diff_key] = {"select": {"name": difficulty}}
                    elif prop_type == "rich_text":
                        properties[diff_key] = {"rich_text": [{"text": {"content": difficulty}}]}
                    break

        # Verdict / Status
        verdict_val = verdict or "Solved"
        for verdict_key in ["Verdict", "Status", "Result"]:
            if verdict_key in schema:
                prop_type = schema[verdict_key]
                if prop_type == "select":
                    properties[verdict_key] = {"select": {"name": verdict_val}}
                elif prop_type == "status":
                    properties[verdict_key] = {"status": {"name": verdict_val}}
                elif prop_type == "rich_text":
                    properties[verdict_key] = {"rich_text": [{"text": {"content": verdict_val}}]}
                break

        # Time Complexity
        if time_complexity:
            for tc_key in ["Time Complexity", "Time complexity", "Time"]:
                if tc_key in schema:
                    prop_type = schema[tc_key]
                    if prop_type == "rich_text":
                        properties[tc_key] = {"rich_text": [{"text": {"content": time_complexity}}]}
                    elif prop_type == "select":
                        properties[tc_key] = {"select": {"name": time_complexity}}
                    break

        # Space Complexity
        if space_complexity:
            for sc_key in ["Space Complexity", "Space complexity", "Space"]:
                if sc_key in schema:
                    prop_type = schema[sc_key]
                    if prop_type == "rich_text":
                        properties[sc_key] = {"rich_text": [{"text": {"content": space_complexity}}]}
                    elif prop_type == "select":
                        properties[sc_key] = {"select": {"name": space_complexity}}
                    break

        # Date
        for date_key in ["Date", "Review Date", "Created Date"]:
            if date_key in schema and schema[date_key] == "date":
                properties[date_key] = {"date": {"start": datetime.now(timezone.utc).strftime("%Y-%m-%d")}}
                break

        # URL
        if problem_url:
            for url_key in ["URL", "Link", "LeetCode URL"]:
                if url_key in schema and schema[url_key] == "url":
                    properties[url_key] = {"url": problem_url}
                    break

        # Patterns
        if patterns:
            for pat_key in ["Patterns", "Pattern", "Tags", "Tag"]:
                if pat_key in schema:
                    prop_type = schema[pat_key]
                    if prop_type == "multi_select":
                        properties[pat_key] = {"multi_select": [{"name": p} for p in patterns[:5]]}
                    elif prop_type == "select":
                        properties[pat_key] = {"select": {"name": patterns[0]}}
                    break

        # Build child blocks
        children_blocks = self._build_leetcode_blocks(
            verdict=verdict,
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            is_optimal=is_optimal,
            review_text=review_text,
            testing_questions=testing_questions,
            code=code,
        )

        return self._request_with_retry(
            self.client.pages.create,
            parent={"database_id": target_db_id},
            properties=properties,
            children=children_blocks,
        )


