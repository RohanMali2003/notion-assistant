"""Dynamic Notion Database Schema Introspection and Polymorphic Property Adapters.

Encapsulates Notion database schema discovery, field matching, and type-safe
property formatting for arbitrary Notion databases without hardcoded schema assumptions.
"""

from abc import ABC, abstractmethod
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from app.matcher import resolve_natural_date

logger = logging.getLogger("notion-assistant.schema_engine")

# Max text length permitted by Notion per rich_text block element
NOTION_TEXT_LIMIT = 2000


class BasePropertyAdapter(ABC):
    """Abstract base class for polymorphic Notion property adapters."""

    def __init__(self, name: str, property_type: str, raw_schema: Dict[str, Any]):
        self.name = name
        self.property_type = property_type
        self.raw_schema = raw_schema
        self.name_clean = self._clean_token(name)

    @staticmethod
    def _clean_token(text: str) -> str:
        return re.sub(r"[^\w\s]", " ", text.lower()).strip()

    @abstractmethod
    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        """Format an arbitrary Python value into the Notion property JSON payload."""
        pass

    def matches_field_name(self, field_name: str) -> bool:
        """Check if an input field name matches this property name."""
        f_clean = self._clean_token(field_name)
        if not f_clean:
            return False
        return f_clean == self.name_clean or f_clean in self.name_clean or self.name_clean in f_clean


class TitlePropertyAdapter(BasePropertyAdapter):
    """Adapter for Notion 'title' property."""

    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        text_str = str(value).strip()[:NOTION_TEXT_LIMIT]
        return {
            "title": [{"type": "text", "text": {"content": text_str}}]
        }


class RichTextPropertyAdapter(BasePropertyAdapter):
    """Adapter for Notion 'rich_text' property."""

    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        text_str = str(value).strip()
        if not text_str:
            return None
        # Chunk into <= 2000 char chunks if long
        chunks = [text_str[i:i + NOTION_TEXT_LIMIT] for i in range(0, len(text_str), NOTION_TEXT_LIMIT)]
        return {
            "rich_text": [{"type": "text", "text": {"content": c}} for c in chunks[:10]]
        }


class SelectPropertyAdapter(BasePropertyAdapter):
    """Adapter for Notion 'select' property with case-insensitive option matching."""

    def __init__(self, name: str, property_type: str, raw_schema: Dict[str, Any]):
        super().__init__(name, property_type, raw_schema)
        select_meta = raw_schema.get("select", {}) or {}
        self.options: List[Dict[str, Any]] = select_meta.get("options", [])
        self.option_names: List[str] = [opt.get("name", "") for opt in self.options if opt.get("name")]

    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        val_str = str(value).strip()
        if not val_str:
            return None

        # Try exact or case-insensitive match against existing options
        target_name = val_str
        for opt_name in self.option_names:
            if opt_name.lower() == val_str.lower():
                target_name = opt_name
                break

        return {"select": {"name": target_name[:100]}}


class StatusPropertyAdapter(BasePropertyAdapter):
    """Adapter for Notion 'status' property with options matching."""

    def __init__(self, name: str, property_type: str, raw_schema: Dict[str, Any]):
        super().__init__(name, property_type, raw_schema)
        status_meta = raw_schema.get("status", {}) or {}
        self.options: List[Dict[str, Any]] = status_meta.get("options", [])
        self.option_names: List[str] = [opt.get("name", "") for opt in self.options if opt.get("name")]

    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        val_str = str(value).strip()
        if not val_str:
            return None

        target_name = val_str
        for opt_name in self.option_names:
            if opt_name.lower() == val_str.lower():
                target_name = opt_name
                break

        return {"status": {"name": target_name[:100]}}


class MultiSelectPropertyAdapter(BasePropertyAdapter):
    """Adapter for Notion 'multi_select' property."""

    def __init__(self, name: str, property_type: str, raw_schema: Dict[str, Any]):
        super().__init__(name, property_type, raw_schema)
        ms_meta = raw_schema.get("multi_select", {}) or {}
        self.options: List[Dict[str, Any]] = ms_meta.get("options", [])
        self.option_names: List[str] = [opt.get("name", "") for opt in self.options if opt.get("name")]

    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None

        tags: List[str] = []
        if isinstance(value, (list, tuple, set)):
            tags = [str(t).strip() for t in value if str(t).strip()]
        elif isinstance(value, str):
            # Split on comma or semicolon if multiple tags provided in string
            tags = [t.strip() for t in re.split(r"[,;]", value) if t.strip()]
        else:
            tags = [str(value).strip()]

        if not tags:
            return None

        resolved_options = []
        for tag in tags:
            matched = tag
            for opt_name in self.option_names:
                if opt_name.lower() == tag.lower():
                    matched = opt_name
                    break
            resolved_options.append({"name": matched[:100]})

        return {"multi_select": resolved_options}


class DatePropertyAdapter(BasePropertyAdapter):
    """Adapter for Notion 'date' property with relative date resolution."""

    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        raw_str = str(value).strip()
        if not raw_str:
            return None

        # Resolve natural date string (e.g. "tomorrow", "next tuesday", "2026-08-30")
        resolved = resolve_natural_date(raw_str) or raw_str
        return {"date": {"start": resolved}}


class NumberPropertyAdapter(BasePropertyAdapter):
    """Adapter for Notion 'number' property."""

    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return {"number": value}
            val_str = str(value).strip()
            # Extract first number
            match = re.search(r"[-+]?\d*\.?\d+", val_str)
            if match:
                num = float(match.group(0)) if "." in match.group(0) else int(match.group(0))
                return {"number": num}
        except Exception:
            pass
        return None


class UrlPropertyAdapter(BasePropertyAdapter):
    """Adapter for Notion 'url' property."""

    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        val_str = str(value).strip()
        if not val_str:
            return None
        if not val_str.startswith(("http://", "https://")):
            val_str = f"https://{val_str}"
        return {"url": val_str[:2000]}


class CheckboxPropertyAdapter(BasePropertyAdapter):
    """Adapter for Notion 'checkbox' property."""

    def format_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, bool):
            return {"checkbox": value}
        val_str = str(value).strip().lower()
        is_checked = val_str in ("true", "1", "yes", "checked", "done")
        return {"checkbox": is_checked}


# Common semantic aliases mapping user item attributes to database property types
_FIELD_ALIASES: Dict[str, List[str]] = {
    "title": ["title", "name", "task", "book", "item", "problem", "topic", "subject", "article"],
    "author": ["author", "writer", "creator", "by"],
    "status": ["status", "state", "stage", "progress"],
    "details": ["details", "notes", "summary", "description", "content", "takeaways", "note", "comment"],
    "tags": ["tags", "tag", "genre", "category", "categories", "topics", "domain", "type", "patterns"],
    "priority": ["priority", "prio", "urgency", "importance"],
    "rating": ["rating", "score", "stars", "grade"],
    "url": ["url", "link", "website", "href", "source"],
    "date": ["due_date", "date", "due", "deadline", "scheduled", "time"],
}


class NotionDatabaseSchema:
    """Encapsulates the introspected schema of a Notion database and builds property payloads."""

    def __init__(self, database_id: str, raw_schema: Dict[str, Any], title: str = ""):
        self.database_id = database_id
        self.raw_schema = raw_schema
        self.title = title
        self.adapters: Dict[str, BasePropertyAdapter] = {}
        self.title_property_name: Optional[str] = None
        self._build_adapters()

    def _build_adapters(self) -> None:
        raw_properties = self.raw_schema.get("properties", {})
        for prop_name, prop_meta in raw_properties.items():
            prop_type = prop_meta.get("type", "")
            adapter: Optional[BasePropertyAdapter] = None

            if prop_type == "title":
                adapter = TitlePropertyAdapter(prop_name, prop_type, prop_meta)
                self.title_property_name = prop_name
            elif prop_type == "rich_text":
                adapter = RichTextPropertyAdapter(prop_name, prop_type, prop_meta)
            elif prop_type == "select":
                adapter = SelectPropertyAdapter(prop_name, prop_type, prop_meta)
            elif prop_type == "status":
                adapter = StatusPropertyAdapter(prop_name, prop_type, prop_meta)
            elif prop_type == "multi_select":
                adapter = MultiSelectPropertyAdapter(prop_name, prop_type, prop_meta)
            elif prop_type == "date":
                adapter = DatePropertyAdapter(prop_name, prop_type, prop_meta)
            elif prop_type == "number":
                adapter = NumberPropertyAdapter(prop_name, prop_type, prop_meta)
            elif prop_type == "url":
                adapter = UrlPropertyAdapter(prop_name, prop_type, prop_meta)
            elif prop_type == "checkbox":
                adapter = CheckboxPropertyAdapter(prop_name, prop_type, prop_meta)

            if adapter:
                self.adapters[prop_name] = adapter

    def find_adapter_for_field(self, field_name: str) -> Optional[BasePropertyAdapter]:
        """Find the best property adapter matching an input field name."""
        f_clean = field_name.lower().strip()

        # 1. Exact match on property name
        for p_name, adapter in self.adapters.items():
            if p_name.lower() == f_clean:
                return adapter

        # 2. Alias match
        matched_category = None
        for category, aliases in _FIELD_ALIASES.items():
            if f_clean in aliases:
                matched_category = category
                break

        if matched_category:
            for p_name, adapter in self.adapters.items():
                for alias in _FIELD_ALIASES[matched_category]:
                    if adapter.matches_field_name(alias):
                        return adapter

        # 3. Substring match
        for p_name, adapter in self.adapters.items():
            if adapter.matches_field_name(f_clean):
                return adapter

        return None

    def build_page_properties(self, item_fields: Dict[str, Any], default_status: Optional[str] = None) -> Dict[str, Any]:
        """Build full type-safe Notion page properties JSON payload from arbitrary field dict."""
        properties_payload: Dict[str, Any] = {}
        assigned_properties: Set[str] = set()

        # 1. Always format Title property
        title_val = item_fields.get("title") or item_fields.get("name") or "Untitled"
        if self.title_property_name and self.title_property_name in self.adapters:
            title_payload = self.adapters[self.title_property_name].format_value(title_val)
            if title_payload:
                properties_payload[self.title_property_name] = title_payload
                assigned_properties.add(self.title_property_name)

        # 2. Map other fields
        for field_name, field_val in item_fields.items():
            if field_name in ("title", "name") or field_val is None:
                continue

            adapter = self.find_adapter_for_field(field_name)
            if adapter and adapter.name not in assigned_properties:
                prop_payload = adapter.format_value(field_val)
                if prop_payload:
                    properties_payload[adapter.name] = prop_payload
                    assigned_properties.add(adapter.name)

        # 3. Handle default status if Status/Select exists and was not assigned
        if default_status:
            status_adapter = next(
                (a for a in self.adapters.values() if a.property_type in ("status", "select") and a.name not in assigned_properties and "status" in a.name.lower()),
                None
            )
            if status_adapter:
                status_payload = status_adapter.format_value(default_status)
                if status_payload:
                    properties_payload[status_adapter.name] = status_payload
                    assigned_properties.add(status_adapter.name)

        return properties_payload


class NotionSchemaEngine:
    """Engine for caching and retrieving dynamic Notion database schemas."""

    def __init__(self, ttl_seconds: float = 300.0):
        self.ttl_seconds = ttl_seconds
        self._schema_cache: Dict[str, Tuple[float, NotionDatabaseSchema]] = {}

    def get_schema(self, database_id: str, notion_client: Any) -> Optional[NotionDatabaseSchema]:
        """Retrieve and cache the NotionDatabaseSchema for a database_id."""
        if not database_id or not notion_client or not getattr(notion_client, "client", None):
            return None

        clean_id = database_id.strip()
        now = time.time()

        # Check cache
        if clean_id in self._schema_cache:
            ts, schema = self._schema_cache[clean_id]
            if (now - ts) < self.ttl_seconds:
                return schema

        # Fetch from Notion API
        try:
            db_meta = notion_client._request_with_retry(
                notion_client.client.databases.retrieve,
                database_id=clean_id,
            )
            title_text = ""
            raw_title = db_meta.get("title", [])
            if isinstance(raw_title, list) and raw_title:
                title_text = "".join(t.get("plain_text", "") for t in raw_title)

            schema = NotionDatabaseSchema(database_id=clean_id, raw_schema=db_meta, title=title_text)
            self._schema_cache[clean_id] = (now, schema)
            logger.info("Introspected schema for database '%s' (%s) with %d properties", title_text or clean_id, clean_id, len(schema.adapters))
            return schema
        except Exception as exc:
            logger.warning("Failed to introspect schema for database %s: %s", clean_id, exc)
            return None

    def clear_cache(self) -> None:
        """Clear cached schemas."""
        self._schema_cache.clear()


# Global Singleton Instance
schema_engine = NotionSchemaEngine()
