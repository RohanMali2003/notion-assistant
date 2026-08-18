"""Standalone script to find expired LeetCode tasks and mark them as Done in Notion."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load local .env if present
env_file = Path(".env")
if env_file.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)
    except ImportError:
        pass

try:
    from notion_client import Client
except ImportError:
    Client = None


def cleanup_expired_leetcode_tasks() -> int:
    """Find expired tasks tagged 'Leetcode' with Due date < today and mark them as Done."""
    api_key = os.getenv("NOTION_API_KEY") or os.getenv("NOTION_TOKEN")
    tasks_db_id = os.getenv("NOTION_TASKS_DB_ID") or os.getenv("NOTION_DATABASE_ID")

    missing_vars = []
    if not api_key:
        missing_vars.append("NOTION_API_KEY/NOTION_TOKEN")
    if not tasks_db_id:
        missing_vars.append("NOTION_TASKS_DB_ID/NOTION_DATABASE_ID")

    if missing_vars:
        missing_str = ", ".join(missing_vars)
        raise ValueError(f"Missing required environment variable(s): [{missing_str}]")

    if Client is None:
        raise ImportError("notion-client package is required but not installed.")

    try:
        notion = Client(auth=api_key, notion_version="2022-06-28")
    except TypeError:
        notion = Client(auth=api_key)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Searching for expired Leetcode tasks before {today_str}...")

    def _query_db(payload: dict) -> dict:
        if hasattr(notion, "databases") and hasattr(notion.databases, "query"):
            return notion.databases.query(**payload)
        elif hasattr(notion, "request"):
            db_id = payload.get("database_id", tasks_db_id)
            body = {k: v for k, v in payload.items() if k != "database_id"}
            return notion.request(path=f"databases/{db_id}/query", method="POST", body=body)
        elif hasattr(notion, "data_sources") and hasattr(notion.data_sources, "query"):
            db_id = payload.get("database_id", tasks_db_id)
            body = {k: v for k, v in payload.items() if k != "database_id"}
            return notion.data_sources.query(data_source_id=db_id, **body)
        else:
            raise AttributeError("No query endpoint available on notion client")

    results = []
    has_more = True
    start_cursor = None

    while has_more:
        query_payload = {
            "database_id": tasks_db_id,
            "filter": {
                "and": [
                    {
                        "property": "Tags",
                        "multi_select": {
                            "contains": "Leetcode"
                        }
                    },
                    {
                        "property": "Due date",
                        "date": {
                            "before": today_str
                        }
                    },
                    {
                        "property": "Status",
                        "status": {
                            "does_not_equal": "Done"
                        }
                    }
                ]
            }
        }
        if start_cursor:
            query_payload["start_cursor"] = start_cursor

        try:
            response = _query_db(query_payload)
        except Exception as exc:
            # Fallback if property names or filter formats differ
            print(f"Standard filter query encountered an issue ({exc}). Retrying without tag filter...")
            query_payload["filter"] = {
                "and": [
                    {
                        "property": "Due date",
                        "date": {
                            "before": today_str
                        }
                    },
                    {
                        "property": "Status",
                        "status": {
                            "does_not_equal": "Done"
                        }
                    }
                ]
            }
            try:
                response = _query_db(query_payload)
            except Exception as retry_exc:
                print(f"Database query failed: {retry_exc}", file=sys.stderr)
                break

        results.extend(response.get("results", []))
        has_more = response.get("has_more", False)
        start_cursor = response.get("next_cursor")

    if not results:
        print("No expired Leetcode tasks found. We're all clean.")
        return 0

    print(f"Found {len(results)} task(s). Marking as Done...")
    closed_count = 0

    for task in results:
        page_id = task.get("id")
        if not page_id:
            continue
        try:
            notion.pages.update(
                page_id=page_id,
                properties={
                    "Status": {
                        "status": {
                            "name": "Done"
                        }
                    }
                }
            )
            print(f" - Closed task: {page_id}")
            closed_count += 1
        except Exception as err:
            # Try select property structure if status property update fails
            try:
                notion.pages.update(
                    page_id=page_id,
                    properties={
                        "Status": {
                            "select": {
                                "name": "Done"
                            }
                        }
                    }
                )
                print(f" - Closed task (select format): {page_id}")
                closed_count += 1
            except Exception as select_err:
                print(f" - Failed to close task {page_id}: {err} | {select_err}", file=sys.stderr)

    print(f"Successfully closed {closed_count}/{len(results)} task(s).")
    return closed_count


if __name__ == "__main__":
    cleanup_expired_leetcode_tasks()
