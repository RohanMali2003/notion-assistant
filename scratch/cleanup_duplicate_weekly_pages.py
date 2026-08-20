import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from dotenv import load_dotenv
load_dotenv()

from app.notion_client import NotionAssistantClient

notion = NotionAssistantClient()
res = notion._query_database(database_id=notion.tasks_db_id, page_size=100)

weekly_pages = []
for p in res.get("results", []):
    props = p.get("properties", {})
    title_list = props.get("Task name", {}).get("title", [])
    title = "".join(t.get("plain_text", "") for t in title_list)
    if "Weekly Velocity Review" in title:
        weekly_pages.append((p.get("id"), title, p.get("created_time")))

print(f"Found {len(weekly_pages)} review pages.")
# Keep the newest one, delete/archive the other 8
if len(weekly_pages) > 1:
    # Sort by created_time descending
    weekly_pages.sort(key=lambda x: x[2], reverse=True)
    to_keep = weekly_pages[0]
    to_delete = weekly_pages[1:]
    print(f"Keeping newest: {to_keep[0]} ({to_keep[2]})")
    for pid, title, ctime in to_delete:
        try:
            notion._request_with_retry(notion.client.pages.update, page_id=pid, archived=True)
            print(f"Archived duplicate: {pid}")
        except Exception as e:
            print(f"Failed to archive {pid}: {e}")
print("Cleanup complete.")
