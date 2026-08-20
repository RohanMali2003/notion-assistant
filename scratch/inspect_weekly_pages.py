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

print(f"Found {len(weekly_pages)} weekly review pages in Tasks Tracker:")
for pid, title, ctime in weekly_pages:
    print(f" - {pid} | {title} | {ctime}")
