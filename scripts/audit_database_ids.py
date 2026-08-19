import os
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

from notion_client import Client

def main():
    token = os.getenv("NOTION_TOKEN")
    if not token:
        print("Error: NOTION_TOKEN is not set.")
        return

    client = Client(auth=token)

    true_database_ids = {
        "NOTION_DATABASE_ID": "3b938af8-cb58-803a-b959-f1a85a4bceb3",
        "NOTION_TASKS_DB_ID": "3b938af8-cb58-803a-b959-f1a85a4bceb3",
        "NOTION_RAMBLINGS_ID": "3bc38af8-cb58-8015-98af-f26353131d74",
        "NOTION_SUBSTACK_ID": "3bc38af8-cb58-808f-aaf3-fca024abea7b",
        "NOTION_DAILY_LOGS_ID": "3b938af8-cb58-80b3-9ae5-cf5227292db0",
        "NOTION_SUBJECTS_DB_ID": "90538af8-cb58-8225-bb41-812b0ab3bf25",
        "NOTION_RESOURCES_DB_ID": "54d38af8-cb58-82d6-8487-81ed4c16cd0f",
        "NOTION_LEETCODE_LOG_DB_ID": "3bc38af8-cb58-8027-9823-f67cced165b0",
    }

    print("==================================================")
    print(" VERIFYING ALL 7 RECOVERED DATABASE BLOCK IDS")
    print("==================================================")

    all_passed = True
    for key, db_id in true_database_ids.items():
        try:
            db_res = client.databases.retrieve(database_id=db_id)
            title_objs = db_res.get("title", [])
            title = "".join([t.get("plain_text", "") for t in title_objs]) or "Untitled DB"
            print(f"✅ {key:<26} -> {db_id} [OK: '{title}']")
        except Exception as e:
            all_passed = False
            print(f"❌ {key:<26} -> {db_id} [FAIL: {e}]")

    print("\n--------------------------------------------------")
    if all_passed:
        print("🎉 ALL 7 DATABASES SUCCESSFULLY VALIDATED VIA NOTION API!")
    else:
        print("⚠️ Some databases failed validation.")

if __name__ == "__main__":
    main()
