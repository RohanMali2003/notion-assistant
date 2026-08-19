import os
import sys
import json
import requests
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

def main():
    token = os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN")
    waba_id = "1662217541542785"

    if not token:
        print("❌ Error: Neither WHATSAPP_ACCESS_TOKEN nor WHATSAPP_TOKEN found in .env")
        return

    print(f"Subscribing WABA ID '{waba_id}' to App webhooks via Graph API...")
    url = f"https://graph.facebook.com/v20.0/{waba_id}/subscribed_apps"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers)
        print(f"HTTP Status: {response.status_code}")
        print("API Response:", json.dumps(response.json(), indent=2))

        # Also GET subscribed apps to verify
        get_resp = requests.get(url, headers=headers)
        print("\nChecking current subscribed apps on WABA:")
        print("GET Response:", json.dumps(get_resp.json(), indent=2))
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    main()
