"""Multimodal Vision & Media Processing Service for Ocean."""

import logging
import os
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from app.notion_client import NotionAssistantClient
from app.tag_directory import CANONICAL_TAG_NAMES, match_closest_tag
from app.telegram_client import TelegramAssistantClient
from app.whatsapp_client import WhatsAppAssistantClient

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def get_gemini_client():
    """Create and return a google-genai Client instance."""
    if genai is None:
        raise RuntimeError("google-genai library is not installed or available")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


VISION_ANALYSIS_SYSTEM_INSTRUCTION = (
    "You are an elite multimodal intelligence analyst and knowledge capture engine for Ocean.\n"
    "Your goal is to accurately transcribe, analyze, and synthesize any image, whiteboard diagram, handwritten note, code screenshot, document, or receipt.\n\n"
    "Strict Requirements:\n"
    "1. FOCUS ON DEPTH AND ACCURACY: Completely transcribe all legible text, explain architecture diagrams, and extract any actionable to-dos.\n"
    "2. FORMATTING RULES:\n"
    "   - DO NOT use LaTeX math notation or dollar signs (write O(N), 10^9, N directly, NEVER $O(N)$ or $10^9$).\n"
    "   - DO NOT use double asterisks (**). Use clean standard markdown formatting.\n"
    "3. STRUCTURED OUTPUT FORMAT:\n"
    "TITLE: <Concise, descriptive title for the image/note>\n"
    "CATEGORY: <WHITEBOARD_DIAGRAM / HANDWRITTEN_NOTES / CODE_SCREENSHOT / DOCUMENT_RECEIPT / GENERAL_IMAGE>\n"
    "DOMAIN: <Best matching tag: AI Research, System Design, Distributed Systems, Leetcode, Finances, Schoolwork, Projects, etc.>\n"
    "SYNTHESIS:\n<Comprehensive transcription and deep analytical explanation of the content, diagrams, code, or data>\n"
    "ACTION_ITEMS:\n"
    "- <Concrete action item or task 1 (if any detected)>\n"
    "- <Concrete action item or task 2 (if any detected)>"
)


def analyze_image_with_gemini(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    caption: str = "",
) -> Dict[str, Any]:
    """Analyze image using Gemini Multimodal Vision API."""
    client = get_gemini_client()

    prompt_text = (
        f"User Caption / Note: {caption or 'None'}\n\n"
        "Please analyze this image, transcribe all text and diagrams, and provide your structured synthesis."
    )

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)),
            contents=[image_part, prompt_text],
            config=types.GenerateContentConfig(
                system_instruction=VISION_ANALYSIS_SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
        resp_text = response.text or ""
    except Exception as exc:
        logger.error("Gemini Vision analysis failed (%s)", exc)
        resp_text = ""

    # Parse response fields
    title = caption[:50] if caption else "Visual Note"
    title_m = re.search(r"TITLE:\s*(.+)", resp_text, re.IGNORECASE)
    if title_m:
        title = title_m.group(1).strip()

    category = "GENERAL_IMAGE"
    cat_m = re.search(r"CATEGORY:\s*([A-Za-z_]+)", resp_text, re.IGNORECASE)
    if cat_m:
        category = cat_m.group(1).strip().upper()

    domain_tag = "Miscellaneous"
    tag_m = re.search(r"DOMAIN:\s*(.+)", resp_text, re.IGNORECASE)
    if tag_m:
        domain_tag = match_closest_tag(tag_m.group(1).strip(), title)

    synthesis = ""
    syn_m = re.search(r"SYNTHESIS:\s*(.*?)(?=ACTION_ITEMS:|$)", resp_text, re.DOTALL | re.IGNORECASE)
    if syn_m:
        synthesis = syn_m.group(1).strip()
    else:
        synthesis = resp_text.strip()

    action_items = []
    act_m = re.search(r"ACTION_ITEMS:\s*(.*)$", resp_text, re.DOTALL | re.IGNORECASE)
    if act_m:
        for line in act_m.group(1).strip().split("\n"):
            clean_l = line.strip().lstrip("-*•0123456789. ")
            if clean_l and clean_l.lower() not in ("none", "n/a", "no action items"):
                action_items.append(clean_l)

    return {
        "title": title,
        "category": category,
        "domain_tag": domain_tag,
        "synthesis": synthesis,
        "action_items": action_items,
        "full_text": resp_text,
    }


def execute_media_pipeline(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    caption: str = "",
    to_phone: Optional[str] = None,
    chat_id: Optional[str] = None,
    notion_client: Optional[NotionAssistantClient] = None,
    whatsapp_client: Optional[WhatsAppAssistantClient] = None,
    telegram_client: Optional[TelegramAssistantClient] = None,
) -> Dict[str, Any]:
    """Process an image in the background, log to Notion, and send a reply."""
    notion = notion_client or NotionAssistantClient()
    whatsapp = whatsapp_client or WhatsAppAssistantClient()
    telegram = telegram_client or TelegramAssistantClient()

    analysis = analyze_image_with_gemini(image_bytes, mime_type, caption)

    title = analysis["title"]
    category = analysis["category"]
    domain_tag = analysis["domain_tag"]
    synthesis = analysis["synthesis"]
    action_items = analysis["action_items"]

    # Log to Notion
    notion_url = None
    notion_page_id = None

    try:
        # If receipt/document with specific tasks, create task in Tasks DB
        if category == "DOCUMENT_RECEIPT" or (action_items and category not in ("WHITEBOARD_DIAGRAM", "CODE_SNIPPET")):
            task_res = notion.create_task(
                title=title,
                priority="Medium",
                tag=domain_tag if domain_tag in CANONICAL_TAG_NAMES else "Miscellaneous",
                description=synthesis[:2000],
            )
            notion_page_id = task_res.get("id")
            notion_url = task_res.get("url")
        # For whiteboard diagrams, code screenshots, handwritten notes -> log to MIND / Daily Logs
        else:
            mind_res = notion.create_rambling(
                title=f"📸 {title}",
                summary=synthesis[:300],
                full_text=synthesis,
                tags=[domain_tag, category.lower()],
            )
            notion_page_id = mind_res.get("id")
            notion_url = mind_res.get("url")

        if not notion_url and notion_page_id:
            clean_id = str(notion_page_id).replace("-", "")
            notion_url = f"https://www.notion.so/{clean_id}"
    except Exception as notion_err:
        logger.error("Failed to log visual analysis to Notion (%s)", notion_err)

    # Format reply message for WhatsApp / Telegram
    cat_emoji = {
        "WHITEBOARD_DIAGRAM": "📊",
        "HANDWRITTEN_NOTES": "📝",
        "CODE_SCREENSHOT": "💻",
        "DOCUMENT_RECEIPT": "🧾",
        "GENERAL_IMAGE": "🖼️",
    }.get(category, "📸")

    msg_lines = [
        f"{cat_emoji} *Visual Analysis: {title}*",
        f"🏷️ *Tag:* {domain_tag} | 📂 *Type:* {category.replace('_', ' ').title()}\n",
        f"💡 *Overview:*\n{synthesis}",
    ]

    if action_items:
        items_str = "\n".join([f"• {act}" for act in action_items[:4]])
        msg_lines.append(f"\n🎯 *Action Items Extracted:*\n{items_str}")

    if notion_url:
        msg_lines.append(f"\n🔗 *Notion Log:* {notion_url}")

    reply_text = "\n".join(msg_lines).strip()

    # Send confirmation
    if to_phone:
        try:
            whatsapp.send_message(to=to_phone, text=reply_text, preview_url=bool(notion_url))
            logger.info("Sent WhatsApp visual analysis to %s", to_phone)
        except Exception as wa_err:
            logger.error("Failed to send WhatsApp message: %s", wa_err)

    if chat_id:
        try:
            telegram.send_message(text=reply_text, chat_id=str(chat_id))
            logger.info("Sent Telegram visual analysis to %s", chat_id)
        except Exception as tg_err:
            logger.error("Failed to send Telegram message: %s", tg_err)

    return {
        "status": "ok",
        "title": title,
        "category": category,
        "domain_tag": domain_tag,
        "notion_url": notion_url,
        "reply_text": reply_text,
    }
