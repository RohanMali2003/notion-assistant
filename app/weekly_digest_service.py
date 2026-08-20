"""Sunday Evening Life & Study Velocity Executive Digest Service for Ocean."""

from datetime import datetime, timedelta, timezone
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
from app.schemas import WeeklyVelocityReport
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


WEEKLY_DIGEST_SYSTEM_INSTRUCTION = (
    "You are an elite executive coach, research mentor, and velocity strategist for Ocean.\n"
    "Your objective is to evaluate the user's past week across Tasks, Research & Learning Subjects, Problem Solving, and Deep-Dives.\n\n"
    "Strict Requirements:\n"
    "1. Calculate an accurate Velocity Score (0 to 100) reflecting momentum, completion rate, and intellectual depth.\n"
    "2. Provide an honest, inspiring Verdict phrase (e.g. 'High Momentum', 'Steady Execution', 'Breakthrough Week', 'Rebalancing Needed').\n"
    "3. Synthesize:\n"
    "   - A 1-sentence Executive Headline\n"
    "   - Key Milestones & Completed Highlights\n"
    "   - Learning & Research Progress (Subjects, papers, concepts mastered)\n"
    "   - Algorithmic / LeetCode Summary (if present)\n"
    "   - Bottlenecks / Stalled Tasks carried over\n"
    "   - Top 3 High-Leverage Strategic Priorities for Next Week\n"
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


def fetch_past_week_workspace_activity(
    notion_client: Optional[NotionAssistantClient] = None,
    days: int = 7,
) -> Dict[str, Any]:
    """Query Notion databases to aggregate activity, completions, and roadmaps from the past week."""
    notion = notion_client or NotionAssistantClient()
    client = notion.client
    if client is None:
        raise RuntimeError("Notion client is not initialized")

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff_dt.isoformat()

    tasks_completed = []
    tasks_in_progress = []
    tasks_pending = []
    tasks_overdue = []

    subjects_active = []
    resources_logged = []
    daily_notes = []

    # 1. Fetch Tasks
    if notion.tasks_db_id:
        try:
            res = notion._query_database(
                database_id=notion.tasks_db_id,
                page_size=100,
            )
            for page in res.get("results", []):
                title = _extract_page_title(page)
                url = _extract_page_url(page)
                props = page.get("properties", {})
                status = ""
                if "Status" in props and props["Status"].get("status"):
                    status = props["Status"]["status"].get("name", "")

                due_date = ""
                if "Due Date" in props and props["Due Date"].get("date"):
                    due_date = props["Due Date"]["date"].get("start", "")

                tags = []
                if "Tags" in props and props["Tags"].get("multi_select"):
                    tags = [t.get("name", "") for t in props["Tags"]["multi_select"]]

                task_obj = {
                    "id": page.get("id"),
                    "title": title,
                    "url": url,
                    "status": status,
                    "due_date": due_date,
                    "tags": tags,
                }

                if status == "Done":
                    tasks_completed.append(task_obj)
                elif status == "In progress":
                    tasks_in_progress.append(task_obj)
                else:
                    tasks_pending.append(task_obj)

                # Check if overdue
                if due_date and status != "Done":
                    try:
                        due_dt = datetime.fromisoformat(due_date.split("T")[0]).replace(tzinfo=timezone.utc)
                        if due_dt < datetime.now(timezone.utc):
                            tasks_overdue.append(task_obj)
                    except Exception:
                        pass
        except Exception as exc:
            logger.error("Failed to query tasks for weekly digest: %s", exc)

    # 2. Fetch Subjects & Roadmaps
    if notion.subjects_db_id:
        try:
            res = notion._query_database(
                database_id=notion.subjects_db_id,
                page_size=50,
            )
            for page in res.get("results", []):
                title = _extract_page_title(page)
                url = _extract_page_url(page)
                props = page.get("properties", {})
                status = ""
                if "Status" in props and props["Status"].get("status"):
                    status = props["Status"]["status"].get("name", "")

                created_time = page.get("created_time", "")
                subjects_active.append({
                    "id": page.get("id"),
                    "title": title,
                    "url": url,
                    "status": status,
                    "created_time": created_time,
                })
        except Exception as exc:
            logger.error("Failed to query subjects for weekly digest: %s", exc)

    # 3. Fetch Resources
    if notion.resources_db_id:
        try:
            res = notion._query_database(
                database_id=notion.resources_db_id,
                page_size=50,
            )
            for page in res.get("results", []):
                title = _extract_page_title(page)
                url = _extract_page_url(page)
                props = page.get("properties", {})
                res_type = ""
                if "Type" in props and props["Type"].get("select"):
                    res_type = props["Type"]["select"].get("name", "")
                link_url = ""
                if "URL" in props and props["URL"].get("url"):
                    link_url = props["URL"].get("url", "")

                resources_logged.append({
                    "id": page.get("id"),
                    "title": title,
                    "url": url,
                    "link_url": link_url,
                    "type": res_type,
                })
        except Exception as exc:
            logger.error("Failed to query resources for weekly digest: %s", exc)

    return {
        "cutoff_days": days,
        "tasks_completed": tasks_completed,
        "tasks_in_progress": tasks_in_progress,
        "tasks_pending": tasks_pending,
        "tasks_overdue": tasks_overdue,
        "subjects": subjects_active,
        "resources": resources_logged,
        "total_completed": len(tasks_completed),
        "total_in_progress": len(tasks_in_progress),
        "total_pending": len(tasks_pending),
        "total_overdue": len(tasks_overdue),
    }


def synthesize_velocity_digest(
    activity_data: Dict[str, Any],
) -> WeeklyVelocityReport:
    """Evaluate activity dataset via Gemini and generate structured WeeklyVelocityReport."""
    client = get_gemini_client()

    completed_list = [t["title"] for t in activity_data.get("tasks_completed", [])]
    in_progress_list = [t["title"] for t in activity_data.get("tasks_in_progress", [])]
    overdue_list = [t["title"] for t in activity_data.get("tasks_overdue", [])]
    subjects_list = [s["title"] for s in activity_data.get("subjects", [])]
    resources_list = [f"{r['title']} ({r['type']})" for r in activity_data.get("resources", [])]

    prompt = (
        f"PAST {activity_data.get('cutoff_days', 7)} DAYS ACTIVITY SUMMARY:\n\n"
        f"Tasks Completed ({len(completed_list)}):\n" + "\n".join(f"- {c}" for c in completed_list[:25]) + "\n\n"
        f"Tasks In Progress ({len(in_progress_list)}):\n" + "\n".join(f"- {p}" for p in in_progress_list[:15]) + "\n\n"
        f"Tasks Overdue / Stalled ({len(overdue_list)}):\n" + "\n".join(f"- {o}" for o in overdue_list[:10]) + "\n\n"
        f"Active Study Subjects ({len(subjects_list)}):\n" + "\n".join(f"- {s}" for s in subjects_list[:10]) + "\n\n"
        f"Resources & Papers Logged ({len(resources_list)}):\n" + "\n".join(f"- {r}" for r in resources_list[:15]) + "\n\n"
        "Please provide your structured Weekly Velocity Report."
    )

    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_DIGEST_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=WEEKLY_DIGEST_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=WeeklyVelocityReport,
                temperature=0.3,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, WeeklyVelocityReport):
                report = response.parsed
            else:
                report = WeeklyVelocityReport.model_validate(response.parsed)
        elif response.text:
            report = WeeklyVelocityReport.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Weekly Velocity synthesizer")
    except Exception as exc:
        logger.error("Gemini Weekly Velocity synthesis failed (%s). Using algorithmic fallback.", exc)
        total_comp = len(completed_list)
        total_pend = len(in_progress_list) + len(overdue_list)
        score = min(100, max(40, int((total_comp / max(1, total_comp + total_pend)) * 100)))
        report = WeeklyVelocityReport(
            velocity_score=score,
            verdict="Steady Execution" if score >= 70 else "Rebalancing Needed",
            headline=f"Completed {total_comp} tasks and maintained active study subjects.",
            tasks_completed_count=total_comp,
            tasks_pending_count=total_pend,
            completed_highlights=completed_list[:5],
            learning_progress=subjects_list[:4],
            bottlenecks=overdue_list[:3],
            next_week_priorities=["Advance core research subjects", "Clear pending high-priority tasks"],
        )

    report.tasks_completed_count = len(completed_list)
    report.tasks_pending_count = len(in_progress_list) + len(overdue_list)

    # Format clean WhatsApp / Telegram Markdown
    score_emoji = "⚡" if report.velocity_score >= 80 else "📈"
    lines = [
        f"📊 *Ocean Weekly Velocity Digest*",
        f"{score_emoji} *Score:* {report.velocity_score}/100 • *Verdict:* {report.verdict}",
        f"🎯 *Headline:* {clean_math_and_markdown(report.headline, for_whatsapp=True)}\n",
    ]

    if report.completed_highlights:
        lines.append("🏆 *Key Accomplishments:*")
        for h in report.completed_highlights[:4]:
            lines.append(f"• {clean_math_and_markdown(h, for_whatsapp=True)}")
        lines.append("")

    if report.learning_progress:
        lines.append("🏛️ *Research & Learning:*")
        for l in report.learning_progress[:3]:
            lines.append(f"• {clean_math_and_markdown(l, for_whatsapp=True)}")
        lines.append("")

    if report.bottlenecks:
        lines.append("⚠️ *Carried Over / Bottlenecks:*")
        for b in report.bottlenecks[:3]:
            lines.append(f"• {clean_math_and_markdown(b, for_whatsapp=True)}")
        lines.append("")

    if report.next_week_priorities:
        lines.append("🚀 *Strategic Priorities for Next Week:*")
        for p in report.next_week_priorities[:3]:
            lines.append(f"• {clean_math_and_markdown(p, for_whatsapp=True)}")

    report.full_digest_markdown = "\n".join(lines).strip()
    return report


def create_notion_weekly_review_page(
    report: WeeklyVelocityReport,
    notion_client: Optional[NotionAssistantClient] = None,
) -> Tuple[str, str]:
    """Create a structured Weekly Review page in Notion. Returns (page_id, page_url)."""
    notion = notion_client or NotionAssistantClient()
    client = notion.client
    if client is None:
        raise RuntimeError("Notion client is not initialized")

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    page_title = f"📈 Weekly Velocity Review — {today_str}"

    parent_id = notion.tasks_db_id or notion.subjects_db_id
    parent_obj = {"database_id": parent_id} if parent_id else {"page_id": parent_id}

    blocks: List[Dict[str, Any]] = [
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "icon": {"type": "emoji", "emoji": "⚡"},
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": f"Velocity Score: {report.velocity_score}/100 • {report.verdict}\n"},
                        "annotations": {"bold": True},
                    },
                    {
                        "type": "text",
                        "text": {"content": f"{clean_math_and_markdown(report.headline)}\n"},
                    },
                    {
                        "type": "text",
                        "text": {"content": f"Tasks Completed: {report.tasks_completed_count} | Pending/Carried Over: {report.tasks_pending_count}"},
                        "annotations": {"italic": True},
                    },
                ],
            },
        },
        {"object": "block", "type": "divider", "divider": {}},
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "🏆 Key Milestones & Accomplishments"}}]},
        },
    ]

    for h in report.completed_highlights:
        blocks.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": clean_math_and_markdown(h)}}]},
        })

    if report.learning_progress:
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "🏛️ Research & Study Velocity"}}]},
        })
        for l in report.learning_progress:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": clean_math_and_markdown(l)}}]},
            })

    if report.bottlenecks:
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "⚠️ Friction Points & Carryover Items"}}]},
        })
        for b in report.bottlenecks:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": clean_math_and_markdown(b)}}]},
            })

    if report.next_week_priorities:
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "🚀 Strategic Focus Areas for Next Week"}}]},
        })
        for idx, p in enumerate(report.next_week_priorities, 1):
            blocks.append({
                "object": "block",
                "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": [{"type": "text", "text": {"content": clean_math_and_markdown(p)}}]},
            })

    # Create page in Tasks DB or generic parent
    try:
        new_page = notion._request_with_retry(
            client.pages.create,
            parent={"database_id": notion.tasks_db_id},
            properties={
                "Task name": {"title": [{"text": {"content": page_title}}]},
                "Tags": {"multi_select": [{"name": "Miscellaneous"}]},
                "Status": {"status": {"name": "Done"}},
            },
            children=blocks,
        )
        page_id = new_page.get("id", "")
        page_url = _extract_page_url(new_page)
        return page_id, page_url
    except Exception as create_err:
        logger.warning("Could not create in Tasks DB (%s). Attempting generic page create.", create_err)
        new_page = notion._request_with_retry(
            client.pages.create,
            parent=parent_obj,
            properties={"title": [{"text": {"content": page_title}}]},
            children=blocks,
        )
        page_id = new_page.get("id", "")
        page_url = _extract_page_url(new_page)
        return page_id, page_url


def execute_weekly_digest_pipeline(
    to_phone: Optional[str] = None,
    chat_id: Optional[str] = None,
    days: int = 7,
    notion_client: Optional[NotionAssistantClient] = None,
    whatsapp_client: Optional[WhatsAppAssistantClient] = None,
    telegram_client: Optional[TelegramAssistantClient] = None,
) -> Dict[str, Any]:
    """Execute complete Sunday velocity digest pipeline, log to Notion, and send messages."""
    notion = notion_client or NotionAssistantClient()
    whatsapp = whatsapp_client or WhatsAppAssistantClient()
    telegram = telegram_client or TelegramAssistantClient()

    # 1. Fetch activity
    activity = fetch_past_week_workspace_activity(notion_client=notion, days=days)

    # 2. Synthesize with Gemini
    report = synthesize_velocity_digest(activity)

    # 3. Create Notion Weekly Review page
    page_id, page_url = "", ""
    try:
        page_id, page_url = create_notion_weekly_review_page(report, notion_client=notion)
        report.notion_page_url = page_url
    except Exception as page_err:
        logger.error("Failed to create Notion Weekly Review page (%s)", page_err)

    # Append Notion link to message if available
    delivery_text = report.full_digest_markdown
    if page_url:
        delivery_text += f"\n\n🔗 *Notion Weekly Review:* {page_url}"

    # 4. Deliver to WhatsApp
    if to_phone:
        try:
            whatsapp.send_message(to=to_phone, text=delivery_text, preview_url=bool(page_url))
            logger.info("Sent WhatsApp Weekly Digest to %s", to_phone)
        except Exception as wa_err:
            logger.error("Failed to send WhatsApp Weekly Digest: %s", wa_err)

    # 5. Deliver to Telegram
    if chat_id:
        try:
            telegram.send_message(text=delivery_text, chat_id=str(chat_id))
            logger.info("Sent Telegram Weekly Digest to %s", chat_id)
        except Exception as tg_err:
            logger.error("Failed to send Telegram Weekly Digest: %s", tg_err)

    return {
        "status": "ok",
        "velocity_score": report.velocity_score,
        "verdict": report.verdict,
        "headline": report.headline,
        "notion_page_url": page_url,
        "digest_text": delivery_text,
        "report": report,
    }
