import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from app.notion_client import NotionAssistantClient
from app.schemas import (
    LEARNING_TAG,
    LearningPlanSynthesis,
    LearningRequest,
    ResourceTypeLiteral,
    VerifiedResource,
)
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


def get_gemini_model() -> str:
    """Return the Gemini model identifier for grounding synthesis."""
    return os.getenv("GEMINI_GROUNDING_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))


# --- Step 2: Link Verification & Resource Type Inference ---

def verify_link_liveness(url: str, timeout: float = 6.0) -> Tuple[bool, Optional[int], Optional[str]]:
    """Perform a live HEAD request (or GET fallback if HEAD is blocked) to verify link liveness.

    Returns (is_valid, status_code, error_message).
    Only 2xx and 3xx statuses are treated as valid.
    """
    if not url or not url.startswith(("http://", "https://")):
        logger.warning("Dropped invalid URL format: %s", url)
        return False, None, "Invalid URL scheme"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            try:
                head_resp = client.head(url)
                status = head_resp.status_code
                if 200 <= status < 400:
                    return True, status, None
                # If HEAD is blocked or method not allowed (400, 403, 405), fallback to GET
                if status in (400, 403, 405):
                    get_resp = client.get(url)
                    status = get_resp.status_code
                    if 200 <= status < 400:
                        return True, status, None
                    logger.warning("Dropped link %s: GET returned status %s", url, status)
                    return False, status, f"HTTP {status}"
                else:
                    logger.warning("Dropped link %s: HEAD returned status %s", url, status)
                    return False, status, f"HTTP {status}"
            except (httpx.RequestError, httpx.HTTPStatusError) as req_err:
                # Attempt GET fallback on connection/request error with HEAD
                try:
                    get_resp = client.get(url)
                    status = get_resp.status_code
                    if 200 <= status < 400:
                        return True, status, None
                    logger.warning("Dropped link %s: GET fallback returned status %s", url, status)
                    return False, status, f"HTTP {status}"
                except Exception as get_err:
                    logger.warning("Dropped link %s: Unreachable (%s)", url, get_err)
                    return False, None, str(get_err)
    except Exception as exc:
        logger.warning("Dropped link %s: Exception during check (%s)", url, exc)
        return False, None, str(exc)


def infer_resource_type(url: str, title: str = "") -> ResourceTypeLiteral:
    """Infer resource type (Article/Video/Docs/Paper) from URL and title."""
    url_lower = url.lower()
    title_lower = title.lower()

    # Video detection
    video_domains = ["youtube.com", "youtu.be", "vimeo.com", "twitch.tv", "loom.com", "coursera.org", "udemy.com"]
    if any(vd in url_lower for vd in video_domains) or "video" in title_lower or "lecture" in title_lower:
        return "Video"

    # Paper detection
    paper_indicators = ["arxiv.org", "doi.org", "ieee.org", "acm.org", "semanticscholar.org", "nature.com", "biorxiv.org", "openreview.net"]
    if any(pi in url_lower for pi in paper_indicators) or url_lower.endswith(".pdf") or "paper" in title_lower or "thesis" in title_lower:
        return "Paper"

    # Docs detection
    docs_indicators = [
        "docs.", "documentation", "readthedocs.io", "manual",
        "developer.mozilla.org", "rust-lang.org/book", "devdocs.io",
        "/docs/", "/documentation/", "/manual/", "/tutorial/", "tutorial", "guide", "handbook", "reference"
    ]
    if any(di in url_lower for di in docs_indicators) or "documentation" in title_lower or "official guide" in title_lower or "tutorial" in title_lower or "guide" in title_lower:
        return "Docs"

    # Default to Article
    return "Article"


# --- Step 1: Gemini Grounding Curriculum Compiler ---

GROUNDING_SYSTEM_INSTRUCTION = (
    "You are an expert curriculum designer and educator. Given a topic to study, compile a comprehensive, novice-friendly study curriculum.\n"
    "Strict requirements:\n"
    "1. Generate a clear, concise SUBJECT TITLE for the topic.\n"
    "2. Generate a flat, continuous numbered list of individual curriculum topics (e.g. 1. Topic One\\n2. Topic Two\\n3. Topic Three). "
    "NO section headers, NO grouping or phases, each topic on its own line, novice-level framing throughout, built from the ground up, with NO corporate-style framing.\n"
    "3. Identify 1 to 3 concrete, immediately-actionable first steps (starter tasks: what the student should do first, not the whole curriculum).\n"
    "4. Search the web using grounding to find high-quality learning resources (official documentation, books, key articles, videos, foundational papers) and include their URLs."
)


def _extract_urls_and_titles_from_response(response: Any) -> List[Dict[str, str]]:
    """Extract surfaced URLs and titles from Gemini response grounding metadata and text."""
    resources: List[Dict[str, str]] = []
    seen_urls = set()

    # 1. Extract from grounding metadata chunks
    try:
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            gm = getattr(cand, "grounding_metadata", None)
            if gm:
                chunks = getattr(gm, "grounding_chunks", []) or []
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web:
                        uri = getattr(web, "uri", None)
                        title = getattr(web, "title", "") or ""
                        if uri and uri.startswith(("http://", "https://")) and uri not in seen_urls:
                            seen_urls.add(uri)
                            resources.append({"url": uri, "title": title.strip()})
    except Exception as exc:
        logger.debug("Could not extract grounding metadata chunks: %s", exc)

    # 2. Extract from response text via regex
    try:
        text = response.text or ""
        # Match markdown links [title](url)
        md_matches = re.findall(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", text)
        for title, url in md_matches:
            if url not in seen_urls:
                seen_urls.add(url)
                resources.append({"url": url, "title": title.strip()})

        # Match raw URLs
        raw_urls = re.findall(r"(https?://[^\s\)\],\"'<>]+)", text)
        for url in raw_urls:
            clean_url = url.rstrip(".,;:)")
            if clean_url not in seen_urls:
                seen_urls.add(clean_url)
                resources.append({"url": clean_url, "title": ""})
    except Exception as exc:
        logger.debug("Could not extract text URLs: %s", exc)

    return resources


def _parse_curriculum_text(text: str, fallback_topic: str) -> Tuple[str, List[str], List[str]]:
    """Parse subject title, flat numbered curriculum topics, and starter tasks from response text."""
    subject_title = fallback_topic
    curriculum_topics: List[str] = []
    starter_tasks: List[str] = []

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    in_starter_tasks = False

    for line in lines:
        lower = line.lower()
        if "starter task" in lower or "first step" in lower or "what to do first" in lower or "actionable task" in lower:
            in_starter_tasks = True
            continue

        if "subject title:" in lower:
            subject_title = re.sub(r"(?i)^#*\s*subject title:\s*", "", line).strip().strip('"*')
            continue
        elif line.startswith("# ") and not curriculum_topics and not in_starter_tasks:
            subject_title = line.lstrip("# ").strip().strip('"*')
            continue

        # Match numbered list item e.g. "1. Topic Name" or "1) Topic Name"
        num_match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if num_match:
            item_text = num_match.group(1).strip().strip("*_")
            # Remove any trailing parenthetical tags
            if in_starter_tasks:
                starter_tasks.append(item_text)
            else:
                curriculum_topics.append(item_text)
        elif line.startswith("- ") or line.startswith("* "):
            bullet_text = line[2:].strip().strip("*_")
            if in_starter_tasks:
                starter_tasks.append(bullet_text)
            elif not curriculum_topics:
                # If not inside starter tasks and no numbered items yet, treat as topic
                curriculum_topics.append(bullet_text)

    # Fallback if no numbered topics extracted
    if not curriculum_topics:
        curriculum_topics = [
            f"Core Concepts and Foundations of {fallback_topic}",
            f"Basic Setup, Architecture, and Building Blocks",
            f"Key Mechanisms, Protocols, and Algorithms",
            f"Hands-on Implementation and Practice",
            f"Advanced Concepts and Real-world Application",
        ]

    if not starter_tasks:
        starter_tasks = [
            f"Set up development environment and study notes for {fallback_topic}",
            f"Read foundational introductory overview on {fallback_topic}",
        ]

    return subject_title, curriculum_topics, starter_tasks


def compile_learning_curriculum(learning_req: LearningRequest) -> LearningPlanSynthesis:
    """Compile curriculum using Gemini web search grounding.

    Generates subject title, flat numbered curriculum list (novice framing, no headers),
    starter tasks, and surfaces reference URLs.
    """
    topic = learning_req.topic or "Computer Science Foundations"
    prompt = (
        f"Create a beginner-friendly study curriculum for: {topic}\n"
        f"Domain/Category: {learning_req.category or 'General'}\n"
        f"Goal: {learning_req.goal or 'Build a solid novice-to-intermediate understanding'}\n"
        f"Proficiency Level: {learning_req.proficiency_level or 'Novice'}\n"
        f"Requested Materials: {learning_req.resources_requested or 'Best free documentation, tutorials, papers'}\n\n"
        "Provide:\n"
        "SUBJECT TITLE: <A concise title for the subject>\n\n"
        "CURRICULUM TOPICS:\n"
        "1. <Topic 1>\n"
        "2. <Topic 2>\n"
        "3. <Topic 3>\n"
        "(A flat continuous numbered list of 5-10 individual topics, each on its own line, built from the ground up, no headers, no groupings)\n\n"
        "STARTER TASKS:\n"
        "1. <Concrete immediately-actionable first step>\n"
        "2. <Second actionable first step>\n"
    )

    client = get_gemini_client()
    model_name = get_gemini_model()

    try:
        # Search grounding tool config with client-side AFC disabled (handled server-side by Google)
        afc_config = (
            types.AutomaticFunctionCallingConfig(disable=True)
            if hasattr(types, "AutomaticFunctionCallingConfig")
            else None
        )
        config = types.GenerateContentConfig(
            system_instruction=GROUNDING_SYSTEM_INSTRUCTION,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            automatic_function_calling=afc_config,
            temperature=0.2,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        text_content = response.text or ""
        surfaced_resources = _extract_urls_and_titles_from_response(response)
        subj_title, topics, starter_tasks = _parse_curriculum_text(text_content, topic)

        return LearningPlanSynthesis(
            subject_title=subj_title,
            curriculum_topics=topics,
            starter_tasks=starter_tasks,
            surfaced_resources=surfaced_resources,
        )
    except Exception as exc:
        logger.error("Gemini search grounding compilation failed (%s). Using fallback synthesis.", exc)
        return LearningPlanSynthesis(
            subject_title=f"{topic} Fundamentals",
            curriculum_topics=[
                f"Introduction to {topic}",
                f"Basic Concepts and Fundamentals",
                f"Core Mechanics and Operations",
                f"Practical Exercises and Building",
                f"Next Steps and Advanced Concepts",
            ],
            starter_tasks=[
                f"Set up study workspace for {topic}",
                f"Review foundational documentation for {topic}",
            ],
            surfaced_resources=[],
        )


# --- Step 3 & 4: Background Pipeline Orchestrator ---

def execute_learning_background_pipeline(
    learning_req: LearningRequest,
    to_phone: Optional[str] = None,
    chat_id: Optional[str] = None,
    notion_client: Optional[NotionAssistantClient] = None,
    whatsapp_client: Optional[WhatsAppAssistantClient] = None,
    telegram_client: Optional[TelegramAssistantClient] = None,
) -> Dict[str, Any]:
    """Asynchronous background pipeline for LEARNING requests:

    1. Grounding synthesis call (separate from intent router)
    2. Per-link liveness checks & type inference
    3. Sequential Notion writes (Subject DB -> Resources DB -> Tasks DB) with fault tolerance
    4. Completion message back to WhatsApp/Telegram with Subject page link and summary.
    """
    logger.info("Starting background LEARNING pipeline for topic='%s'", learning_req.topic)

    notion = notion_client or NotionAssistantClient()
    whatsapp = whatsapp_client or WhatsAppAssistantClient()
    telegram = telegram_client or TelegramAssistantClient()

    # Step 1: Gemini Grounding call
    synthesis = compile_learning_curriculum(learning_req)
    subject_title = synthesis.subject_title or learning_req.topic or "Study Subject"

    # Step 2: Per-link live liveness checks
    verified_resources: List[VerifiedResource] = []
    dropped_links_count = 0

    for res in synthesis.surfaced_resources:
        url = res.get("url", "")
        raw_title = res.get("title", "")
        if not url:
            continue

        is_valid, status_code, err = verify_link_liveness(url)
        if is_valid:
            r_type = infer_resource_type(url, raw_title)
            display_title = raw_title or f"{subject_title} {r_type}"
            verified_resources.append(
                VerifiedResource(
                    name=display_title,
                    url=url,
                    resource_type=r_type,
                )
            )
        else:
            dropped_links_count += 1
            logger.info("Dropped invalid resource link: %s (status=%s, err=%s)", url, status_code, err)

    # Step 3a: Create Subject page in NOTION_SUBJECTS_DB_ID
    subject_page: Optional[Dict[str, Any]] = None
    subject_page_id: Optional[str] = None
    subject_url: Optional[str] = None

    try:
        subject_page = notion.create_subject_page(
            title=subject_title,
            curriculum_topics=synthesis.curriculum_topics,
        )
        subject_page_id = subject_page.get("id") if isinstance(subject_page, dict) else None
        subject_url = subject_page.get("url") if isinstance(subject_page, dict) else None
        if not subject_url and subject_page_id:
            clean_id = subject_page_id.replace("-", "")
            subject_url = f"https://www.notion.so/{clean_id}"
        logger.info("Created Notion Subject page '%s' (id=%s)", subject_title, subject_page_id)
    except Exception as subj_exc:
        logger.error("Failed to create Subject page in Notion: %s", subj_exc)
        # If Subject creation completely fails, report failure to WhatsApp/Telegram
        fail_msg = f"❌ Failed to create study plan for *{subject_title}*: {subj_exc}"
        if to_phone:
            try:
                whatsapp.send_message(to=to_phone, text=fail_msg)
            except Exception as wa_err:
                logger.error("Failed to send failure message to WhatsApp: %s", wa_err)
        if chat_id:
            try:
                telegram.send_message(text=fail_msg, chat_id=str(chat_id))
            except Exception as tg_err:
                logger.error("Failed to send failure message to Telegram: %s", tg_err)
        return {
            "status": "error",
            "stage": "create_subject_page",
            "error": str(subj_exc),
        }

    # Step 3b: Create rows in NOTION_RESOURCES_DB_ID
    total_resources = len(verified_resources)
    successful_resources = 0
    failed_resources: List[Tuple[str, str]] = []  # (resource_name, error_reason)

    if subject_page_id and verified_resources:
        for v_res in verified_resources:
            try:
                notion.create_resource_row(
                    name=v_res.name,
                    url=v_res.url,
                    resource_type=v_res.resource_type,
                    subject_page_id=subject_page_id,
                )
                successful_resources += 1
            except Exception as res_err:
                err_str = str(res_err)
                short_err = "rate limited" if "429" in err_str or "rate" in err_str.lower() else "write error"
                failed_resources.append((v_res.name, short_err))
                logger.warning("Failed to create resource row for '%s': %s", v_res.name, res_err)

    # Step 3c: Create starter tasks in NOTION_TASKS_DB_ID with Tags=['Learning']
    total_tasks = len(synthesis.starter_tasks)
    successful_tasks = 0
    failed_tasks: List[Tuple[str, str]] = []

    for task_title in synthesis.starter_tasks:
        try:
            notion.create_starter_task(
                title=task_title,
                subject_page_id=subject_page_id,
            )
            successful_tasks += 1
        except Exception as task_err:
            err_str = str(task_err)
            short_err = "rate limited" if "429" in err_str or "rate" in err_str.lower() else "write error"
            failed_tasks.append((task_title, short_err))
            logger.warning("Failed to create starter task '%s': %s", task_title, task_err)

    # Step 4: Construct Completion Message
    summary_parts = ["Subject created"]

    if total_resources > 0:
        if failed_resources:
            fail_reasons = ", ".join(set([r for _, r in failed_resources]))
            summary_parts.append(
                f"{successful_resources}/{total_resources} resources logged, {len(failed_resources)} failed ({fail_reasons})"
            )
        else:
            summary_parts.append(f"{successful_resources}/{total_resources} resources logged")
    elif dropped_links_count > 0:
        summary_parts.append(f"0 resources logged ({dropped_links_count} invalid links dropped)")

    if total_tasks > 0:
        if failed_tasks:
            summary_parts.append(f"{successful_tasks}/{total_tasks} starter tasks added ({len(failed_tasks)} failed)")
        else:
            summary_parts.append(f"{successful_tasks} starter tasks added")

    one_line_summary = ", ".join(summary_parts) + "."

    completion_message = (
        f"✅ *Study Plan Ready: {subject_title}*\n"
        f"🔗 {subject_url or ''}\n\n"
        f"{one_line_summary}"
    )

    # Send completion message
    if to_phone:
        try:
            whatsapp.send_message(to=to_phone, text=completion_message, preview_url=True)
            logger.info("Sent learning completion message to WhatsApp (to=%s)", to_phone)
        except Exception as wa_err:
            logger.error("Failed to send completion message to WhatsApp: %s", wa_err)

    if chat_id:
        try:
            telegram.send_message(text=completion_message, chat_id=str(chat_id))
            logger.info("Sent learning completion message to Telegram (chat_id=%s)", chat_id)
        except Exception as tg_err:
            logger.error("Failed to send completion message to Telegram: %s", tg_err)

    return {
        "status": "ok",
        "subject_title": subject_title,
        "subject_page_id": subject_page_id,
        "subject_url": subject_url,
        "resources_logged": successful_resources,
        "resources_failed": len(failed_resources),
        "tasks_added": successful_tasks,
        "tasks_failed": len(failed_tasks),
        "summary": one_line_summary,
    }
