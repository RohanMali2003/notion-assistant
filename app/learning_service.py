import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx

from app.ai import DEFAULT_GEMINI_MODEL, get_gemini_client, get_genai_types
from app.notion_client import NotionAssistantClient
from app.notifier import send_notification
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
    "You are an expert curriculum designer, research scientist, and educator. Given a topic to study, compile a comprehensive, high-quality study curriculum.\n"
    "Strict requirements:\n"
    "1. Generate a clear, concise, professional SUBJECT TITLE for the topic.\n"
    "2. Generate a flat, continuous numbered list of 4 to 8 individual curriculum topics (e.g. 1. **Topic Title:** Detailed explanation of what is covered).\n"
    "3. Identify 1 to 3 concrete, immediately-actionable first steps (starter tasks: what the student should read, code, or do first).\n"
    "4. In the RESOURCES section, provide 3 to 6 high-quality, canonical learning materials (seminal research papers with ArXiv/DOI URLs, official documentation links, GitHub repos, books, or top tutorials) using markdown links with a 1-sentence summary: - [Resource Title](https://...) — Brief summary of what this resource covers."
)


def _extract_urls_and_titles_from_response(response: Any) -> List[Dict[str, str]]:
    """Extract surfaced URLs, titles, and summaries from Gemini response text and grounding metadata."""
    resources: List[Dict[str, str]] = []
    seen_urls = set()

    # 1. Extract line-by-line from response text to capture title, url, and summary
    try:
        text = response.text or ""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            md_match = re.search(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)(?:\s*[-—:]\s*(.+))?", line)
            if md_match:
                title = md_match.group(1).strip()
                url = md_match.group(2).strip().rstrip(".,;:)")
                summary = (md_match.group(3) or "").strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    resources.append({"url": url, "title": title, "summary": summary})

        # Fallback regex for any markdown links not caught line-by-line
        for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", text):
            clean_url = url.rstrip(".,;:)")
            if clean_url and clean_url not in seen_urls:
                seen_urls.add(clean_url)
                resources.append({"url": clean_url, "title": title.strip(), "summary": ""})

        # Fallback regex for raw URLs
        for raw_url in re.findall(r"(https?://[^\s\)\],\"'<>]+)", text):
            clean_url = raw_url.rstrip(".,;:)")
            if clean_url and clean_url not in seen_urls:
                seen_urls.add(clean_url)
                resources.append({"url": clean_url, "title": "", "summary": ""})
    except Exception as exc:
        logger.debug("Could not extract text URLs: %s", exc)

    # 2. Extract from grounding metadata chunks if present
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
                            resources.append({"url": uri, "title": title.strip(), "summary": ""})
    except Exception as exc:
        logger.debug("Could not extract grounding metadata chunks: %s", exc)

    return resources


def _parse_curriculum_text(text: str, fallback_topic: str) -> Tuple[str, List[str], List[str]]:
    """Parse subject title, flat numbered curriculum topics, and starter tasks from response text."""
    subject_title = fallback_topic
    curriculum_topics: List[str] = []
    starter_tasks: List[str] = []

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    in_starter_tasks = False
    in_resources = False

    for line in lines:
        lower = line.lower()
        # Section header detection (must NOT be a numbered item or bullet)
        if not re.match(r"^\d+[\.\)]", line) and not line.startswith(("- ", "* ")):
            if any(k in lower for k in ("resource", "reference", "reading", "paper", "material", "link")):
                in_starter_tasks = False
                in_resources = True
                continue

            if "starter task" in lower or "first step" in lower or "what to do first" in lower or "actionable task" in lower:
                in_starter_tasks = True
                in_resources = False
                continue

            if "curriculum topic" in lower or "topics to cover" in lower or "syllabus" in lower:
                in_starter_tasks = False
                in_resources = False
                continue

        if "subject title:" in lower:
            subject_title = re.sub(r"(?i)^#*\s*subject title:\s*", "", line).strip().strip('"*')
            continue
        elif line.startswith("# ") and not curriculum_topics and not in_starter_tasks and not in_resources:
            subject_title = line.lstrip("# ").strip().strip('"*')
            continue

        if in_resources:
            continue

        # Match numbered list item e.g. "1. Topic Name" or "1) Topic Name"
        num_match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if num_match:
            item_text = num_match.group(1).strip().strip("*_")
            if in_starter_tasks:
                starter_tasks.append(item_text)
            else:
                curriculum_topics.append(item_text)
        elif line.startswith("- ") or line.startswith("* "):
            bullet_text = line[2:].strip().strip("*_")
            if in_starter_tasks:
                starter_tasks.append(bullet_text)
            elif not curriculum_topics:
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
    """Compile curriculum using Gemini with knowledge synthesis and search grounding fallback."""
    topic = learning_req.topic or "Computer Science Foundations"
    prompt = (
        f"Create a comprehensive study curriculum and gather foundational resources for: {topic}\n"
        f"Domain/Category: {learning_req.category or 'Computer Science / AI'}\n"
        f"Goal: {learning_req.goal or 'Master foundational concepts and seminal papers'}\n"
        f"Proficiency Level: {learning_req.proficiency_level or 'Novice to Advanced'}\n"
        f"Requested Materials: {learning_req.resources_requested or 'Seminal papers, official documentation, tutorials'}\n\n"
        "Provide your output formatted exactly as:\n"
        "SUBJECT TITLE: <A concise title for the subject>\n\n"
        "CURRICULUM TOPICS:\n"
        "1. <Topic 1>\n"
        "2. <Topic 2>\n"
        "3. <Topic 3>\n\n"
        "STARTER TASKS:\n"
        "1. <Concrete immediately-actionable first step>\n"
        "2. <Second actionable first step>\n\n"
        "RESOURCES:\n"
        "- [Canonical Paper / Resource 1](https://...)\n"
        "- [Official Docs / Resource 2](https://...)\n"
        "- [Tutorial / GitHub Repo 3](https://...)\n"
    )

    client = get_gemini_client()
    model_name = get_gemini_model()
    response = None
    gen_types = get_genai_types()

    # 1. Attempt with Search Grounding if configured
    if gen_types is not None:
        try:
            afc_config = (
                gen_types.AutomaticFunctionCallingConfig(disable=True)
                if hasattr(gen_types, "AutomaticFunctionCallingConfig")
                else None
            )
            config = gen_types.GenerateContentConfig(
                system_instruction=GROUNDING_SYSTEM_INSTRUCTION,
                tools=[gen_types.Tool(google_search=gen_types.GoogleSearch())],
                automatic_function_calling=afc_config,
                temperature=0.2,
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
        except Exception as ground_err:
            logger.warning(
                "Search grounding tool call failed (%s). Falling back to direct Gemini knowledge synthesis.",
                ground_err,
            )

    # 2. If search grounding failed or was skipped, call direct Gemini model
    if response is None or not (response.text or "").strip():
        try:
            config_direct = None
            if gen_types is not None:
                config_direct = gen_types.GenerateContentConfig(
                    system_instruction=GROUNDING_SYSTEM_INSTRUCTION,
                    temperature=0.2,
                )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config_direct,
            )
        except Exception as direct_err:
            logger.error("Direct Gemini curriculum generation failed: %s", direct_err)


    if response and (response.text or "").strip():
        text_content = response.text or ""
        surfaced_resources = _extract_urls_and_titles_from_response(response)
        subj_title, topics, starter_tasks = _parse_curriculum_text(text_content, topic)

        return LearningPlanSynthesis(
            subject_title=subj_title,
            curriculum_topics=topics,
            starter_tasks=starter_tasks,
            surfaced_resources=surfaced_resources,
        )

    logger.error("All Gemini API attempts failed. Using static fallback synthesis.")
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
        summary = res.get("summary", "")
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
                    summary=summary or None,
                )
            )
        else:
            dropped_links_count += 1
            logger.info("Dropped invalid resource link: %s (status=%s, err=%s)", url, status_code, err)

    # Step 3a: Create Subject page in NOTION_SUBJECTS_DB_ID with rich sections
    subject_page: Optional[Dict[str, Any]] = None
    subject_page_id: Optional[str] = None
    subject_url: Optional[str] = None

    try:
        subject_page = notion.create_subject_page(
            title=subject_title,
            curriculum_topics=synthesis.curriculum_topics,
            resources=verified_resources,
            starter_tasks=synthesis.starter_tasks,
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

    try:
        from app.motion import evidence_ingestion_engine
        evidence_ingestion_engine.ingest_learning_milestone(
            topic_title=subject_title,
            summary=one_line_summary,
            page_url=subject_url,
            duration_hours=1.5,
        )
    except Exception as err:
        logger.debug("Motion evidence ingestion skipped: %s", err)

    # Send completion message
    send_notification(
        completion_message,
        to_phone=to_phone,
        chat_id=chat_id,
        preview_url=True,
        whatsapp_client=whatsapp,
        telegram_client=telegram,
    )


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
