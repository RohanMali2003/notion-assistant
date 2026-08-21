"""1-Tap Drop & Digest URL Ingestion and Synthesis Service for Ocean."""

import html
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx

from app.ai import DEFAULT_GEMINI_MODEL, generate_text, get_gemini_client, get_gemini_model
from app.notion_client import NotionAssistantClient
from app.tag_directory import CANONICAL_TAG_NAMES, match_closest_tag

logger = logging.getLogger(__name__)
URL_REGEX = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)



def extract_urls(text: str) -> List[str]:
    """Extract all HTTP/HTTPS URLs from a given text string."""
    if not text:
        return []
    matches = URL_REGEX.findall(text)
    # Strip trailing punctuation often accidentally included (e.g. '.', ')', ',')
    cleaned_urls = []
    for m in matches:
        clean = m.rstrip(".,;!?)>]\'\"")
        if clean and clean not in cleaned_urls:
            cleaned_urls.append(clean)
    return cleaned_urls


def is_url_dominant_message(text: str) -> bool:
    """Return True if message contains a URL and minimal surrounding text."""
    urls = extract_urls(text)
    if not urls:
        return False
    # If text is basically just the URL with <= 6 words of commentary
    stripped = text
    for u in urls:
        stripped = stripped.replace(u, "")
    words = stripped.strip().split()
    return len(words) <= 8


from app.ai import DEFAULT_GEMINI_MODEL, get_gemini_client, get_gemini_model
from app.notifier import send_notification



def fetch_url_content(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """Fetch metadata and main body text from an external URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    # 1. Specialized ArXiv Handler
    if "arxiv.org" in url.lower():
        arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", url, re.IGNORECASE)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            try:
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    resp = client.get(abs_url, headers=headers)
                    if resp.status_code == 200:
                        content = resp.text
                        # Title
                        title_m = re.search(r'<h1 class="title mathjax"><span class="descriptor">Title:</span>(.*?)</h1>', content, re.DOTALL | re.IGNORECASE)
                        title = html.unescape(title_m.group(1).strip()) if title_m else f"ArXiv Paper {arxiv_id}"
                        # Abstract
                        abs_m = re.search(r'<blockquote class="abstract mathjax"><span class="descriptor">Abstract:</span>(.*?)</blockquote>', content, re.DOTALL | re.IGNORECASE)
                        abstract = html.unescape(abs_m.group(1).strip()) if abs_m else ""
                        # Authors
                        authors_m = re.search(r'<div class="authors"><span class="descriptor">Authors:</span>(.*?)</div>', content, re.DOTALL | re.IGNORECASE)
                        authors_raw = re.sub(r'<[^>]+>', '', authors_m.group(1)).strip() if authors_m else ""

                        return {
                            "url": url,
                            "canonical_url": abs_url,
                            "title": title,
                            "authors": authors_raw,
                            "raw_text": f"Paper Title: {title}\nAuthors: {authors_raw}\nAbstract:\n{abstract}",
                            "inferred_format": "Paper",
                            "source": "ArXiv",
                        }
            except Exception as arxiv_err:
                logger.warning("ArXiv fetch failed for %s: %s", url, arxiv_err)

    # 2. General Web / GitHub / Substack / Blog Handler
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return {
                    "url": url,
                    "title": url,
                    "raw_text": f"URL: {url} (HTTP {resp.status_code})",
                    "inferred_format": "Article",
                    "source": "Web",
                }

            html_content = resp.text

            # Extract Title
            title = ""
            og_title_m = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']', html_content, re.IGNORECASE)
            if og_title_m:
                title = html.unescape(og_title_m.group(1).strip())
            else:
                title_m = re.search(r'<title>(.*?)</title>', html_content, re.DOTALL | re.IGNORECASE)
                if title_m:
                    title = html.unescape(title_m.group(1).strip())

            # Extract Description
            desc = ""
            og_desc_m = re.search(r'<meta\s+(?:property=["\']og:description["\']|name=["\']description["\'])\s+content=["\'](.*?)["\']', html_content, re.IGNORECASE)
            if og_desc_m:
                desc = html.unescape(og_desc_m.group(1).strip())

            # Clean readable text
            body_text = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
            body_text = re.sub(r"<[^>]+>", " ", body_text)
            body_text = re.sub(r"\s+", " ", body_text).strip()
            # Truncate to first 6000 chars for LLM context
            body_snippet = body_text[:6000]

            inferred_fmt = "Article"
            if "github.com" in url:
                inferred_fmt = "Repository"
            elif "youtube.com" in url or "youtu.be" in url:
                inferred_fmt = "Video"
            elif "arxiv.org" in url or "openreview.net" in url:
                inferred_fmt = "Paper"

            return {
                "url": url,
                "title": title or url,
                "description": desc,
                "raw_text": f"Title: {title}\nDescription: {desc}\n\nContent Excerpt:\n{body_snippet}",
                "inferred_format": inferred_fmt,
                "source": "Web",
            }
    except Exception as exc:
        logger.warning("Generic URL fetch failed for %s: %s", url, exc)
        return {
            "url": url,
            "title": url,
            "raw_text": f"URL: {url}",
            "inferred_format": "Article",
            "source": "Web",
        }


URL_SYNTHESIS_SYSTEM_INSTRUCTION = (
    "You are an elite researcher and knowledge curator for Ocean.\n"
    "Your goal is to digest, deeply understand, and synthesize material from an external URL into a high-insight, comprehensive summary.\n\n"
    "Strict Requirements:\n"
    "1. FOCUS ON QUALITY AND DEPTH: Do NOT artificially compress your analysis into tiny generic bullet points. Capture the true essence, core ideas, architectural/methodological innovations, and practical implications of the material.\n"
    "2. FORMATTING RULES:\n"
    "   - DO NOT use LaTeX math notation or dollar signs (write O(N), 10^9, N directly, NEVER $O(N)$ or $10^9$).\n"
    "   - DO NOT use double asterisks (**). Use clean markdown formatting.\n"
    "3. STRUCTURED OUTPUT FORMAT:\n"
    "TITLE: <Clean, canonical title of the paper/article/tool>\n"
    "FORMAT: <Paper / Article / Docs / Repository / Video>\n"
    "DOMAIN: <Best matching domain tag, e.g. AI Research, System Design, Distributed Systems, Open Source, Career, etc.>\n"
    "ESSENCE: <1-2 sentences capturing the high-level purpose and core thesis>\n"
    "KEY_TAKEAWAYS:\n"
    "- <Insightful takeaway 1>\n"
    "- <Insightful takeaway 2>\n"
    "- <Insightful takeaway 3>\n"
    "- <Insightful takeaway 4 (if applicable)>\n"
    "PRACTICAL_IMPLICATIONS: <1-2 sentences on how this is applied or why it matters>"
)


def summarize_and_log_url(
    url: str,
    user_comment: str = "",
    notion_client: Optional[NotionAssistantClient] = None,
) -> Dict[str, Any]:
    """Fetch URL, synthesize comprehensive summary with Gemini, and log into Notion."""
    notion = notion_client or NotionAssistantClient()
    fetched = fetch_url_content(url)

    client = get_gemini_client()
    prompt = (
        f"URL: {url}\n"
        f"User's Note / Context: {user_comment or 'None'}\n\n"
        f"Fetched Content:\n{fetched.get('raw_text', '')}\n\n"
        "Please provide your structured synthesis following the system instruction."
    )

    try:
        model_target = os.getenv("GEMINI_URL_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
        resp_text = generate_text(
            prompt=prompt,
            system_instruction=URL_SYNTHESIS_SYSTEM_INSTRUCTION,
            model=model_target,
            temperature=0.2,
            fallback_default="",
        )
        if not resp_text:
            # Fallback direct call if generate_text yielded empty
            resp = client.models.generate_content(
                model=model_target,
                contents=prompt,
            )
            resp_text = resp.text or ""
    except Exception as exc:
        logger.error("Gemini URL synthesis failed (%s). Using fallback.", exc)
        resp_text = ""


    # Parse fields from Gemini response
    title = fetched.get("title", url)
    title_m = re.search(r"TITLE:\s*(.+)", resp_text, re.IGNORECASE)
    if title_m:
        title = title_m.group(1).strip()

    res_format = fetched.get("inferred_format", "Article")
    fmt_m = re.search(r"FORMAT:\s*([A-Za-z]+)", resp_text, re.IGNORECASE)
    if fmt_m:
        res_format = fmt_m.group(1).strip().capitalize()

    domain_tag = match_closest_tag(None, f"{title} {fetched.get('raw_text', '')}")
    tag_m = re.search(r"DOMAIN:\s*(.+)", resp_text, re.IGNORECASE)
    if tag_m:
        domain_tag = match_closest_tag(tag_m.group(1).strip(), title)

    essence = ""
    essence_m = re.search(r"ESSENCE:\s*(.*?)(?=KEY_TAKEAWAYS:|$)", resp_text, re.DOTALL | re.IGNORECASE)
    if essence_m:
        essence = essence_m.group(1).strip()

    takeaways = []
    takeaways_m = re.search(r"KEY_TAKEAWAYS:\s*(.*?)(?=PRACTICAL_IMPLICATIONS:|$)", resp_text, re.DOTALL | re.IGNORECASE)
    if takeaways_m:
        for line in takeaways_m.group(1).strip().split("\n"):
            clean_l = line.strip().lstrip("-*•0123456789. ")
            if clean_l:
                takeaways.append(clean_l)

    implications = ""
    imp_m = re.search(r"PRACTICAL_IMPLICATIONS:\s*(.*)$", resp_text, re.DOTALL | re.IGNORECASE)
    if imp_m:
        implications = imp_m.group(1).strip()

    # Build comprehensive summary text for Notion and WhatsApp
    summary_parts = []
    if essence:
        summary_parts.append(f"💡 *Core Essence:*\n{essence}")
    if takeaways:
        takeaway_str = "\n".join([f"• {t}" for t in takeaways])
        summary_parts.append(f"🔍 *Key Takeaways & Contributions:*\n{takeaway_str}")
    if implications:
        summary_parts.append(f"⚙️ *Practical Implications:*\n{implications}")

    full_summary = "\n\n".join(summary_parts) if summary_parts else fetched.get("description", "Material logged from external URL.")

    # Log into Notion Resources DB
    notion_url = None
    notion_page_id = None
    try:
        created_resource = notion.create_resource_row(
            title=title,
            url=url,
            resource_type=res_format,
            summary=essence or full_summary[:300],
            tags=[domain_tag],
        )
        notion_page_id = created_resource.get("id") if isinstance(created_resource, dict) else None
        notion_url = created_resource.get("url") if isinstance(created_resource, dict) else None
        if not notion_url and notion_page_id:
            clean_id = str(notion_page_id).replace("-", "")
            notion_url = f"https://www.notion.so/{clean_id}"
    except Exception as notion_err:
        logger.error("Failed to log resource to Notion (%s)", notion_err)

    # Format delivery message for WhatsApp / Telegram
    msg_lines = [
        f"🔗 *Digested Material:* {title}",
        f"📌 *Domain:* {domain_tag} | 📄 *Type:* {res_format}\n",
        full_summary,
    ]
    if notion_url:
        msg_lines.append(f"\n🔗 *Notion Resource:* {notion_url}")

    reply_text = "\n".join(msg_lines).strip()

    return {
        "status": "ok",
        "url": url,
        "title": title,
        "domain_tag": domain_tag,
        "format": res_format,
        "essence": essence,
        "takeaways": takeaways,
        "implications": implications,
        "summary": full_summary,
        "notion_url": notion_url,
        "reply_text": reply_text,
    }


def execute_url_digest_background_pipeline(
    url: str,
    user_comment: str = "",
    to_phone: Optional[str] = None,
    chat_id: Optional[str] = None,
    notion_client: Optional[NotionAssistantClient] = None,
    whatsapp_client: Optional[Any] = None,
    telegram_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Digest a URL in background, create Notion resource, and send confirmation to WhatsApp / Telegram."""
    notion = notion_client or NotionAssistantClient()
    whatsapp = whatsapp_client
    telegram = telegram_client

    summary_data = summarize_and_log_url(url=url, user_comment=user_comment, notion_client=notion)
    reply_text = summary_data.get("reply_text", "")
    notion_url = summary_data.get("notion_url")

    if reply_text:
        send_notification(
            reply_text,
            to_phone=to_phone,
            chat_id=chat_id,
            preview_url=bool(notion_url),
            whatsapp_client=whatsapp_client,
            telegram_client=telegram_client,
        )

    return summary_data

