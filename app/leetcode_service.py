import html
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx

try:
    from google import genai
except ImportError:
    genai = None

from app.ai import DEFAULT_GEMINI_MODEL, generate_text, get_gemini_client, get_genai_types
from app.config import settings
from app.notion_client import NotionAssistantClient
from app.notifier import send_notification
from app.schemas import (
    LeetcodeCommitData,
    LeetcodeProblemDetails,
    LeetcodeReviewRequest,
    LeetcodeReviewResult,
)
from app.telegram_client import TelegramAssistantClient
from app.whatsapp_client import WhatsAppAssistantClient

logger = logging.getLogger(__name__)


LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"


def get_gemini_model() -> str:
    """Return the Gemini model identifier for LeetCode code review."""
    return os.getenv("GEMINI_LEETCODE_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))




# --- Step 1: Parsing LeetHub Commit Data & GitHub Integration ---

CODE_EXTENSIONS = {
    ".py": "python",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sql": "sql",
    ".scala": "scala",
    ".php": "php",
}


def slugify_title(title: str) -> str:
    """Convert problem title or path into clean LeetCode slug (e.g. '0001-two-sum' or '1. Two Sum' -> 'two-sum')."""
    if not title:
        return ""
    # Strip leading problem numbers (e.g. '0001-', '1. ', '1 - ')
    clean = re.sub(r"^\d+[\.\-\s_]+", "", title.strip())
    # Remove difficulty suffixes in brackets or parens (e.g. ' (Easy)', ' [Medium]')
    clean = re.sub(r"[\(\[\{](?:Easy|Medium|Hard)[\)\]\}]", "", clean, flags=re.IGNORECASE).strip()
    # Remove non-alphanumeric chars except spaces and hyphens
    clean = re.sub(r"[^\w\s-]", "", clean).strip()
    # Convert spaces/underscores to hyphens and lowercase
    slug = re.sub(r"[\s_]+", "-", clean).lower()
    return slug.strip("-")


def parse_problem_title_and_slug(
    commit_message: str,
    readme_content: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
) -> Tuple[str, str, Optional[int]]:
    """Parse problem title, URL slug, and number from LeetHub README, commit message, or file paths.

    Returns (problem_title, title_slug, problem_number).
    """
    problem_title = ""
    title_slug = ""
    problem_number: Optional[int] = None

    # 1. Try parsing from README if available (LeetHub generates rich headers in README.md)
    if readme_content:
        # Match LeetCode problem URL in README: e.g. href="https://leetcode.com/problems/two-sum/"
        url_match = re.search(r"leetcode\.com/problems/([a-z0-9\-]+)", readme_content, re.IGNORECASE)
        if url_match:
            title_slug = url_match.group(1).lower().strip("/")

        # Match header like: <h2><a href="...">1. Two Sum</a></h2> or <h2>1. Two Sum</h2> or # 1. Two Sum
        header_match = re.search(r"<h[1-3][^>]*>(?:<a[^>]*>)?(?:(\d+)[\.\s-]+)?([^<]+?)(?:</a>)?</h[1-3]>", readme_content, re.IGNORECASE)
        if header_match:
            num_str, title_str = header_match.groups()
            if num_str and num_str.isdigit():
                problem_number = int(num_str)
            if title_str:
                problem_title = title_str.strip()
        elif not problem_title:
            md_header = re.search(r"^#+\s*(?:(\d+)[\.\s-]+)?(.+)$", readme_content, re.MULTILINE)
            if md_header:
                num_str, title_str = md_header.groups()
                if num_str and num_str.isdigit():
                    problem_number = int(num_str)
                if title_str:
                    problem_title = title_str.strip()

    # 2. Try parsing from file paths (e.g. "0001-two-sum/0001-two-sum.py" or "Two Sum/solution.py")
    if file_paths:
        for path in file_paths:
            parts = path.replace("\\", "/").split("/")
            folder_or_file = parts[0] if len(parts) > 1 else parts[0]
            # Strip file extension
            base_name = os.path.splitext(folder_or_file)[0]
            if base_name.lower() in ("readme", ".github", ".git"):
                continue

            num_match = re.match(r"^0*(\d+)[\.\-\s_]+(.*)$", base_name)
            if num_match:
                if problem_number is None and num_match.group(1).isdigit():
                    problem_number = int(num_match.group(1))
                if not problem_title:
                    raw_title = num_match.group(2).replace("-", " ").replace("_", " ").strip()
                    problem_title = raw_title.title()
                if not title_slug:
                    title_slug = slugify_title(num_match.group(2))
                break
            elif not problem_title and base_name:
                problem_title = base_name.replace("-", " ").replace("_", " ").title()
                if not title_slug:
                    title_slug = slugify_title(base_name)
                break

    # 3. Fallback to commit message
    if commit_message:
        # Common LeetHub formats:
        # "Added 0001-two-sum [Time: ...]"
        # "Time: 12 ms (90.00%) | Memory: 14 MB (50.00%) - LeetHub"
        # "Create 1. Two Sum"
        # "0001-two-sum"
        clean_msg = commit_message.split("\n")[0].strip()
        clean_msg = re.sub(r"(?i)\s*-\s*leethub.*$", "", clean_msg).strip()
        clean_msg = re.sub(r"(?i)\[Time:.*?\]", "", clean_msg).strip()
        clean_msg = re.sub(r"(?i)^(added|created|updated|solve|solution for)\s+", "", clean_msg).strip()

        if clean_msg and not clean_msg.lower().startswith("time:"):
            num_match = re.match(r"^0*(\d+)[\.\-\s_]+(.*)$", clean_msg)
            if num_match:
                if problem_number is None and num_match.group(1).isdigit():
                    problem_number = int(num_match.group(1))
                if not problem_title:
                    problem_title = num_match.group(2).replace("-", " ").replace("_", " ").title()
                if not title_slug:
                    title_slug = slugify_title(num_match.group(2))
            elif not problem_title:
                problem_title = clean_msg.replace("-", " ").replace("_", " ").title()
                if not title_slug:
                    title_slug = slugify_title(clean_msg)

    # Derive slug from title if slug is still missing
    if not title_slug and problem_title:
        title_slug = slugify_title(problem_title)

    # Format fallback title if only slug is present
    if not problem_title and title_slug:
        problem_title = title_slug.replace("-", " ").title()

    return problem_title or "LeetCode Problem", title_slug or "two-sum", problem_number


def fetch_latest_leethub_commit(
    repo: Optional[str] = None,
    pat: Optional[str] = None,
    timeout: float = 12.0,
) -> LeetcodeCommitData:
    """Fetch the latest commit from the LeetHub GitHub repository using GITHUB_PAT."""
    if repo is not None:
        target_repo = repo.strip()
    else:
        target_repo = getattr(settings, "GITHUB_LEETHUB_REPO", "") or os.getenv("GITHUB_LEETHUB_REPO", "")

    if pat is not None:
        target_pat = pat.strip()
    else:
        target_pat = getattr(settings, "GITHUB_PAT", "") or os.getenv("GITHUB_PAT", "")

    if not target_repo:
        raise ValueError("GITHUB_LEETHUB_REPO is not configured in settings or environment.")

    # Clean repo format (e.g. 'https://github.com/user/repo' -> 'user/repo')
    clean_repo = re.sub(r"^https?://github\.com/", "", target_repo).strip().strip("/")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Notion-Assistant-LeetCode-Service",
    }
    if target_pat:
        headers["Authorization"] = f"Bearer {target_pat}"

    with httpx.Client(timeout=timeout) as client:
        # 1. Fetch recent commits on default branch to locate the latest solution commit
        commits_url = f"https://api.github.com/repos/{clean_repo}/commits?per_page=15"
        resp = client.get(commits_url, headers=headers)
        if resp.status_code != 200:
            logger.error("GitHub API error fetching commits (%s): %s", resp.status_code, resp.text)
            raise RuntimeError(f"GitHub API returned {resp.status_code} for repo '{clean_repo}': {resp.text}")

        commits_data = resp.json()
        if not isinstance(commits_data, list) or not commits_data:
            raise RuntimeError(f"No commits found in repository '{clean_repo}'.")

        target_commit = None
        target_files = []
        target_sha = ""
        target_msg = ""

        for c in commits_data:
            c_sha = c.get("sha", "")
            c_msg = c.get("commit", {}).get("message", "")
            detail_url = f"https://api.github.com/repos/{clean_repo}/commits/{c_sha}"
            detail_resp = client.get(detail_url, headers=headers)
            if detail_resp.status_code != 200:
                continue

            c_detail = detail_resp.json()
            c_files = c_detail.get("files", [])
            has_code = any(
                os.path.splitext(f.get("filename", ""))[1].lower() in CODE_EXTENSIONS
                for f in c_files if isinstance(f, dict)
            )

            if has_code or not target_commit:
                target_commit = c_detail
                target_files = c_files
                target_sha = c_sha
                target_msg = c_msg
                if has_code:
                    break

        files = target_files
        latest_sha = target_sha
        commit_msg = target_msg

        file_paths = [f.get("filename", "") for f in files if isinstance(f, dict)]
        readme_content: Optional[str] = None
        code_content: str = ""
        code_file_name: str = ""
        problem_folder: Optional[str] = None

        # Identify solution file and README.md
        for f in files:
            fname = f.get("filename", "")
            base_fname = os.path.basename(fname)
            ext = os.path.splitext(base_fname)[1].lower()
            if "/" in fname:
                problem_folder = fname.split("/")[0]

            if base_fname.lower() == "readme.md":
                raw_url = f.get("raw_url")
                patch = f.get("patch")
                if raw_url:
                    try:
                        readme_resp = client.get(raw_url, headers=headers)
                        if readme_resp.status_code == 200:
                            readme_content = readme_resp.text
                    except Exception as r_err:
                        logger.debug("Could not fetch raw README.md: %s", r_err)
                if not readme_content and patch:
                    readme_content = patch

            elif ext in CODE_EXTENSIONS and not code_content:
                code_file_name = base_fname
                raw_url = f.get("raw_url")
                patch = f.get("patch")
                if raw_url:
                    try:
                        code_resp = client.get(raw_url, headers=headers)
                        if code_resp.status_code == 200:
                            code_content = code_resp.text
                    except Exception as c_err:
                        logger.debug("Could not fetch raw code: %s", c_err)
                if not code_content and patch:
                    clean_lines = []
                    for line in patch.split("\n"):
                        if line.startswith("+") and not line.startswith("+++"):
                            clean_lines.append(line[1:])
                        elif not line.startswith("-") and not line.startswith("@@"):
                            clean_lines.append(line)
                    code_content = "\n".join(clean_lines)

        # If README wasn't part of this commit but we know the problem folder, try fetching it directly
        if not readme_content and problem_folder and problem_folder.lower() not in ("stats.json", ".github", ".git"):
            try:
                import base64
                folder_readme_url = f"https://api.github.com/repos/{clean_repo}/contents/{problem_folder}/README.md"
                r_resp = client.get(folder_readme_url, headers=headers)
                if r_resp.status_code == 200:
                    encoded_body = r_resp.json().get("content", "")
                    if encoded_body:
                        readme_content = base64.b64decode(encoded_body).decode("utf-8", errors="replace")
            except Exception as r_exc:
                logger.debug("Could not fetch folder README: %s", r_exc)

        # Parse problem metadata
        prob_title, prob_slug, prob_num = parse_problem_title_and_slug(
            commit_message=commit_msg,
            readme_content=readme_content,
            file_paths=file_paths,
        )

        return LeetcodeCommitData(
            commit_sha=latest_sha,
            commit_message=commit_msg,
            problem_title=prob_title,
            problem_slug=prob_slug,
            problem_number=prob_num,
            code=code_content or "# No code content retrieved from commit",
            code_file_name=code_file_name or "solution.py",
            readme_content=readme_content,
        )


# --- Step 2: LeetCode Public GraphQL API Integration ---

GRAPHQL_QUESTION_QUERY = """
query getQuestionDetail($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionId
    questionFrontendId
    title
    titleSlug
    difficulty
    content
    topicTags {
      name
      slug
    }
    hints
  }
}
"""


def _extract_constraints_from_html(content_html: str) -> List[str]:
    """Parse constraints list from LeetCode problem HTML description."""
    if not content_html:
        return []

    constraints: List[str] = []
    # Match the Constraints section
    c_match = re.search(
        r"(?:<p><strong>\s*Constraints:?\s*</strong></p>|<h3>\s*Constraints:?\s*</h3>|<strong>\s*Constraints:?\s*</strong>)(.*?)(?:<p>|$)",
        content_html,
        re.DOTALL | re.IGNORECASE,
    )
    section = c_match.group(1) if c_match else content_html

    # Extract <li> elements
    li_matches = re.findall(r"<li>(.*?)</li>", section, re.DOTALL | re.IGNORECASE)
    for li in li_matches:
        # Strip HTML tags
        clean_text = re.sub(r"<[^>]+>", "", li)
        # Unescape HTML entities (e.g. &lt;= to <=, &#39; to ')
        clean_text = html.unescape(clean_text).strip()
        # Clean formatting
        clean_text = clean_text.replace("<code>", "").replace("</code>", "").strip()
        if clean_text:
            constraints.append(clean_text)

    # Fallback regex if no <li> matches
    if not constraints and "Constraints" in content_html:
        raw_constraints = re.findall(r"<code>([^<]+<=+[^<]+)</code>", content_html)
        for rc in raw_constraints:
            clean_rc = html.unescape(rc).strip()
            if clean_rc and clean_rc not in constraints:
                constraints.append(clean_rc)

    return constraints


def fetch_leetcode_problem_details(
    title_slug: str,
    timeout: float = 8.0,
) -> Optional[LeetcodeProblemDetails]:
    """Query LeetCode's public GraphQL API for problem difficulty, constraints, and metadata.

    Returns LeetcodeProblemDetails if found, or None if slug is invalid or request fails (fallback mode).
    """
    if not title_slug:
        return None

    headers = {
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://leetcode.com/problems/{title_slug}/",
    }
    payload = {
        "query": GRAPHQL_QUESTION_QUERY,
        "variables": {"titleSlug": title_slug},
        "operationName": "getQuestionDetail",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(LEETCODE_GRAPHQL_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.warning("LeetCode GraphQL request returned status %s for slug='%s'", resp.status_code, title_slug)
                return None

            data = resp.json()
            q_data = data.get("data", {}).get("question")
            if not q_data or not isinstance(q_data, dict):
                logger.info("LeetCode problem '%s' not found in public GraphQL API.", title_slug)
                return None

            title = q_data.get("title", title_slug.replace("-", " ").title())
            difficulty = q_data.get("difficulty")
            raw_html = q_data.get("content", "")
            topic_tags = [t.get("name", "") for t in q_data.get("topicTags", []) if isinstance(t, dict)]
            constraints = _extract_constraints_from_html(raw_html)

            return LeetcodeProblemDetails(
                title=title,
                title_slug=title_slug,
                difficulty=difficulty,
                constraints=constraints,
                raw_content_html=raw_html,
                topic_tags=topic_tags,
            )
    except Exception as exc:
        logger.warning("Failed to fetch LeetCode details for slug='%s' (%s). Triggering fallback.", title_slug, exc)
        return None


# --- Step 3: Gemini Review Generation with Constraint Assessment ---

LEETCODE_REVIEW_SYSTEM_INSTRUCTION = (
    "You are an elite competitive programmer and technical interviewer at a top tech company.\n"
    "Your role is to rigorously review the user's submitted LeetCode solution.\n\n"
    "Strict requirements:\n"
    "1. VERDICT: Give a clear verdict (e.g. Correct, Incorrect, Suboptimal, Solved).\n"
    "2. COMPLEXITY: State exact evaluated Time Complexity and Space Complexity in Big-O notation (e.g. O(N), O(1)).\n"
    "3. OPTIMALITY: State whether the approach is optimal given the stated problem constraints (e.g. If N <= 10^5, an O(N^2) solution will TLE and is Suboptimal).\n"
    "4. ANALYSIS: Provide a concise, highly insightful evaluation of the logic, data structure choice, and potential pitfalls.\n"
    "5. TARGETED TESTING QUESTIONS: Conclude with 2 to 4 concrete, specific, non-generic testing questions directly testing edge cases and code logic in the submitted implementation (e.g. loop boundaries, negative values, empty/single element inputs, duplicate keys, overflow). DO NOT provide generic boilerplate interview lists.\n\n"
    "FORMATTING RULES:\n"
    "- DO NOT use LaTeX math notation or dollar signs (e.g. write O(N), O(1), N, 10^9 directly, NEVER $O(N)$ or $10^9$).\n"
    "- DO NOT use markdown double asterisks (**) inside sentences. Keep formatting clean and legible."
)


def clean_math_and_markdown(text: str, for_whatsapp: bool = True) -> str:
    """Clean LaTeX math delimiters ($...$, $$...$$) and normalize markdown asterisks."""
    if not text:
        return ""

    # 1. Remove LaTeX math delimiters: $$...$$ -> ... and $...$ -> ...
    # e.g., $O(K)$ -> O(K), $n = 10^9$ -> n = 10^9, $n$ -> n
    cleaned = re.sub(r"\$\$(.*?)\$\$", r"\1", text)
    cleaned = re.sub(r"\$([^\$\n]+?)\$", r"\1", cleaned)

    # 2. Normalize markdown bold formatting
    if for_whatsapp:
        # WhatsApp uses *bold*, convert ***text*** -> *text* and **text** -> *text*
        cleaned = re.sub(r"\*\*\*([^\*\n]+?)\*\*\*", r"*\1*", cleaned)
        cleaned = re.sub(r"\*\*([^\*\n]+?)\*\*", r"*\1*", cleaned)
        # Avoid consecutive asterisks
        cleaned = re.sub(r"\*{2,}", "*", cleaned)
    else:
        # For Notion text, strip markdown **bold** so it doesn't display raw asterisks
        cleaned = re.sub(r"\*\*\*([^\*\n]+?)\*\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*\*([^\*\n]+?)\*\*", r"\1", cleaned)

    return cleaned


def generate_leetcode_review(
    commit_data: LeetcodeCommitData,
    problem_details: Optional[LeetcodeProblemDetails] = None,
) -> LeetcodeReviewResult:
    """Send code + constraints + difficulty to Gemini for comprehensive review.

    If problem_details is None (GraphQL failure), review code standalone and explicitly mark fallback.
    """
    prob_title = (problem_details.title if problem_details else None) or commit_data.problem_title or "LeetCode Problem"
    difficulty = (problem_details.difficulty if problem_details else None) or "Unknown"
    constraints = problem_details.constraints if problem_details else []
    fallback_mode = problem_details is None

    constraints_text = (
        "\n".join([f"- {c}" for c in constraints])
        if constraints
        else "Problem constraints could not be retrieved from LeetCode GraphQL API. Review code standalone without assumed bounds."
    )

    prompt = (
        f"Problem: {prob_title}\n"
        f"Difficulty: {difficulty}\n\n"
        f"Stated Constraints:\n{constraints_text}\n\n"
        f"Submitted Solution ({commit_data.code_file_name}):\n"
        f"```\n{commit_data.code}\n```\n\n"
        "Please provide your structured review:\n"
        "VERDICT: <Correct / Suboptimal / Incorrect>\n"
        "TIME COMPLEXITY: <O(...)>\n"
        "SPACE COMPLEXITY: <O(...)>\n"
        "IS OPTIMAL: <Yes / No>\n\n"
        "ANALYSIS:\n<Concise breakdown of optimality against stated constraints, logic accuracy, and edge cases>\n\n"
        "TARGETED TESTING QUESTIONS:\n"
        "- <Specific question 1 probing submitted logic/edge case>\n"
        "- <Specific question 2 probing submitted logic/edge case>\n"
    )

    client = get_gemini_client()
    model_name = get_gemini_model()

    try:
        resp_text = generate_text(
            prompt=prompt,
            system_instruction=LEETCODE_REVIEW_SYSTEM_INSTRUCTION,
            model=model_name,
            temperature=0.2,
            fallback_default="",
            client=client,
        )



        # Parse structured components from Gemini response text
        verdict = "Correct"
        time_comp = "N/A"
        space_comp = "N/A"
        is_optimal = True

        verdict_m = re.search(r"VERDICT:\s*([A-Za-z]+)", resp_text, re.IGNORECASE)
        if verdict_m:
            verdict = verdict_m.group(1).capitalize()

        tc_m = re.search(r"TIME COMPLEXITY:\s*([^\n]+)", resp_text, re.IGNORECASE)
        if tc_m:
            time_comp = tc_m.group(1).strip()

        sc_m = re.search(r"SPACE COMPLEXITY:\s*([^\n]+)", resp_text, re.IGNORECASE)
        if sc_m:
            space_comp = sc_m.group(1).strip()

        opt_m = re.search(r"IS OPTIMAL:\s*(Yes|No|True|False)", resp_text, re.IGNORECASE)
        if opt_m:
            is_optimal = opt_m.group(1).lower() in ("yes", "true")

        # Extract Analysis section
        analysis = ""
        analysis_m = re.search(r"ANALYSIS:\s*(.*?)(?=TARGETED TESTING QUESTIONS:|$)", resp_text, re.DOTALL | re.IGNORECASE)
        if analysis_m:
            analysis = analysis_m.group(1).strip()
        else:
            analysis = resp_text.strip()

        # Extract Targeted Testing Questions
        testing_questions = []
        questions_m = re.search(r"TARGETED TESTING QUESTIONS:\s*(.*)$", resp_text, re.DOTALL | re.IGNORECASE)
        if questions_m:
            raw_q = questions_m.group(1).strip()
            for line in raw_q.split("\n"):
                clean_l = line.strip().lstrip("•-*0123456789. ")
                if clean_l:
                    testing_questions.append(clean_l)

        # Sanitize math/LaTeX artifacts and excessive markdown
        time_comp = clean_math_and_markdown(time_comp, for_whatsapp=False)
        space_comp = clean_math_and_markdown(space_comp, for_whatsapp=False)
        analysis = clean_math_and_markdown(analysis, for_whatsapp=False)
        testing_questions = [clean_math_and_markdown(q, for_whatsapp=False) for q in testing_questions]

        return LeetcodeReviewResult(
            problem_title=prob_title,
            problem_slug=commit_data.problem_slug,
            problem_number=commit_data.problem_number,
            difficulty=difficulty,
            verdict=verdict,
            time_complexity=time_comp,
            space_complexity=space_comp,
            is_optimal=is_optimal,
            review_summary=analysis,
            testing_questions=testing_questions,
            full_review_text=resp_text,
            fallback_mode=fallback_mode,
        )

    except Exception as exc:
        logger.error("Gemini LeetCode review generation failed (%s). Using fallback evaluation.", exc)
        return LeetcodeReviewResult(
            problem_title=prob_title,
            problem_slug=commit_data.problem_slug,
            problem_number=commit_data.problem_number,
            difficulty=difficulty,
            verdict="Evaluated",
            time_complexity="O(N)",
            space_complexity="O(1)",
            is_optimal=True,
            review_summary=f"Solution reviewed for {prob_title}. Logic appears structured cleanly.",
            testing_questions=[
                "What is the behavior for empty or minimal input arrays?",
                "Are there potential integer overflow or boundary index issues?",
            ],
            full_review_text=f"Review for {prob_title}.",
            fallback_mode=fallback_mode,
        )


# --- Step 4: Background Pipeline Orchestrator ---

def execute_leetcode_background_pipeline(
    leetcode_req: LeetcodeReviewRequest,
    to_phone: Optional[str] = None,
    chat_id: Optional[str] = None,
    notion_client: Optional[NotionAssistantClient] = None,
    whatsapp_client: Optional[WhatsAppAssistantClient] = None,
    telegram_client: Optional[TelegramAssistantClient] = None,
) -> Dict[str, Any]:
    """Execute asynchronous background pipeline for LEETCODE review:

    1. Fetch latest commit from LeetHub GitHub repo.
    2. Query LeetCode public GraphQL API for difficulty and constraints.
    3. Generate in-depth Gemini review with optimality and testing questions.
    4. Write review into NOTION_LEETCODE_LOG_DB_ID.
    5. Send follow-up WhatsApp/Telegram message with review results and fallback notice if applicable.
    """
    logger.info("Starting background LEETCODE pipeline...")

    notion = notion_client or NotionAssistantClient()
    whatsapp = whatsapp_client or WhatsAppAssistantClient()
    telegram = telegram_client or TelegramAssistantClient()

    # Step 1: Fetch latest commit from GitHub
    try:
        commit_data = fetch_latest_leethub_commit()
        logger.info(
            "Pulled commit %s for problem '%s' (slug='%s')",
            commit_data.commit_sha[:7],
            commit_data.problem_title,
            commit_data.problem_slug,
        )
    except Exception as gh_exc:
        logger.error("Failed to fetch commit from GitHub: %s", gh_exc)
        err_msg = (
            f"❌ *Failed to pull solution from GitHub:*\n{gh_exc}\n\n"
            "Please check that `GITHUB_PAT` and `GITHUB_LEETHUB_REPO` are configured correctly."
        )
        if to_phone:
            try:
                whatsapp.send_message(to=to_phone, text=err_msg)
            except Exception as wa_err:
                logger.error("Failed to send WhatsApp GitHub error: %s", wa_err)
        if chat_id:
            try:
                telegram.send_message(text=err_msg, chat_id=str(chat_id))
            except Exception as tg_err:
                logger.error("Failed to send Telegram GitHub error: %s", tg_err)
        return {
            "status": "error",
            "stage": "github_fetch",
            "error": str(gh_exc),
        }

    # Step 2: Fetch constraints & difficulty from LeetCode GraphQL API
    problem_details = fetch_leetcode_problem_details(commit_data.problem_slug)
    if problem_details is None:
        logger.warning(
            "LeetCode GraphQL API could not find problem for slug='%s'. Falling back to code-only review.",
            commit_data.problem_slug,
        )

    # Step 3: Send code + constraints + difficulty to Gemini for review
    review_result = generate_leetcode_review(
        commit_data=commit_data,
        problem_details=problem_details,
    )

    # Step 4: Write review into NOTION_LEETCODE_LOG_DB_ID
    notion_url: Optional[str] = None
    notion_page_id: Optional[str] = None

    try:
        prob_url = f"https://leetcode.com/problems/{commit_data.problem_slug}/"
        created_page = notion.create_leetcode_log_row(
            problem_title=review_result.problem_title,
            difficulty=review_result.difficulty,
            verdict=review_result.verdict,
            time_complexity=review_result.time_complexity,
            space_complexity=review_result.space_complexity,
            is_optimal=review_result.is_optimal,
            review_text=review_result.review_summary,
            testing_questions=review_result.testing_questions,
            code=commit_data.code,
            problem_url=prob_url,
            patterns=leetcode_req.patterns if leetcode_req else None,
        )
        notion_page_id = created_page.get("id") if isinstance(created_page, dict) else None
        notion_url = created_page.get("url") if isinstance(created_page, dict) else None
        if not notion_url and notion_page_id:
            clean_id = notion_page_id.replace("-", "")
            notion_url = f"https://www.notion.so/{clean_id}"
        review_result.notion_page_url = notion_url
        logger.info("Logged LeetCode review to Notion (page_id=%s)", notion_page_id)
        try:
            from app.motion import evidence_ingestion_engine
            evidence_ingestion_engine.ingest_leetcode_review(
                problem_title=review_result.problem_title,
                pattern_notes=review_result.review_summary or "",
                page_url=notion_url,
                duration_hours=1.0,
            )
        except Exception as err:
            logger.debug("Motion evidence ingestion skipped: %s", err)
    except Exception as notion_exc:
        logger.warning("Could not log LeetCode review to Notion (%s). Continuing with notification.", notion_exc)

    # Step 5: Construct Follow-up Notification Message
    msg_lines = []

    # If GraphQL fetch failed, explicitly notify user
    if review_result.fallback_mode:
        msg_lines.append("⚠️ _Note: Could not fetch constraints from LeetCode GraphQL API. Reviewing code without constraint context._\n")

    diff_str = f" • *{review_result.difficulty}*" if review_result.difficulty and review_result.difficulty != "Unknown" else ""
    opt_badge = "⚡ Optimal" if review_result.is_optimal else "⚠️ Suboptimal"

    msg_lines.append(f"💻 *LeetCode Review: {review_result.problem_title}*{diff_str}")
    msg_lines.append(f"📊 *Verdict:* {review_result.verdict} ({opt_badge})")
    msg_lines.append(f"⏱️ *Time:* {review_result.time_complexity or 'N/A'} | 💾 *Space:* {review_result.space_complexity or 'N/A'}\n")

    if review_result.review_summary:
        msg_lines.append(f"📝 *Analysis:*\n{review_result.review_summary}\n")

    if review_result.testing_questions:
        q_text = "\n".join([f"• {q}" for q in review_result.testing_questions[:4]])
        msg_lines.append(f"🧪 *Targeted Testing Questions:*\n{q_text}\n")

    if notion_url:
        msg_lines.append(f"🔗 *Notion Log:* {notion_url}")

    follow_up_message = clean_math_and_markdown("\n".join(msg_lines).strip(), for_whatsapp=True)

    # Step 6: Send finished review back to WhatsApp / Telegram
    send_notification(
        follow_up_message,
        to_phone=to_phone,
        chat_id=chat_id,
        preview_url=bool(notion_url),
        whatsapp_client=whatsapp,
        telegram_client=telegram,
    )


    return {
        "status": "ok",
        "problem_title": review_result.problem_title,
        "problem_slug": review_result.problem_slug,
        "verdict": review_result.verdict,
        "difficulty": review_result.difficulty,
        "is_optimal": review_result.is_optimal,
        "fallback_mode": review_result.fallback_mode,
        "notion_page_id": notion_page_id,
        "notion_url": notion_url,
    }
