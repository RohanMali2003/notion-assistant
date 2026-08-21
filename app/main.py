import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import BackgroundTasks, Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from app.config import settings
from app.graph_memory_service import graph_memory
from app.learning_service import execute_learning_background_pipeline
from app.leetcode_service import execute_leetcode_background_pipeline
from app.media_service import execute_media_pipeline
from app.memory import conversation_memory
from app.notion_client import NotionAssistantClient
from app.search_service import execute_second_brain_search_pipeline
from app.task_action_service import execute_batch_task_action, execute_task_action
from app.weekly_digest_service import execute_weekly_digest_pipeline
from app.workspace_service import append_blocks_to_document
from app.url_digest_service import (
    execute_url_digest_background_pipeline,
    extract_urls,
    is_url_dominant_message,
)
from app.schemas import (
    BatchTaskActionAnalysis,
    DocumentAppendAnalysis,
    LearningRequest,
    LeetcodeReviewRequest,
    MemoryGovernanceAnalysis,
    MindEntry,
    ModuleClassification,
    SearchQueryAnalysis,
    TaskActionAnalysis,
    TaskAnalysis,
    TelegramWebhookUpdate,
    WebhookResponse,
    WeeklyVelocityReport,
)
from app.telegram_client import TelegramAssistantClient
from app.whatsapp_client import WhatsAppAssistantClient
from app.motion import (
    AccountabilityMonitor,
    CausalAttributionTree,
    CognitiveEnergyReport,
    DailyCheckInPrompt,
    DecisionStatus,
    EvidenceIngestionEngine,
    EvidenceIngestionEvent,
    EvidenceItem,
    ExecutiveBriefing,
    HumanOverride,
    InitiativeMilestone,
    MotionConsultationResponse,
    MotionIdentity,
    MotionInitiative,
    MotionTrajectory,
    MotionWeeklyReview,
    MultiHopAttributionEngine,
    MultiWindowSynthesisEngine,
    PersonaType,
    ScenarioSimulationRequest,
    ScenarioSimulationResult,
    SocraticInquiryResult,
    StrategicDriftDetector,
    StrategicDriftReport,
    accountability_monitor,
    drift_detector,
    energy_monitor,
    evidence_ingestion_engine,
    executive_briefing_engine,
    milestone_engine,
    motion_mentorship_service,
    motion_review_service,
    motion_storage,
    multi_hop_attribution_engine,
    persona_router,
    scenario_engine,
    socratic_engine,
    strategic_model_service,
    synthesis_engine,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

app = FastAPI(
    title="Notion Assistant Webhook API",
    description="FastAPI service handling Notion and Telegram webhook events.",
    version="1.0.0",
)


def get_gemini_client():
    """Create and return a google-genai Client instance with configured API keys."""
    if genai is None or types is None:
        raise RuntimeError("google-genai library is not installed or available")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def get_gemini_model() -> str:
    """Return the Gemini model identifier to use (default: gemini-3.5-flash-lite)."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


# --- Stage 1: Module Classification ---

STAGE1_SYSTEM_INSTRUCTION = (
    "You are a lightweight intent routing classifier. Classify the user's message into exactly one MODULE:\n"
    "- TASKS: Creating single new tasks, querying pending or today's tasks, priority queries (e.g. 'high priority tasks'), and conversational follow-ups (e.g. 'others?', 'show more', 'next').\n"
    "- TASK_ACTION: Modifying a specific existing task ('mark Berkshire Dining done', 'done with GPAF form', 'postpone Berkshire Dining to next Tuesday', 'delete duplicate task').\n"
    "- BATCH_TASK_ACTION: Executing batch task commands across multiple tasks ('mark all UMass tasks as done', 'postpone all high priority items by 3 days', 'archive all completed tasks').\n"
    "- DOCUMENT_APPEND: Appending ideas, thoughts, bullets, or to-do notes directly into existing Notion document pages ('add Build AI voice agent to Ideas for projects', 'append to Year 1 Budget: Laptop insurance 200 USD', 'add note to Finances for Umass fall: Email bursar').\n"
    "- MEMORY_CONTROL: Memory governance, forgetting stale facts, updating long-term memory, or memory inspection commands ('forget grad school application notes', 'update memory: I am now a student at UMass', 'what do you remember about UMass?', 'forget X').\n"
    "- MIND: Substack drafts, journaling, brain dumps, rambling, daily reflections/logs, personal thoughts. NOTE: Short conversational follow-up questions (e.g. 'others?', 'what about tomorrow?') are NOT Mind entries; they belong to TASKS.\n"
    "- LEARNING: Explicit new study topic requests, learning roadmaps, syllabus inquiries (e.g. 'i want to learn about Gemini AI', 'explore gemma models', 'build study plan for transformers').\n"
    "- LEETCODE: LeetCode problem review requests, algorithm practice notes, problem solution tracking.\n"
    "- DIGEST: Requests for weekly velocity summaries, weekly review, or retrospective (e.g. 'how was my week?', 'weekly digest', 'weekly velocity', 'run weekly review').\n"
    "- MOTION: Strategic trajectory inquiries, mentorship questions, accountability check-ins, or explicit @motion requests ('@motion what is my biggest opportunity?', 'motion: evaluate my trajectory', 'strategic review of my projects', 'what are my biggest strategic risks?').\n"
    "- SEARCH: Inquiries querying past notes, search questions, folder exploration (e.g. 'what's in my notes?', 'what is in miscellaneous?', 'show my notes folder'), document inspection (e.g. 'what's in that year one budget?', 'tell me what's in finances for umass fall', 'did I write about finances for umass?'), archive suggestions (e.g. 'archive year one budget', 'send to archive'), or requests for information from the user's second brain.\n"
    "Pass the raw user message into the raw_text field."
)


def classify_module_stage1(text: str, context: Optional[str] = None) -> ModuleClassification:
    """Stage 1: Classify user message into target functional module using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        prompt_content = f"Recent conversation context:\n{context}\n\nUser message: {text}" if context else text
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=STAGE1_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ModuleClassification,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, ModuleClassification):
                result = response.parsed
            else:
                result = ModuleClassification.model_validate(response.parsed)
            if not result.raw_text:
                result.raw_text = text
            return result
        elif response.text:
            result = ModuleClassification.model_validate_json(response.text)
            if not result.raw_text:
                result.raw_text = text
            return result
        else:
            raise ValueError("Empty response from Gemini Stage 1 classification")
    except Exception as exc:
        logger.warning("Stage 1 classification failed (%s). Falling back to module=TASKS.", exc)
        return ModuleClassification(module="TASKS", raw_text=text)


# --- Stage 2: Module-Specific Parsers ---

STAGE2_MEMORY_CONTROL_INSTRUCTION = (
    "Extract memory governance details from the user message:\n"
    "- command: FORGET (if user asks to forget/delete memory like 'forget grad school notes', 'remove X from memory'), "
    "UPDATE_STATUS (if user updates current status like 'update memory: I am enrolled at UMass', 'I started my MSCS'), "
    "INSPECT_MEMORY (if user asks what Ocean remembers like 'what do you remember about UMass?', 'show memory for X').\n"
    "- target_entity: The core entity or topic name.\n"
    "- new_state_summary: Summary of updated memory thesis or state if updating status."
)


def parse_memory_control_stage2(text: str, context: Optional[str] = None) -> MemoryGovernanceAnalysis:
    """Stage 2: Parse MEMORY_CONTROL module intent using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        prompt_content = f"Recent conversation context:\n{context}\n\nUser message: {text}" if context else text
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_MEMORY_CONTROL_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=MemoryGovernanceAnalysis,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, MemoryGovernanceAnalysis):
                return response.parsed
            return MemoryGovernanceAnalysis.model_validate(response.parsed)
        elif response.text:
            return MemoryGovernanceAnalysis.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 MEMORY_CONTROL parser")
    except Exception as exc:
        logger.warning("Stage 2 MEMORY_CONTROL parsing failed (%s). Falling back to default MemoryGovernanceAnalysis.", exc)
        return MemoryGovernanceAnalysis(command="FORGET", target_entity=text)


STAGE2_BATCH_TASK_ACTION_INSTRUCTION = (
    "Extract batch task action details from the user message:\n"
    "- action: MARK_DONE, MARK_IN_PROGRESS, UPDATE_DUE_DATE, or DELETE_TASK.\n"
    "- tag_filter: Optional tag filter (e.g. UMass Admin, Leetcode, Finances).\n"
    "- priority_filter: Optional priority filter (High, Medium, Low).\n"
    "- target_query: Optional search keyword matching multiple tasks.\n"
    "- new_due_date_iso: Resolved YYYY-MM-DD due date if updating due date.\n"
    "- new_status_name: Done, In progress, or Not started."
)


def parse_batch_task_action_stage2(text: str, context: Optional[str] = None) -> BatchTaskActionAnalysis:
    """Stage 2: Parse BATCH_TASK_ACTION module intent using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        prompt_content = f"Recent conversation context:\n{context}\n\nUser message: {text}" if context else text
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_BATCH_TASK_ACTION_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=BatchTaskActionAnalysis,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, BatchTaskActionAnalysis):
                return response.parsed
            return BatchTaskActionAnalysis.model_validate(response.parsed)
        elif response.text:
            return BatchTaskActionAnalysis.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 BATCH_TASK_ACTION parser")
    except Exception as exc:
        logger.warning("Stage 2 BATCH_TASK_ACTION parsing failed (%s). Falling back to default BatchTaskActionAnalysis.", exc)
        return BatchTaskActionAnalysis(action="MARK_DONE", target_query=text)


# --- Stage 2: Module-Specific Parsers ---

STAGE2_TASK_ACTION_INSTRUCTION = (
    "Extract task modification details from the user message:\n"
    "- action: MARK_DONE (e.g. 'done with X', 'mark X done', 'complete X'), MARK_IN_PROGRESS (e.g. 'set X to in progress', 'working on X'), UPDATE_DUE_DATE (e.g. 'postpone X to tuesday', 'move X to August 30', 'push X to tomorrow'), DELETE_TASK (e.g. 'delete X', 'remove X from tasks', 'archive X task').\n"
    "- task_target_title: Target task title or keywords to match.\n"
    "- new_due_date_iso: Resolved due date in YYYY-MM-DD format if updating due date.\n"
    "- new_status_name: Done, In progress, or Not started.\n"
    "- ordinal_index: 1 for first task, 2 for second task, etc. if user referred to position."
)


def parse_task_action_stage2(text: str, context: Optional[str] = None) -> TaskActionAnalysis:
    """Stage 2: Parse TASK_ACTION module intent using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        prompt_content = f"Recent conversation context:\n{context}\n\nUser message: {text}" if context else text
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_TASK_ACTION_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=TaskActionAnalysis,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, TaskActionAnalysis):
                return response.parsed
            return TaskActionAnalysis.model_validate(response.parsed)
        elif response.text:
            return TaskActionAnalysis.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 TASK_ACTION parser")
    except Exception as exc:
        logger.warning("Stage 2 TASK_ACTION parsing failed (%s). Falling back to default TaskActionAnalysis.", exc)
        return TaskActionAnalysis(action="MARK_DONE", task_target_title=text)


STAGE2_DOCUMENT_APPEND_INSTRUCTION = (
    "Extract document append details from the user message:\n"
    "- target_document_title: Exact title of target document or container page (e.g. 'Ideas for projects', 'Year 1 Budget', 'Finances for Umass fall', 'Notes').\n"
    "- content_to_append: The bullet point, idea, thought, or item text to append into the document.\n"
    "- block_type: 'bulleted_list_item' (default), 'to_do', 'paragraph', or 'callout'."
)


def parse_document_append_stage2(text: str, context: Optional[str] = None) -> DocumentAppendAnalysis:
    """Stage 2: Parse DOCUMENT_APPEND module details using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        prompt_content = f"Recent conversation context:\n{context}\n\nUser message: {text}" if context else text
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_DOCUMENT_APPEND_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=DocumentAppendAnalysis,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, DocumentAppendAnalysis):
                return response.parsed
            return DocumentAppendAnalysis.model_validate(response.parsed)
        elif response.text:
            return DocumentAppendAnalysis.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 DOCUMENT_APPEND parser")
    except Exception as exc:
        logger.warning("Stage 2 DOCUMENT_APPEND parsing failed (%s). Falling back to default DocumentAppendAnalysis.", exc)
        return DocumentAppendAnalysis(target_document_title="Notes", content_to_append=text)


# --- Stage 2: Module-Specific Parsers ---

STAGE2_TASKS_INSTRUCTION = (
    "Extract task management details from the user message:\n"
    "- CREATE_TASK: Adding a new task, with optional priority (High/Medium/Low), tag, due_date (YYYY-MM-DD), description.\n"
    "- UPDATE_TASK: Updating status of an existing task (In progress/Done/Not started) or due date.\n"
    "- QUERY_TODAY: Querying tasks due today.\n"
    "- QUERY_PENDING: Querying pending or upcoming tasks. If user asks for high/medium/low priority (e.g. 'high priority tasks', 'urgent tasks'), set priority_filter='High' (or Medium/Low). If user asks 'others?', 'more', 'next', 'what else?', set is_followup=True and intent='QUERY_PENDING'.\n"
    "- DAILY_LOG: Recording daily notes or log entry."
)


def parse_tasks_stage2(text: str, context: Optional[str] = None) -> TaskAnalysis:
    """Stage 2: Parse TASKS module intent and task structure using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        prompt_content = f"Recent conversation context:\n{context}\n\nUser message: {text}" if context else text
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_TASKS_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=TaskAnalysis,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, TaskAnalysis):
                return response.parsed
            return TaskAnalysis.model_validate(response.parsed)
        elif response.text:
            return TaskAnalysis.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 TASKS parser")
    except Exception as exc:
        logger.warning("Stage 2 TASKS parsing failed (%s). Falling back to intent=DAILY_LOG.", exc)
        return TaskAnalysis(
            intent="DAILY_LOG",
            title=text[:50] if text else "Daily Log",
            log_content=text,
        )


STAGE2_MIND_INSTRUCTION = (
    "Extract mind/thought entry details from the user message:\n"
    "- entry_type: DRAFT_SUBSTACK (article drafts, newsletter ideas, essays), RAMBLING (stream of consciousness, quick thoughts, brain dumps), or DAILY_LOG (daily reflections, logs, journaling).\n"
    "- title: A concise, descriptive title or headline for the database entry.\n"
    "- core_thesis: Exactly one sentence summarizing the core thesis, key insight, or main premise.\n"
    "- content: The complete text, thoughts, or drafted body.\n"
    "- summary: Optional brief summary or key takeaways.\n"
    "- tags: Relevant topics or tags."
)


def parse_mind_stage2(text: str) -> MindEntry:
    """Stage 2: Parse MIND module entry (substack draft, rambling, daily log) using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_MIND_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=MindEntry,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, MindEntry):
                result = response.parsed
            else:
                result = MindEntry.model_validate(response.parsed)
            if not result.content:
                result.content = text
            if not result.title:
                result.title = text[:50] if text else "Untitled Entry"
            if not result.core_thesis:
                first_sent = (result.content or text).strip().split(".")[0].strip()
                result.core_thesis = (first_sent + ".") if first_sent else "Daily reflection."
            return result
        elif response.text:
            result = MindEntry.model_validate_json(response.text)
            if not result.content:
                result.content = text
            if not result.title:
                result.title = text[:50] if text else "Untitled Entry"
            if not result.core_thesis:
                first_sent = (result.content or text).strip().split(".")[0].strip()
                result.core_thesis = (first_sent + ".") if first_sent else "Daily reflection."
            return result
        else:
            raise ValueError("Empty response from Gemini Stage 2 MIND parser")
    except Exception as exc:
        logger.warning("Stage 2 MIND parsing failed (%s). Falling back to default MindEntry.", exc)
        first_sent = text.strip().split(".")[0].strip() if text else "Daily reflection"
        return MindEntry(
            entry_type="DAILY_LOG",
            title=text[:50] if text else "Daily Reflection",
            core_thesis=(first_sent + ".") if first_sent and not first_sent.endswith(".") else first_sent,
            content=text,
        )


STAGE2_LEARNING_INSTRUCTION = (
    "Extract learning/study request details from the user message:\n"
    "- topic: The subject or topic to study (e.g. Distributed Systems, Rust Borrow Checker).\n"
    "- category: Broader domain (e.g. Computer Science, Math, Systems Engineering).\n"
    "- goal: Specific learning objective, subtopics, or questions.\n"
    "- proficiency_level: Beginner, Intermediate, or Advanced.\n"
    "- resources_requested: Any specific resources, books, papers, tutorials, or roadmaps asked for."
)


def parse_learning_stage2(text: str) -> LearningRequest:
    """Stage 2: Parse LEARNING module request (new study topic) using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_LEARNING_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=LearningRequest,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, LearningRequest):
                return response.parsed
            return LearningRequest.model_validate(response.parsed)
        elif response.text:
            return LearningRequest.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 LEARNING parser")
    except Exception as exc:
        logger.warning("Stage 2 LEARNING parsing failed (%s). Falling back to default LearningRequest.", exc)
        return LearningRequest(
            topic=text[:50] if text else "New Study Topic",
            goal=text,
        )


STAGE2_LEETCODE_INSTRUCTION = (
    "Extract LeetCode review request details from the user message:\n"
    "- problem_name: Problem title or name (e.g. Two Sum, Trapping Rain Water).\n"
    "- problem_number: Problem number integer if mentioned.\n"
    "- difficulty: Easy, Medium, or Hard.\n"
    "- patterns: Algorithmic techniques used (e.g. Two Pointers, Dynamic Programming, Monotonic Stack).\n"
    "- review_notes: Key insights, complexities, pitfalls, or review notes.\n"
    "- status: Solved, Review Needed, Failed, or Mastered."
)


def parse_leetcode_stage2(text: str) -> LeetcodeReviewRequest:
    """Stage 2: Parse LEETCODE module review request using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_LEETCODE_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=LeetcodeReviewRequest,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, LeetcodeReviewRequest):
                return response.parsed
            return LeetcodeReviewRequest.model_validate(response.parsed)
        elif response.text:
            return LeetcodeReviewRequest.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 LEETCODE parser")
    except Exception as exc:
        logger.warning("Stage 2 LEETCODE parsing failed (%s). Falling back to default LeetcodeReviewRequest.", exc)
        return LeetcodeReviewRequest(
            problem_name=text[:50] if text else "LeetCode Review",
            review_notes=text,
        )


STAGE2_SEARCH_INSTRUCTION = (
    "Extract search query parameters from the user's knowledge inquiry:\n"
    "- query: The clean core question or search query.\n"
    "- target_domain: Optional domain tag filter (AI Research, System Design, Distributed Systems, Leetcode, Finances, Schoolwork, etc.).\n"
    "- time_filter: Optional time filter (e.g. yesterday, past week, last month).\n"
    "- search_type: 'FOLDER_EXPLORE' (if asking what's in a folder/page like 'what\\'s in my notes?', 'what is in miscellaneous?'), "
    "'PAGE_INSPECT' (if asking to read or inspect content of a specific page/note like 'what\\'s in that year one budget?', 'tell me what\\'s in finances for umass fall'), "
    "'ARCHIVE_SUGGEST' (if asking to archive or send a page to archive like 'archive year one budget', 'send to archive'), "
    "'FIND_NOTES', 'LIST_SUBJECTS', 'LIST_TASKS', or 'QUESTION'.\n"
    "- container_name: Target folder/container name if exploring (e.g. 'Notes', 'Miscellaneous', 'YouTube', 'Archive').\n"
    "- page_name: Target document or note title if inspecting or archiving (e.g. 'year one budget', 'Finances for Umass fall')."
)


def parse_search_stage2(text: str, context: Optional[str] = None) -> SearchQueryAnalysis:
    """Stage 2: Parse SEARCH module parameters using gemini-3.5-flash-lite."""
    try:
        client = get_gemini_client()
        model_name = get_gemini_model()
        prompt_content = f"Recent conversation context:\n{context}\n\nUser query: {text}" if context else text
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=STAGE2_SEARCH_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=SearchQueryAnalysis,
            ),
        )
        if response.parsed:
            if isinstance(response.parsed, SearchQueryAnalysis):
                return response.parsed
            return SearchQueryAnalysis.model_validate(response.parsed)
        elif response.text:
            return SearchQueryAnalysis.model_validate_json(response.text)
        else:
            raise ValueError("Empty response from Gemini Stage 2 SEARCH parser")
    except Exception as exc:
        logger.warning("Stage 2 SEARCH parsing failed (%s). Falling back to default SearchQueryAnalysis.", exc)
        return SearchQueryAnalysis(query=text)


# --- Two-Stage Pipeline Orchestration ---

def analyze_user_text_two_stage(
    text: str,
    context: Optional[str] = None,
) -> Tuple[str, Union[TaskAnalysis, MindEntry, LearningRequest, LeetcodeReviewRequest, SearchQueryAnalysis, str]]:
    """Execute two-stage Gemini pipeline: Stage 1 classification -> Stage 2 module-specific parsing."""
    if context:
        stage1_res = classify_module_stage1(text, context=context)
    else:
        stage1_res = classify_module_stage1(text)

    module = stage1_res.module
    raw_text = stage1_res.raw_text or text

    # Anti-rambling guardrail: Short ambiguous queries (<= 4 words) ending with '?' or conversational follow-up words
    # must never be categorized as MIND entries unless explicit journaling keywords are present.
    text_lower = text.strip().lower()
    explicit_mind_keywords = ("substack", "rambling", "brain dump", "daily log", "journal", "feeling", "thought:", "reflection:")
    is_short_query = (
        len(text.strip().split()) <= 4
        and (
            text_lower.endswith("?")
            or text_lower in ("others", "others?", "more", "next", "what else", "what else?", "show more", "and?", "next page")
        )
    )
    if module == "MIND" and is_short_query and not any(kw in text_lower for kw in explicit_mind_keywords):
        logger.info("Anti-rambling guardrail redirected short query '%s' to TASKS", text)
        module = "TASKS"

    if module == "TASKS":
        parsed = parse_tasks_stage2(raw_text, context=context) if context else parse_tasks_stage2(raw_text)
    elif module == "TASK_ACTION":
        parsed = parse_task_action_stage2(raw_text, context=context) if context else parse_task_action_stage2(raw_text)
    elif module == "BATCH_TASK_ACTION":
        parsed = parse_batch_task_action_stage2(raw_text, context=context) if context else parse_batch_task_action_stage2(raw_text)
    elif module == "DOCUMENT_APPEND":
        parsed = parse_document_append_stage2(raw_text, context=context) if context else parse_document_append_stage2(raw_text)
    elif module == "MEMORY_CONTROL":
        parsed = parse_memory_control_stage2(raw_text, context=context) if context else parse_memory_control_stage2(raw_text)
    elif module == "MIND":
        parsed = parse_mind_stage2(raw_text)
    elif module == "LEARNING":
        parsed = parse_learning_stage2(raw_text)
    elif module == "LEETCODE":
        parsed = parse_leetcode_stage2(raw_text)
    elif module == "SEARCH":
        parsed = parse_search_stage2(raw_text, context=context) if context else parse_search_stage2(raw_text)
    elif module == "DIGEST":
        parsed = raw_text
    elif module == "MOTION":
        parsed = raw_text
    else:
        parsed = parse_tasks_stage2(raw_text, context=context) if context else parse_tasks_stage2(raw_text)

    return module, parsed


def analyze_user_text_with_gemini(text: str) -> TaskAnalysis:
    """Legacy helper: calls Stage 2 TASKS parsing directly for backward compatibility."""
    return parse_tasks_stage2(text)


@app.get("/health", response_model=WebhookResponse)
def health_check():
    """Health check endpoint for uptime pings and Render health checks."""
    return WebhookResponse(status="ok")


@app.get("/webhook")
def whatsapp_webhook_handshake(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    """WhatsApp Cloud API webhook handshake verification endpoint."""
    expected_verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN") or getattr(settings, "WHATSAPP_VERIFY_TOKEN", "")
    if hub_mode == "subscribe" and hub_verify_token and hub_verify_token == expected_verify_token:
        logger.info("WhatsApp webhook handshake verified successfully.")
        return PlainTextResponse(content=hub_challenge or "", status_code=200)

    logger.warning(
        "WhatsApp webhook handshake failed: mode=%s, token match=%s",
        hub_mode,
        bool(hub_verify_token and hub_verify_token == expected_verify_token),
    )
    raise HTTPException(status_code=403, detail="Verification token mismatch")


def _extract_whatsapp_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sender, event type (text, image), text body or media ID and caption from WhatsApp Cloud API payload."""
    try:
        entries = payload.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                messages = value.get("messages", [])
                for msg in messages:
                    sender = msg.get("from")
                    msg_type = msg.get("type", "text")
                    if msg_type == "text":
                        text = msg.get("text", {}).get("body", "")
                        return {"sender": str(sender) if sender else None, "type": "text", "text": text}
                    elif msg_type == "image":
                        image_info = msg.get("image", {})
                        media_id = image_info.get("id")
                        caption = image_info.get("caption", "")
                        mime_type = image_info.get("mime_type", "image/jpeg")
                        return {
                            "sender": str(sender) if sender else None,
                            "type": "image",
                            "media_id": media_id,
                            "caption": caption,
                            "mime_type": mime_type,
                            "text": caption,
                        }
                    elif "text" in msg:
                        text = msg.get("text", {}).get("body", "") or str(msg.get("text"))
                        return {"sender": str(sender) if sender else None, "type": "text", "text": text}
    except Exception as exc:
        logger.debug("Error extracting WhatsApp event: %s", exc)
    return {"sender": None, "type": None, "text": None}


def _extract_whatsapp_message(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract sender phone and text from WhatsApp Cloud API webhook payload."""
    ev = _extract_whatsapp_event(payload)
    return ev.get("sender"), ev.get("text")


async def handle_pending_menu_reply(
    sender_id: str,
    text: str,
    notion_client: NotionAssistantClient,
) -> Optional[str]:
    """Check if user text resolves a pending interactive disambiguation menu."""
    pending_menu = conversation_memory.get_pending_menu(sender_id)
    if not pending_menu:
        return None

    clean = text.strip().lower()
    idx = None
    if clean in ("1", "1st", "first", "option 1", "choice 1", "#1"):
        idx = 0
    elif clean in ("2", "2nd", "second", "option 2", "choice 2", "#2"):
        idx = 1
    elif clean in ("3", "3rd", "third", "option 3", "choice 3", "#3"):
        idx = 2

    candidates = pending_menu.get("candidates", [])
    if idx is not None and 0 <= idx < len(candidates):
        selected_item = candidates[idx]
        conversation_memory.clear_pending_menu(sender_id)

        module = pending_menu.get("module")
        if module == "TASK_ACTION":
            raw_action = pending_menu.get("action_analysis", {})
            action_analysis = TaskActionAnalysis.model_validate(raw_action)
            # Perform action on selected item directly
            page_id = selected_item.get("page_id")
            task_title = selected_item.get("title", "Untitled")
            task_url = selected_item.get("url", "#")
            action_type = action_analysis.action

            if action_type == TaskActionType.MARK_DONE:
                status_name = action_analysis.new_status_name or "Done"
                notion_client.client.pages.update(page_id=page_id, properties={"Status": {"status": {"name": status_name}}})
                return f"✅ Selected **#{idx+1}**: Marked **[{task_title}]({task_url})** as *{status_name}*!"
            elif action_type == TaskActionType.MARK_IN_PROGRESS:
                status_name = action_analysis.new_status_name or "In progress"
                notion_client.client.pages.update(page_id=page_id, properties={"Status": {"status": {"name": status_name}}})
                return f"🔄 Selected **#{idx+1}**: Set **[{task_title}]({task_url})** to *{status_name}*!"
            elif action_type == TaskActionType.DELETE_TASK:
                notion_client.client.pages.update(page_id=page_id, archived=True)
                return f"🗑️ Selected **#{idx+1}**: Archived **[{task_title}]({task_url})**!"

    return None


async def _handle_module_action(
    module: str,
    parsed_result: Any,
    text: str,
    notion_client: NotionAssistantClient,
    sender_id: Optional[str] = None,
) -> str:
    """Execute standard synchronous actions for TASKS, TASK_ACTION, BATCH_TASK_ACTION, DOCUMENT_APPEND, MEMORY_CONTROL, and MIND modules."""
    if module == "MEMORY_CONTROL":
        mem_analysis: MemoryGovernanceAnalysis = parsed_result
        cmd = mem_analysis.command
        target = mem_analysis.target_entity or text

        if cmd == "FORGET":
            count, forgotten = graph_memory.forget_entity(target)
            if count > 0:
                return f"🧠 *Memory Updated!* Soft-deleted {count} graph entity node(s) matching *'{target}'*:\n" + "\n".join(f"• ~{n}~" for n in forgotten)
            return f"🧠 *Memory Inspection:* No active knowledge graph nodes found matching *'{target}'*."

        elif cmd == "UPDATE_STATUS":
            summary = mem_analysis.new_state_summary or f"Updated status to {target}"
            count = graph_memory.supersede_nodes(target, target, reason=summary)
            node = graph_memory.add_node(target, summary=summary, status="ACTIVE")
            return f"🧠 *Knowledge Graph Updated!*\n\n📌 **{node['name']}**: {summary}\n*(Superseded {count} older related nodes.)*"

        elif cmd in ("INSPECT_MEMORY", "SYNC_GRAPH"):
            if "sync" in text.lower() or cmd == "SYNC_GRAPH":
                sync_res = graph_memory.sync_graph_to_notion(notion_client)
                return sync_res.get("reply_text", "Knowledge Graph synced to Notion!")
            nodes = graph_memory.query_active_knowledge(query=target, limit=10)
            if nodes:
                lines = [f"• *{n['name']}* [{n['entity_type']}]: {n.get('summary', 'Active')}" for n in nodes]
                return f"🧠 *Active Knowledge Graph ({len(nodes)} items):*\n" + "\n".join(lines)
            return f"🧠 *Knowledge Graph:* No active nodes found matching *'{target}'*."

        return "🧠 Memory governance action processed."

    elif module == "BATCH_TASK_ACTION":
        batch_analysis: BatchTaskActionAnalysis = parsed_result
        batch_res = await run_in_threadpool(
            execute_batch_task_action,
            batch_analysis=batch_analysis,
            notion_client=notion_client,
        )
        return batch_res.get("reply_text", "Batch action complete!")

    elif module == "TASK_ACTION":
        action_analysis: TaskActionAnalysis = parsed_result
        action_res = await run_in_threadpool(
            execute_task_action,
            action_analysis=action_analysis,
            notion_client=notion_client,
            user_id=sender_id,
        )
        return action_res.get("reply_text", "Done!")

    elif module == "DOCUMENT_APPEND":
        append_analysis: DocumentAppendAnalysis = parsed_result
        append_res = await run_in_threadpool(
            append_blocks_to_document,
            target_title_or_id=append_analysis.target_document_title,
            content=append_analysis.content_to_append,
            block_type=append_analysis.block_type,
            notion_client=notion_client,
        )
        return append_res.get("reply_text", "Done!")

    elif module == "TASKS":
        parsed_task: TaskAnalysis = parsed_result
        intent = parsed_task.intent

        if intent == "CREATE_TASK":
            created_page = await run_in_threadpool(
                notion_client.create_task,
                title=parsed_task.title or text,
                priority=parsed_task.priority,
                tag=parsed_task.tag,
                due_date=parsed_task.due_date,
                description=parsed_task.description,
            )
            page_url = created_page.get("url") if isinstance(created_page, dict) else None
            if not page_url and isinstance(created_page, dict) and created_page.get("id"):
                clean_id = created_page["id"].replace("-", "")
                page_url = f"https://www.notion.so/{clean_id}"

            reply_text = f"✅ Task created: *{parsed_task.title or text}*"
            if page_url:
                reply_text += f"\n🔗 {page_url}"
            if parsed_task.due_date:
                reply_text += f"\n📅 Due: {parsed_task.due_date}"
            if parsed_task.priority:
                reply_text += f"\n⚡ Priority: {parsed_task.priority}"
            if parsed_task.tag:
                reply_text += f"\n🏷 Tag: {parsed_task.tag}"

            if sender_id:
                conversation_memory.update_query_state(sender_id, last_module="TASKS", last_intent="CREATE_TASK")
            return reply_text

        elif intent == "UPDATE_TASK":
            target_status = parsed_task.target_status or "In progress"
            title_query = parsed_task.title or text
            success, matched_title, updated_page = await run_in_threadpool(
                notion_client.update_task_status,
                title_query=title_query,
                status_name=target_status,
                new_due_date=parsed_task.new_due_date or parsed_task.due_date,
            )
            if success:
                page_url = updated_page.get("url") if isinstance(updated_page, dict) else None
                if not page_url and isinstance(updated_page, dict) and updated_page.get("id"):
                    clean_id = updated_page["id"].replace("-", "")
                    page_url = f"https://www.notion.so/{clean_id}"

                reply_text = f"✅ Updated *{matched_title}*\n🔄 Status: *{target_status}*"
                if page_url:
                    reply_text += f"\n🔗 {page_url}"
                if parsed_task.new_due_date or parsed_task.due_date:
                    reply_text += f"\n📅 Due: {parsed_task.new_due_date or parsed_task.due_date}"
            else:
                reply_text = f"⚠️ Could not find an active task matching: *{title_query}*"

            if sender_id:
                conversation_memory.update_query_state(sender_id, last_module="TASKS", last_intent="UPDATE_TASK")
            return reply_text

        elif intent == "QUERY_TODAY":
            last_state = conversation_memory.get_last_query_state(sender_id) if sender_id else {}
            p_filter = parsed_task.priority_filter or (last_state.get("priority_filter") if parsed_task.is_followup else None)
            t_filter = parsed_task.tag_filter or (last_state.get("tag_filter") if parsed_task.is_followup else None)

            today_items = await run_in_threadpool(
                notion_client.get_today_tasks,
                priority=p_filter,
                tag=t_filter,
            )

            if sender_id:
                conversation_memory.update_query_state(
                    sender_id,
                    last_module="TASKS",
                    last_intent="QUERY_TODAY",
                    priority_filter=p_filter,
                    tag_filter=t_filter,
                    last_offset=0,
                )

            if today_items:
                if sender_id:
                    today_dicts = [item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else getattr(item, "__dict__", {})) for item in today_items]
                    conversation_memory.set_last_query_results(sender_id, today_dicts)

                header = "📅 *Today's Tasks"
                if p_filter:
                    header += f" ({p_filter} Priority)"
                header += f" ({len(today_items)}):*"

                tasks_lines = []
                for item in today_items:
                    item_title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else str(item))
                    item_due = getattr(item, "due_date", None) or (item.get("due_date") if isinstance(item, dict) else None)
                    item_url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
                    if not item_url:
                        p_id = getattr(item, "page_id", None) or (item.get("page_id") if isinstance(item, dict) else None)
                        if p_id:
                            item_url = f"https://www.notion.so/{str(p_id).replace('-', '')}"

                    line = f"• *{item_title}*"
                    if item_due:
                        line += f" (Due: {item_due})"
                    if item_url:
                        line += f"\n  🔗 {item_url}"
                    tasks_lines.append(line)
                tasks_str = "\n".join(tasks_lines)
                return f"{header}\n{tasks_str}"
            else:
                if p_filter:
                    return f"🎉 No {p_filter} priority tasks due today!"
                return "🎉 No tasks due today!"

        elif intent == "QUERY_PENDING":
            last_state = conversation_memory.get_last_query_state(sender_id) if sender_id else {}
            text_clean = text.strip().lower()
            is_followup_word = text_clean in ("others", "others?", "more", "next", "what else", "what else?", "show more", "next page")

            if parsed_task.is_followup or is_followup_word:
                p_filter = parsed_task.priority_filter or last_state.get("priority_filter")
                t_filter = parsed_task.tag_filter or last_state.get("tag_filter")
                offset = last_state.get("last_offset", 0) + 5
            else:
                p_filter = parsed_task.priority_filter
                t_filter = parsed_task.tag_filter
                offset = parsed_task.offset or 0

            if p_filter or t_filter or offset > 0:
                pending_items = await run_in_threadpool(
                    notion_client.get_pending,
                    limit=5,
                    offset=offset,
                    priority=p_filter,
                    tag=t_filter,
                )
            else:
                pending_items = await run_in_threadpool(
                    notion_client.get_pending,
                    limit=5,
                )

            if sender_id:
                conversation_memory.update_query_state(
                    sender_id,
                    last_module="TASKS",
                    last_intent="QUERY_PENDING",
                    priority_filter=p_filter,
                    tag_filter=t_filter,
                    last_offset=offset,
                )

            if pending_items:
                if sender_id:
                    pending_dicts = [item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else getattr(item, "__dict__", {})) for item in pending_items]
                    conversation_memory.set_last_query_results(sender_id, pending_dicts)

                header = "📋 *Pending Tasks"
                if p_filter:
                    header += f" ({p_filter} Priority)"
                if offset > 0:
                    header += f" (Items {offset + 1}-{offset + len(pending_items)}):*"
                else:
                    header += f" ({len(pending_items)}):*"

                tasks_lines = []
                for item in pending_items:
                    item_title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else str(item))
                    item_due = getattr(item, "due_date", None) or (item.get("due_date") if isinstance(item, dict) else None)
                    item_url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
                    if not item_url:
                        p_id = getattr(item, "page_id", None) or (item.get("page_id") if isinstance(item, dict) else None)
                        if p_id:
                            item_url = f"https://www.notion.so/{str(p_id).replace('-', '')}"

                    line = f"• *{item_title}*"
                    if item_due:
                        line += f" (Due: {item_due})"
                    if item_url:
                        line += f"\n  🔗 {item_url}"
                    tasks_lines.append(line)
                tasks_str = "\n".join(tasks_lines)
                return f"{header}\n{tasks_str}"
            else:
                if offset > 0:
                    return f"🎉 No more {p_filter or ''} pending tasks found!".replace("  ", " ")
                elif p_filter:
                    return f"🎉 No {p_filter} priority pending tasks found!"
                else:
                    return "🎉 No pending tasks found!"

        else:  # DAILY_LOG or fallback
            log_content = parsed_task.log_content or text
            if sender_id:
                conversation_memory.update_query_state(sender_id, last_module="TASKS", last_intent="DAILY_LOG")
            return f"📝 *Daily Log Recorded:*\n{log_content}"

    elif module == "MIND":
        mind_entry: MindEntry = parsed_result
        entry_title = mind_entry.title or (text[:50] if text else "Untitled Entry")
        entry_content = mind_entry.content or text
        entry_sub_intent = mind_entry.sub_intent or "DAILY_LOG"

        created_mind = await run_in_threadpool(
            notion_client.create_mind_entry,
            entry_type=entry_sub_intent,
            title=entry_title,
            content=entry_content,
            core_thesis=mind_entry.core_thesis,
            tags=mind_entry.tags,
        )
        mind_url = created_mind.get("url") if isinstance(created_mind, dict) else None
        if not mind_url and isinstance(created_mind, dict) and created_mind.get("id"):
            clean_id = created_mind["id"].replace("-", "")
            mind_url = f"https://www.notion.so/{clean_id}"

        if entry_sub_intent == "DRAFT_SUBSTACK":
            reply_text = f"✍️ *Substack Draft Created (Idea):* *{entry_title}*"
        elif entry_sub_intent == "RAMBLING":
            reply_text = f"💭 *Rambling Recorded:* *{entry_title}*"
        else:
            reply_text = f"📝 *Daily Log Recorded:* *{entry_title}*"

        if mind_url:
            reply_text += f"\n🔗 {mind_url}"
        if mind_entry.core_thesis:
            reply_text += f"\n💡 *Core Thesis:* {mind_entry.core_thesis}"
        elif mind_entry.summary:
            reply_text += f"\n📌 *Summary:* {mind_entry.summary}"
        if mind_entry.tags:
            reply_text += f"\n🏷 Tags: {', '.join(mind_entry.tags)}"

        if sender_id:
            conversation_memory.update_query_state(sender_id, last_module="MIND", last_intent=entry_sub_intent)
        return reply_text

    elif module == "LEETCODE":
        lc_req: LeetcodeReviewRequest = parsed_result
        num_part = f"#{lc_req.problem_number} " if lc_req.problem_number else ""
        reply_text = f"💻 *LeetCode Review Logged:* *{num_part}{lc_req.problem_name or text}*"
        if lc_req.difficulty:
            reply_text += f"\n⚡ Difficulty: {lc_req.difficulty}"
        if lc_req.patterns:
            reply_text += f"\n🧩 Patterns: {', '.join(lc_req.patterns)}"
        if lc_req.status:
            reply_text += f"\n📊 Status: {lc_req.status}"
        if lc_req.review_notes:
            reply_text += f"\n📝 Notes: {lc_req.review_notes}"

    elif module == "SEARCH":
        search_analysis = parsed_result if isinstance(parsed_result, SearchQueryAnalysis) else SearchQueryAnalysis(query=text)
        search_res = await run_in_threadpool(
            execute_second_brain_search_pipeline,
            query=search_analysis.query,
            search_analysis=search_analysis,
            notion_client=notion_client,
        )
        if sender_id:
            conversation_memory.update_query_state(sender_id, last_module="SEARCH", last_intent=search_analysis.search_type)
        return search_res.get("reply_text", "")

    elif module == "DIGEST":
        digest_res = await run_in_threadpool(
            execute_weekly_digest_pipeline,
            notion_client=notion_client,
        )
        if sender_id:
            conversation_memory.update_query_state(sender_id, last_module="DIGEST", last_intent="GENERATE_DIGEST")
        return digest_res.get("digest_text", "")

    elif module == "MOTION":
        motion_reply = await run_in_threadpool(
            persona_router.process_motion_request,
            text=text,
            sender_id=sender_id,
        )
        if sender_id:
            conversation_memory.update_query_state(sender_id, last_module="MOTION", last_intent="STRATEGIC_CONSULTATION")
        return motion_reply

    else:
        return f"📝 *Recorded:*\n{text}"


@app.post("/webhook")
async def whatsapp_webhook(
    background_tasks: BackgroundTasks,
    request: Request = None,
):
    """WhatsApp Cloud API webhook event endpoint."""
    payload: Dict[str, Any] = {}
    if request is not None:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

    event = _extract_whatsapp_event(payload)
    sender_phone = event.get("sender")
    msg_type = event.get("type")

    if not sender_phone or not msg_type:
        logger.info("WhatsApp webhook received non-message or empty event.")
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    whatsapp_client = WhatsAppAssistantClient()
    notion_client = NotionAssistantClient()

    # 1. Handle Image / Media Messages in WhatsApp
    if msg_type == "image":
        media_id = event.get("media_id")
        caption = event.get("caption", "")
        if media_id:
            try:
                # Immediate acknowledgement
                await run_in_threadpool(
                    whatsapp_client.send_message,
                    to=sender_phone,
                    text="Analyzing your image with Gemini Vision...",
                )
                conversation_memory.add_assistant_message(sender_phone, "Analyzing your image with Gemini Vision...", module="MIND")

                # Download bytes and enqueue background vision pipeline
                image_bytes, mime_type = await run_in_threadpool(
                    whatsapp_client.download_media_bytes,
                    media_id=media_id,
                )
                background_tasks.add_task(
                    execute_media_pipeline,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    caption=caption,
                    to_phone=sender_phone,
                )
            except Exception as media_err:
                logger.error("Failed to process WhatsApp media message: %s", media_err)
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    # 2. Handle Text Messages in WhatsApp
    text = event.get("text", "")
    if not text:
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    # Check for 1-Tap URL Ingestion
    if is_url_dominant_message(text):
        urls = extract_urls(text)
        if urls:
            primary_url = urls[0]
            try:
                await run_in_threadpool(
                    whatsapp_client.send_message,
                    to=sender_phone,
                    text="Digesting link with Gemini...",
                )
                conversation_memory.add_assistant_message(sender_phone, "Digesting link with Gemini...", module="LEARNING")
            except Exception as wa_err:
                logger.error("Failed to send WhatsApp URL digest ack: %s", wa_err)

            background_tasks.add_task(
                execute_url_digest_background_pipeline,
                url=primary_url,
                user_comment=text,
                to_phone=sender_phone,
            )
            return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    # 0. Check for interactive disambiguation menu reply
    pending_menu_reply = await handle_pending_menu_reply(sender_phone, text, notion_client)
    if pending_menu_reply:
        try:
            await run_in_threadpool(whatsapp_client.send_message, to=sender_phone, text=pending_menu_reply)
            conversation_memory.add_assistant_message(sender_phone, pending_menu_reply, module="TASK_ACTION")
        except Exception as wa_err:
            logger.error("Failed to send WhatsApp menu reply: %s", wa_err)
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    # 0.5 Check for Motion Persona Routing (@motion or motion:)
    if persona_router.is_motion_invoked(text):
        conversation_memory.add_user_message(sender_phone, text)
        reply_text = await run_in_threadpool(
            persona_router.process_motion_request,
            text=text,
            sender_id=sender_phone,
        )
        try:
            await run_in_threadpool(whatsapp_client.send_message, to=sender_phone, text=reply_text)
            conversation_memory.add_assistant_message(sender_phone, reply_text, module="MOTION")
        except Exception as wa_err:
            logger.error("Failed to send WhatsApp Motion reply: %s", wa_err)
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    # 1. Add user message to conversation memory & get rolling context + graph memory
    conversation_memory.add_user_message(sender_phone, text)
    context = conversation_memory.format_context_prompt(sender_phone, max_turns=4)
    graph_ctx = graph_memory.format_graph_context_for_prompt(text)
    if graph_ctx:
        context = f"{graph_ctx}\n\n{context}" if context else graph_ctx

    # 2. Two-Stage Gemini Classification & Parsing with Context
    module, parsed_result = await run_in_threadpool(analyze_user_text_two_stage, text, context=context)

    # Enqueue background entity extraction for knowledge modules
    if module in ("MIND", "LEARNING", "LEETCODE", "DOCUMENT_APPEND"):
        background_tasks.add_task(graph_memory.extract_and_index_entities_from_text, text, module)

    if module == "LEARNING":
        # 1. Immediately reply on WhatsApp with short acknowledgement
        try:
            await run_in_threadpool(
                whatsapp_client.send_message,
                to=sender_phone,
                text="Building your study plan...",
            )
            conversation_memory.add_assistant_message(sender_phone, "Building your study plan...", module="LEARNING")
        except Exception as wa_err:
            logger.error("Failed to send WhatsApp learning ack: %s", wa_err)

        # 2. Add background task to compile curriculum and write to Notion
        background_tasks.add_task(
            execute_learning_background_pipeline,
            parsed_result,
            to_phone=sender_phone,
        )

        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    if module == "LEETCODE":
        # 1. Immediately reply on WhatsApp with short acknowledgement
        try:
            await run_in_threadpool(
                whatsapp_client.send_message,
                to=sender_phone,
                text="Pulling your latest solution...",
            )
            conversation_memory.add_assistant_message(sender_phone, "Pulling your latest solution...", module="LEETCODE")
        except Exception as wa_err:
            logger.error("Failed to send WhatsApp leetcode ack: %s", wa_err)

        # 2. Add background task to pull code, fetch constraints, review, write to Notion, and follow up
        background_tasks.add_task(
            execute_leetcode_background_pipeline,
            parsed_result,
            to_phone=sender_phone,
        )

        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    if module == "DIGEST":
        try:
            await run_in_threadpool(
                whatsapp_client.send_message,
                to=sender_phone,
                text="Compiling your weekly velocity digest...",
            )
            conversation_memory.add_assistant_message(sender_phone, "Compiling your weekly velocity digest...", module="DIGEST")
        except Exception as wa_err:
            logger.error("Failed to send WhatsApp digest ack: %s", wa_err)

        background_tasks.add_task(
            execute_weekly_digest_pipeline,
            to_phone=sender_phone,
        )
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    if module == "SEARCH":
        search_analysis = parsed_result if isinstance(parsed_result, SearchQueryAnalysis) else SearchQueryAnalysis(query=text)
        search_q = search_analysis.query
        try:
            await run_in_threadpool(
                whatsapp_client.send_message,
                to=sender_phone,
                text="Searching your second brain...",
            )
            conversation_memory.add_assistant_message(sender_phone, "Searching your second brain...", module="SEARCH")
        except Exception as wa_err:
            logger.error("Failed to send WhatsApp search ack: %s", wa_err)

        background_tasks.add_task(
            execute_second_brain_search_pipeline,
            query=search_q,
            search_analysis=search_analysis,
            to_phone=sender_phone,
        )
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

    # For other modules (TASKS, MIND)
    try:
        reply_text = await _handle_module_action(module, parsed_result, text, notion_client, sender_id=sender_phone)
        if reply_text:
            await run_in_threadpool(
                whatsapp_client.send_message,
                to=sender_phone,
                text=reply_text,
            )
            conversation_memory.add_assistant_message(sender_phone, reply_text, module=module)
    except Exception as exc:
        logger.error("Failed to process WhatsApp message: %s", exc)

    return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)


@app.post("/webhook/telegram")
async def telegram_webhook(
    update: TelegramWebhookUpdate,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None, alias="X-Telegram-Bot-Api-Secret-Token")
):
    """Receive and process incoming Telegram webhooks."""
    # 1. Verify secret token header against config
    secret_in_config = (
        os.getenv("TELEGRAM_WEBHOOK_SECRET")
        or os.getenv("WEBHOOK_SECRET")
        or getattr(settings, "TELEGRAM_WEBHOOK_SECRET", "")
        or getattr(settings, "WEBHOOK_SECRET", "")
    )
    if secret_in_config and x_telegram_bot_api_secret_token != secret_in_config:
        logger.warning("Unauthorized webhook request: secret token mismatch.")
        raise HTTPException(status_code=401, detail="Invalid secret token")

    telegram_client = TelegramAssistantClient()
    notion_client = NotionAssistantClient()

    msg = update.message or {}
    chat_id = msg.get("chat", {}).get("id")
    sender_id = str(chat_id) if chat_id else "unknown_tg"

    # 1. Handle Photo / Image Document in Telegram
    photos = msg.get("photo", [])
    document = msg.get("document", {})
    caption = msg.get("caption", "")

    file_id = None
    if photos and isinstance(photos, list):
        file_id = photos[-1].get("file_id")
    elif document and isinstance(document, dict) and document.get("mime_type", "").startswith("image/"):
        file_id = document.get("file_id")

    if file_id and chat_id:
        try:
            await run_in_threadpool(
                telegram_client.send_message,
                text="Analyzing your image with Gemini Vision...",
                chat_id=str(chat_id),
            )
            conversation_memory.add_assistant_message(sender_id, "Analyzing your image with Gemini Vision...", module="MIND")

            image_bytes, mime_type = await run_in_threadpool(
                telegram_client.download_file_bytes,
                file_id=file_id,
            )
            background_tasks.add_task(
                execute_media_pipeline,
                image_bytes=image_bytes,
                mime_type=mime_type,
                caption=caption,
                chat_id=str(chat_id),
            )
        except Exception as media_err:
            logger.error("Failed to process Telegram media message: %s", media_err)
        return {"status": "ok"}

    # 2. Handle Text Messages in Telegram
    text = msg.get("text")
    if not text:
        logger.info("Webhook request ignored: no message text present.")
        return {"status": "ignored"}

    # Check for 1-Tap URL Ingestion
    if is_url_dominant_message(text):
        urls = extract_urls(text)
        if urls:
            primary_url = urls[0]
            if chat_id:
                try:
                    await run_in_threadpool(
                        telegram_client.send_message,
                        text="Digesting link with Gemini...",
                        chat_id=str(chat_id),
                    )
                    conversation_memory.add_assistant_message(sender_id, "Digesting link with Gemini...", module="LEARNING")
                except Exception as tg_err:
                    logger.error("Failed to send Telegram URL digest ack: %s", tg_err)

            background_tasks.add_task(
                execute_url_digest_background_pipeline,
                url=primary_url,
                user_comment=text,
                chat_id=str(chat_id) if chat_id else None,
            )
            return {"status": "ok"}

    # 0. Check for interactive disambiguation menu reply
    pending_menu_reply = await handle_pending_menu_reply(sender_id, text, notion_client)
    if pending_menu_reply:
        if chat_id:
            try:
                await run_in_threadpool(telegram_client.send_message, text=pending_menu_reply, chat_id=str(chat_id))
                conversation_memory.add_assistant_message(sender_id, pending_menu_reply, module="TASK_ACTION")
            except Exception as tg_err:
                logger.error("Failed to send Telegram menu reply: %s", tg_err)
        return {"status": "ok"}

    # 0.5 Check for Motion Persona Routing (@motion or motion:)
    if persona_router.is_motion_invoked(text):
        conversation_memory.add_user_message(sender_id, text)
        reply_text = await run_in_threadpool(
            persona_router.process_motion_request,
            text=text,
            sender_id=sender_id,
        )
        if chat_id:
            try:
                await run_in_threadpool(telegram_client.send_message, text=reply_text, chat_id=str(chat_id))
                conversation_memory.add_assistant_message(sender_id, reply_text, module="MOTION")
            except Exception as tg_err:
                logger.error("Failed to send Telegram Motion reply: %s", tg_err)
        return {"status": "ok"}

    # 1. Add user message to conversation memory & get rolling context + graph memory
    conversation_memory.add_user_message(sender_id, text)
    context = conversation_memory.format_context_prompt(sender_id, max_turns=4)
    graph_ctx = graph_memory.format_graph_context_for_prompt(text)
    if graph_ctx:
        context = f"{graph_ctx}\n\n{context}" if context else graph_ctx

    # 3 & 4. Two-Stage Gemini Classification and Parsing with Context
    module, parsed_result = await run_in_threadpool(analyze_user_text_two_stage, text, context=context)

    # Enqueue background entity extraction for knowledge modules
    if module in ("MIND", "LEARNING", "LEETCODE", "DOCUMENT_APPEND"):
        background_tasks.add_task(graph_memory.extract_and_index_entities_from_text, text, module)

    # If module is LEARNING, acknowledge immediately and enqueue background task
    if module == "LEARNING":
        if chat_id:
            try:
                await run_in_threadpool(
                    telegram_client.send_message,
                    text="Building your study plan...",
                    chat_id=str(chat_id),
                )
                conversation_memory.add_assistant_message(sender_id, "Building your study plan...", module="LEARNING")
            except Exception as tg_err:
                logger.error("Failed to send Telegram learning ack: %s", tg_err)

        background_tasks.add_task(
            execute_learning_background_pipeline,
            parsed_result,
            chat_id=str(chat_id) if chat_id else None,
        )
        return {"status": "ok"}

    # If module is LEETCODE, acknowledge immediately and enqueue background task
    if module == "LEETCODE":
        if chat_id:
            try:
                await run_in_threadpool(
                    telegram_client.send_message,
                    text="Pulling your latest solution...",
                    chat_id=str(chat_id),
                )
                conversation_memory.add_assistant_message(sender_id, "Pulling your latest solution...", module="LEETCODE")
            except Exception as tg_err:
                logger.error("Failed to send Telegram leetcode ack: %s", tg_err)

        background_tasks.add_task(
            execute_leetcode_background_pipeline,
            parsed_result,
            chat_id=str(chat_id) if chat_id else None,
        )
        return {"status": "ok"}

    # If module is DIGEST, acknowledge immediately and enqueue background task
    if module == "DIGEST":
        if chat_id:
            try:
                await run_in_threadpool(
                    telegram_client.send_message,
                    text="Compiling your weekly velocity digest...",
                    chat_id=str(chat_id),
                )
                conversation_memory.add_assistant_message(sender_id, "Compiling your weekly velocity digest...", module="DIGEST")
            except Exception as tg_err:
                logger.error("Failed to send Telegram digest ack: %s", tg_err)

        background_tasks.add_task(
            execute_weekly_digest_pipeline,
            chat_id=str(chat_id) if chat_id else None,
        )
        return {"status": "ok"}

    # If module is SEARCH, acknowledge immediately and enqueue background task
    if module == "SEARCH":
        search_analysis = parsed_result if isinstance(parsed_result, SearchQueryAnalysis) else SearchQueryAnalysis(query=text)
        search_q = search_analysis.query
        if chat_id:
            try:
                await run_in_threadpool(
                    telegram_client.send_message,
                    text="Searching your second brain...",
                    chat_id=str(chat_id),
                )
                conversation_memory.add_assistant_message(sender_id, "Searching your second brain...", module="SEARCH")
            except Exception as tg_err:
                logger.error("Failed to send Telegram search ack: %s", tg_err)

        background_tasks.add_task(
            execute_second_brain_search_pipeline,
            query=search_q,
            search_analysis=search_analysis,
            chat_id=str(chat_id) if chat_id else None,
        )
        return {"status": "ok"}

    # 5. Route other modules to appropriate synchronous action
    try:
        reply_text = await _handle_module_action(module, parsed_result, text, notion_client, sender_id=sender_id)

        if chat_id and reply_text:
            try:
                await run_in_threadpool(
                    telegram_client.send_message,
                    text=reply_text,
                    chat_id=str(chat_id)
                )
                conversation_memory.add_assistant_message(sender_id, reply_text, module=module)
            except Exception as tg_err:
                logger.error("Failed to send message to Telegram chat_id=%s: %s", chat_id, tg_err)

        # Log request completion
        logger.info(
            "Webhook request processed successfully: chat_id=%s, module=%s, status=success",
            chat_id,
            module,
        )
        return {"status": "ok"}

    except Exception as exc:
        # Log failure
        logger.error(
            "Webhook request failed: chat_id=%s, module=%s, status=failure, error=%s",
            chat_id,
            module,
            exc,
        )
        return {"status": "error", "message": str(exc)}


# ==========================================
# --- Ocean Motion REST Endpoints (v4.0) ---
# ==========================================

@app.post("/motion/consult", response_model=Dict[str, Any])
async def motion_consult(request: Request):
    """Direct strategic consultation endpoint for Motion persona."""
    body = await request.json()
    message = body.get("message", "")
    sender_id = body.get("sender_id", "direct_api")
    if not message:
        raise HTTPException(status_code=400, detail="Missing required 'message' in request body.")

    response = await run_in_threadpool(
        motion_mentorship_service.consult,
        user_message=message,
        sender_id=sender_id,
    )
    return response.model_dump()


@app.get("/motion/trajectory", response_model=Dict[str, Any])
async def get_motion_trajectory():
    """Retrieve current strategic trajectory."""
    trajectory = await run_in_threadpool(strategic_model_service.get_trajectory)
    return trajectory.model_dump()


@app.get("/motion/identity", response_model=Dict[str, Any])
async def get_motion_identity():
    """Retrieve stable user identity."""
    identity = await run_in_threadpool(strategic_model_service.get_identity)
    return identity.model_dump()


@app.put("/motion/identity", response_model=Dict[str, Any])
async def update_motion_identity(identity: MotionIdentity):
    """Explicitly update user identity (never inferred)."""
    updated = await run_in_threadpool(
        strategic_model_service.update_identity,
        new_identity=identity,
        explicit_user_confirmation=True,
    )
    return updated.model_dump()


@app.get("/motion/initiatives", response_model=List[Dict[str, Any]])
async def get_motion_initiatives(status: Optional[str] = Query(None)):
    """Retrieve strategic initiatives, optionally filtered by status."""
    inits = await run_in_threadpool(strategic_model_service.get_initiatives, status=status)
    return [init.model_dump() for init in inits]


@app.post("/motion/initiatives", response_model=Dict[str, Any])
async def create_or_update_initiative(initiative: MotionInitiative):
    """Create or update a strategic initiative."""
    saved = await run_in_threadpool(strategic_model_service.create_or_update_initiative, initiative=initiative)
    return saved.model_dump()


@app.get("/motion/decision-journal", response_model=List[Dict[str, Any]])
async def get_decision_journal(status: Optional[str] = Query(None)):
    """Retrieve Decision Journal records."""
    target_status = DecisionStatus(status) if status else None
    decisions = await run_in_threadpool(strategic_model_service.get_decisions, status=target_status)
    return [d.model_dump() for d in decisions]


@app.post("/motion/decision-journal/{decision_id}/review", response_model=Dict[str, Any])
async def review_decision_journal_entry(decision_id: str, request: Request):
    """Submit actual outcome and user reflection to review and close a Decision Journal record."""
    body = await request.json()
    outcome = body.get("actual_outcome", "")
    reflection = body.get("user_reflection")
    if not outcome:
        raise HTTPException(status_code=400, detail="Missing required 'actual_outcome' field.")

    updated = await run_in_threadpool(
        motion_mentorship_service.handle_decision_review,
        decision_id=decision_id,
        actual_outcome=outcome,
        user_reflection=reflection,
    )
    return updated.model_dump()


@app.post("/motion/overrides", response_model=Dict[str, Any])
async def record_motion_override(override: HumanOverride):
    """Record an explicit human override with a condition-based review trigger."""
    saved = await run_in_threadpool(strategic_model_service.record_override, override=override)
    return saved.model_dump()


@app.post("/motion/review/weekly", response_model=Dict[str, Any])
async def trigger_motion_weekly_review(
    week_start: Optional[str] = Query(None),
    week_end: Optional[str] = Query(None),
):
    """Trigger generation of a weekly strategic review."""
    review = await run_in_threadpool(
        motion_review_service.execute_weekly_review,
        week_start=week_start,
        week_end=week_end,
    )
    return review.model_dump()


@app.post("/motion/review/daily", response_model=Dict[str, Any])
async def trigger_motion_daily_review(request: Request):
    """Trigger processing of daily activities into atomic evidence and observations."""
    body = await request.json()
    activities = body.get("activities", [])
    date_str = body.get("date")
    result = await run_in_threadpool(
        motion_review_service.execute_daily_review,
        raw_activities=activities,
        date_str=date_str,
    )
    return result


@app.get("/motion/provenance/{target_id}", response_model=Dict[str, Any])
async def get_motion_provenance(target_id: str):
    """Explain reasoning and trace provenance for a conclusion or observation ('Why do I believe this?')."""
    explanation = await run_in_threadpool(strategic_model_service.explain_belief, target_id=target_id)
    return explanation


# --- Motion v2 Endpoints ---
@app.get("/motion/drift", response_model=Dict[str, Any])
async def get_motion_drift_report(
    window_days: int = Query(7, ge=1, le=60),
    reference_date: Optional[str] = Query(None),
):
    """Evaluate and retrieve current Strategic Drift and Alignment Report."""
    report = await run_in_threadpool(
        drift_detector.evaluate_drift,
        window_days=window_days,
        reference_date=reference_date,
        save_report=True,
    )
    return report.model_dump()


@app.get("/motion/evidence", response_model=List[Dict[str, Any]])
async def list_motion_evidence(
    limit: int = Query(50, ge=1, le=200),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Query discrete atomic evidence items."""
    evidence_items = await run_in_threadpool(
        motion_storage.load_evidence,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
    )
    return [e.model_dump() for e in evidence_items]


@app.post("/motion/evidence", response_model=Dict[str, Any])
async def ingest_motion_evidence(event: EvidenceIngestionEvent):
    """Ingest a single evidence event asynchronously or directly."""
    evidence = await run_in_threadpool(evidence_ingestion_engine.ingest_event, event=event)
    return evidence.model_dump()


@app.post("/motion/synthesis", response_model=Dict[str, Any])
async def trigger_motion_synthesis(reference_date: Optional[str] = Query(None)):
    """Trigger multi-window observation and conclusion synthesis."""
    observations, conclusions = await run_in_threadpool(
        synthesis_engine.run_synthesis,
        reference_date=reference_date,
        save_records=True,
    )
    return {
        "status": "ok",
        "observations_generated": len(observations),
        "conclusions_generated": len(conclusions),
        "observation_ids": [o.id for o in observations],
        "conclusion_ids": [c.id for c in conclusions],
    }


@app.get("/motion/attribution/{target_id}", response_model=Dict[str, Any])
async def get_motion_attribution(target_id: str):
    """Retrieve full multi-hop causal attribution tree for an item."""
    tree = await run_in_threadpool(multi_hop_attribution_engine.build_attribution_tree, target_id=target_id)
    return tree.model_dump()


@app.get("/motion/checkin", response_model=Dict[str, Any])
async def get_motion_daily_checkin(current_date: Optional[str] = Query(None)):
    """Generate daily proactive strategic check-in status prompt."""
    drift_report = await run_in_threadpool(
        drift_detector.evaluate_drift,
        window_days=7,
        reference_date=current_date,
        save_report=False,
    )
    checkin = await run_in_threadpool(
        accountability_monitor.generate_daily_checkin,
        current_date=current_date,
        drift_report=drift_report,
    )
    return checkin.model_dump()


# --- Motion v3 REST Endpoints ---

@app.post("/motion/simulate", response_model=Dict[str, Any])
async def simulate_motion_scenario(request: ScenarioSimulationRequest):
    """Simulate what-if time and priority reallocation scenario."""
    result = await run_in_threadpool(
        scenario_engine.simulate_scenario,
        request=request,
        save_result=True,
    )
    return result.model_dump()


@app.get("/motion/energy", response_model=Dict[str, Any])
async def get_motion_energy_report(
    window_days: int = Query(7, ge=1, le=30),
    reference_date: Optional[str] = Query(None),
):
    """Retrieve cognitive sustainability and fatigue risk evaluation."""
    report = await run_in_threadpool(
        energy_monitor.evaluate_sustainability,
        window_days=window_days,
        reference_date=reference_date,
        save_report=True,
    )
    return report.model_dump()


@app.get("/motion/milestones", response_model=List[Dict[str, Any]])
async def get_motion_milestones(initiative_id: Optional[str] = Query(None)):
    """Retrieve active initiative milestones."""
    milestones = await run_in_threadpool(
        milestone_engine.get_milestones,
        initiative_id=initiative_id,
    )
    return [m.model_dump() for m in milestones]


@app.post("/motion/milestones", response_model=Dict[str, Any])
async def create_motion_milestone(milestone: InitiativeMilestone):
    """Create or update an initiative milestone."""
    saved_ms = await run_in_threadpool(milestone_engine.record_milestone, milestone=milestone)
    return saved_ms.model_dump()


@app.get("/motion/milestones/critical-path", response_model=Dict[str, Any])
async def get_motion_critical_path(
    initiative_id: Optional[str] = Query(None),
    reference_date: Optional[str] = Query(None),
):
    """Analyze critical path DAG and detect milestone schedule bottlenecks."""
    analysis = await run_in_threadpool(
        milestone_engine.analyze_critical_path,
        initiative_id=initiative_id,
        reference_date=reference_date,
    )
    return analysis


@app.get("/motion/briefing", response_model=Dict[str, Any])
async def get_motion_executive_briefing(
    window_days: int = Query(7, ge=1, le=30),
    reference_date: Optional[str] = Query(None),
):
    """Generate or retrieve autonomous weekly executive strategic briefing."""
    briefing = await run_in_threadpool(
        executive_briefing_engine.generate_briefing,
        window_days=window_days,
        reference_date=reference_date,
        save_briefing=True,
    )
    return briefing.model_dump()


@app.post("/motion/socratic", response_model=Dict[str, Any])
async def conduct_motion_socratic_inquiry(topic: str = Body(..., embed=True)):
    """Perform Socratic assumption deconstruction and pre-mortem failure analysis."""
    inquiry = await run_in_threadpool(socratic_engine.conduct_inquiry, topic=topic)
    return inquiry.model_dump()


