"""Stage 1 & Stage 2 Intent Classifiers and Semantic Parsers.

Orchestrates multi-stage Gemini processing:
- Stage 1: Lightweight intent classification to route user messages into functional modules.
- Stage 2: Module-specific semantic parameter extraction.
"""

import logging
from typing import Any, Optional, Tuple, Union

from app.ai import generate_structured
from app.schemas import (
    BatchTaskActionAnalysis,
    DocumentAppendAnalysis,
    LearningRequest,
    LeetcodeReviewRequest,
    MemoryGovernanceAnalysis,
    MindEntry,
    ModuleClassification,
    RollbackAnalysis,
    SearchQueryAnalysis,
    TaskActionAnalysis,
    TaskAnalysis,
)

logger = logging.getLogger("notion-assistant.parsers")

# ==========================================
# --- Stage 1: Module Classification ---
# ==========================================

STAGE1_SYSTEM_INSTRUCTION = (
    "You are a lightweight intent routing classifier. Classify the user's message into exactly one MODULE:\n"
    "- ROLLBACK: Undoing an action Ocean just performed, reverting a change Ocean just made, or correcting Ocean's own mistake from the last turn (e.g. 'undo that', 'revert', 'cancel last action', 'no, put them in reading list instead', 'undo last task'). Also use for compound rollback + reroute (e.g. 'no, delete those two tasks and put it in reading list'). Do NOT use ROLLBACK when the user wants to delete a specific named task — use TASK_ACTION for that.\n"
    "- DOCUMENT_APPEND: Adding or appending books, articles, notes, ideas, thoughts, bullets, or to-do items into existing Notion workspace pages or databases (e.g. 'add to my reading list: 1. The pragmatic programmer 2. A philosophy of software design', 'add Build AI voice agent to Ideas for projects', 'append to Year 1 Budget: Laptop insurance 200 USD', 'add note to Finances for Umass fall: Email bursar', 'put into Reading List: Clean Code').\n"
    "- TASKS: Creating single new tasks, querying pending or today's tasks, priority queries (e.g. 'high priority tasks'), and conversational follow-ups (e.g. 'others?', 'show more', 'next').\n"
    "- TASK_ACTION: Modifying or deleting a specific existing task by name (e.g. 'mark Berkshire Dining done', 'done with GPAF form', 'postpone Berkshire Dining to next Tuesday', 'delete gemini shrine task', 'remove duplicate task', 'archive the CICS scholarship task'). Use this whenever the user names or describes a specific task to delete, archive, or update — even if they say 'delete that X task'.\n"
    "- BATCH_TASK_ACTION: Executing batch task commands across multiple tasks ('mark all UMass tasks as done', 'postpone all high priority items by 3 days', 'archive all completed tasks').\n"
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
    """Stage 1: Classify user message into target functional module."""
    fallback = ModuleClassification(module="TASKS", raw_text=text)
    result = generate_structured(
        prompt=text,
        schema=ModuleClassification,
        system_instruction=STAGE1_SYSTEM_INSTRUCTION,
        context=context,
        fallback_default=fallback,
    )
    if not result.raw_text:
        result.raw_text = text
    return result


# ==========================================
# --- Stage 2: Module-Specific Parsers ---
# ==========================================

STAGE2_MEMORY_CONTROL_INSTRUCTION = (
    "Extract memory governance details from the user message:\n"
    "- command: FORGET (if user asks to forget/delete memory like 'forget grad school notes', 'remove X from memory'), "
    "UPDATE_STATUS (if user updates current status like 'update memory: I am enrolled at UMass', 'I started my MSCS'), "
    "INSPECT_MEMORY (if user asks what Ocean remembers like 'what do you remember about UMass?', 'show memory for X').\n"
    "- target_entity: The core entity or topic name.\n"
    "- new_state_summary: Summary of updated memory thesis or state if updating status."
)


def parse_memory_control_stage2(text: str, context: Optional[str] = None) -> MemoryGovernanceAnalysis:
    """Stage 2: Parse MEMORY_CONTROL module intent."""
    fallback = MemoryGovernanceAnalysis(command="FORGET", target_entity=text)
    return generate_structured(
        prompt=text,
        schema=MemoryGovernanceAnalysis,
        system_instruction=STAGE2_MEMORY_CONTROL_INSTRUCTION,
        context=context,
        fallback_default=fallback,
    )


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
    """Stage 2: Parse BATCH_TASK_ACTION module intent."""
    fallback = BatchTaskActionAnalysis(action="MARK_DONE", target_query=text)
    return generate_structured(
        prompt=text,
        schema=BatchTaskActionAnalysis,
        system_instruction=STAGE2_BATCH_TASK_ACTION_INSTRUCTION,
        context=context,
        fallback_default=fallback,
    )


STAGE2_TASK_ACTION_INSTRUCTION = (
    "Extract task modification details from the user message:\n"
    "- action: MARK_DONE (e.g. 'done with X', 'mark X done', 'complete X'), MARK_IN_PROGRESS (e.g. 'set X to in progress', 'working on X'), UPDATE_DUE_DATE (e.g. 'postpone X to tuesday', 'move X to August 30', 'push X to tomorrow'), DELETE_TASK (e.g. 'delete X', 'remove X from tasks', 'archive X task').\n"
    "- task_target_title: Target task title or keywords to match.\n"
    "- new_due_date_iso: Resolved due date in YYYY-MM-DD format if updating due date.\n"
    "- new_status_name: Done, In progress, or Not started.\n"
    "- ordinal_index: 1 for first task, 2 for second task, etc. if user referred to position."
)


def parse_task_action_stage2(text: str, context: Optional[str] = None) -> TaskActionAnalysis:
    """Stage 2: Parse TASK_ACTION module intent."""
    fallback = TaskActionAnalysis(action="MARK_DONE", task_target_title=text)
    return generate_structured(
        prompt=text,
        schema=TaskActionAnalysis,
        system_instruction=STAGE2_TASK_ACTION_INSTRUCTION,
        context=context,
        fallback_default=fallback,
    )


STAGE2_ROLLBACK_INSTRUCTION = (
    "Extract rollback and corrective rerouting details from the user message:\n"
    "- command: ROLLBACK_LAST (if user simply asks to undo/cancel/delete previous action like 'undo that', 'revert', 'delete those tasks', 'cancel last action'), "
    "CORRECTION_AND_REROUTE (if user asks to undo the previous action AND put the items into a new target like 'no, delete those two tasks and put it in reading list', 'no, put them in reading list instead', 'delete that and add to Notes').\n"
    "- target_mutation_id: Optional specific mutation ID if mentioned.\n"
    "- correction_instruction: The follow-up correction command (e.g. 'put it in reading list').\n"
    "- new_target_title: The destination document or database name (e.g. 'Reading List', 'Notes', 'Ideas for projects').\n"
    "- extracted_items: List of item names/titles extracted from context or message (e.g. ['The Pragmatic Programmer', 'A Philosophy of Software Design'])."
)


def parse_rollback_stage2(text: str, context: Optional[str] = None) -> RollbackAnalysis:
    """Stage 2: Parse ROLLBACK module details."""
    text_lower = text.lower()
    if "reading list" in text_lower or "put it in" in text_lower or "put them in" in text_lower:
        fallback = RollbackAnalysis(
            command="CORRECTION_AND_REROUTE",
            new_target_title="Reading List" if "reading" in text_lower else "Notes",
            correction_instruction=text,
        )
    else:
        fallback = RollbackAnalysis(command="ROLLBACK_LAST", correction_instruction=text)

    return generate_structured(
        prompt=text,
        schema=RollbackAnalysis,
        system_instruction=STAGE2_ROLLBACK_INSTRUCTION,
        context=context,
        fallback_default=fallback,
    )


STAGE2_DOCUMENT_APPEND_INSTRUCTION = (
    "Extract workspace ingestion / document append details from the user message:\n"
    "- target_name: Exact title of target database or document page (e.g. 'Reading List', 'Ideas for projects', 'Year 1 Budget', 'Finances for Umass fall', 'Notes', 'Media').\n"
    "- items: List of items to add. For each item extract: title (name/book/idea), optional author (if mentioned), optional details, optional status (e.g. 'Want to Read').\n"
    "- content_to_append: Raw fallback text of items to append.\n"
    "- block_type: 'bulleted_list_item' (default), 'to_do', 'paragraph', or 'callout'.\n"
    "- default_status: Default status if database has a Status property (e.g. 'Want to Read')."
)


def parse_document_append_stage2(text: str, context: Optional[str] = None) -> DocumentAppendAnalysis:
    """Stage 2: Parse DOCUMENT_APPEND module details."""
    fallback = DocumentAppendAnalysis(target_document_title="Notes", content_to_append=text)
    return generate_structured(
        prompt=text,
        schema=DocumentAppendAnalysis,
        system_instruction=STAGE2_DOCUMENT_APPEND_INSTRUCTION,
        context=context,
        fallback_default=fallback,
    )


STAGE2_TASKS_INSTRUCTION = (
    "Extract task management details from the user message:\n"
    "- CREATE_TASK: Adding a new task, with optional priority (High/Medium/Low), tag, due_date (YYYY-MM-DD), description.\n"
    "- UPDATE_TASK: Updating status of an existing task (In progress/Done/Not started) or due date.\n"
    "- QUERY_TODAY: Querying tasks due today.\n"
    "- QUERY_PENDING: Querying pending or upcoming tasks. If user asks for high/medium/low priority (e.g. 'high priority tasks', 'urgent tasks'), set priority_filter='High' (or Medium/Low). If user asks 'others?', 'more', 'next', 'what else?', set is_followup=True and intent='QUERY_PENDING'.\n"
    "- DAILY_LOG: Recording daily notes or log entry."
)


def parse_tasks_stage2(text: str, context: Optional[str] = None) -> TaskAnalysis:
    """Stage 2: Parse TASKS module intent and task structure."""
    fallback = TaskAnalysis(
        intent="DAILY_LOG",
        title=text[:50] if text else "Daily Log",
        log_content=text,
    )
    return generate_structured(
        prompt=text,
        schema=TaskAnalysis,
        system_instruction=STAGE2_TASKS_INSTRUCTION,
        context=context,
        fallback_default=fallback,
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
    """Stage 2: Parse MIND module entry (substack draft, rambling, daily log)."""
    first_sent = text.strip().split(".")[0].strip() if text else "Daily reflection"
    fallback = MindEntry(
        entry_type="DAILY_LOG",
        title=text[:50] if text else "Daily Reflection",
        core_thesis=(first_sent + ".") if first_sent and not first_sent.endswith(".") else first_sent,
        content=text,
    )
    result = generate_structured(
        prompt=text,
        schema=MindEntry,
        system_instruction=STAGE2_MIND_INSTRUCTION,
        fallback_default=fallback,
    )
    if not result.content:
        result.content = text
    if not result.title:
        result.title = text[:50] if text else "Untitled Entry"
    if not result.core_thesis:
        sent = (result.content or text).strip().split(".")[0].strip()
        result.core_thesis = (sent + ".") if sent else "Daily reflection."
    return result


STAGE2_LEARNING_INSTRUCTION = (
    "Extract learning/study request details from the user message:\n"
    "- topic: The subject or topic to study (e.g. Distributed Systems, Rust Borrow Checker).\n"
    "- category: Broader domain (e.g. Computer Science, Math, Systems Engineering).\n"
    "- goal: Specific learning objective, subtopics, or questions.\n"
    "- proficiency_level: Beginner, Intermediate, or Advanced.\n"
    "- resources_requested: Any specific resources, books, papers, tutorials, or roadmaps asked for."
)


def parse_learning_stage2(text: str) -> LearningRequest:
    """Stage 2: Parse LEARNING module request."""
    fallback = LearningRequest(
        topic=text[:50] if text else "New Study Topic",
        goal=text,
    )
    return generate_structured(
        prompt=text,
        schema=LearningRequest,
        system_instruction=STAGE2_LEARNING_INSTRUCTION,
        fallback_default=fallback,
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
    """Stage 2: Parse LEETCODE module review request."""
    fallback = LeetcodeReviewRequest(
        problem_name=text[:50] if text else "LeetCode Review",
        review_notes=text,
    )
    return generate_structured(
        prompt=text,
        schema=LeetcodeReviewRequest,
        system_instruction=STAGE2_LEETCODE_INSTRUCTION,
        fallback_default=fallback,
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
    """Stage 2: Parse SEARCH module parameters."""
    fallback = SearchQueryAnalysis(query=text)
    return generate_structured(
        prompt=text,
        schema=SearchQueryAnalysis,
        system_instruction=STAGE2_SEARCH_INSTRUCTION,
        context=context,
        fallback_default=fallback,
    )


# ==========================================
# --- Two-Stage Pipeline Orchestration ---
# ==========================================

def analyze_user_text_two_stage(
    text: str,
    context: Optional[str] = None,
) -> Tuple[str, Union[TaskAnalysis, MindEntry, LearningRequest, LeetcodeReviewRequest, SearchQueryAnalysis, str]]:
    """Execute two-stage Gemini pipeline: Stage 1 classification -> Stage 2 module-specific parsing."""
    import sys
    app_main = sys.modules.get("app.main")
    classify_fn = getattr(app_main, "classify_module_stage1", classify_module_stage1) if app_main else classify_module_stage1

    stage1_res = classify_fn(text, context=context) if context else classify_fn(text)
    module = stage1_res.module
    raw_text = stage1_res.raw_text or text

    # Anti-rambling guardrail: Short ambiguous queries (<= 4 words) ending with '?' or conversational follow-ups
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

    get_fn = lambda name, fallback: getattr(app_main, name, fallback) if app_main else fallback

    if module == "TASKS":
        p_fn = get_fn("parse_tasks_stage2", parse_tasks_stage2)
        parsed = p_fn(raw_text, context=context) if context else p_fn(raw_text)
    elif module == "TASK_ACTION":
        p_fn = get_fn("parse_task_action_stage2", parse_task_action_stage2)
        parsed = p_fn(raw_text, context=context) if context else p_fn(raw_text)
    elif module == "BATCH_TASK_ACTION":
        p_fn = get_fn("parse_batch_task_action_stage2", parse_batch_task_action_stage2)
        parsed = p_fn(raw_text, context=context) if context else p_fn(raw_text)
    elif module == "DOCUMENT_APPEND":
        p_fn = get_fn("parse_document_append_stage2", parse_document_append_stage2)
        parsed = p_fn(raw_text, context=context) if context else p_fn(raw_text)
    elif module == "MEMORY_CONTROL":
        p_fn = get_fn("parse_memory_control_stage2", parse_memory_control_stage2)
        parsed = p_fn(raw_text, context=context) if context else p_fn(raw_text)
    elif module == "MIND":
        p_fn = get_fn("parse_mind_stage2", parse_mind_stage2)
        parsed = p_fn(raw_text)
    elif module == "LEARNING":
        p_fn = get_fn("parse_learning_stage2", parse_learning_stage2)
        parsed = p_fn(raw_text)
    elif module == "LEETCODE":
        p_fn = get_fn("parse_leetcode_stage2", parse_leetcode_stage2)
        parsed = p_fn(raw_text)
    elif module == "SEARCH":
        p_fn = get_fn("parse_search_stage2", parse_search_stage2)
        parsed = p_fn(raw_text, context=context) if context else p_fn(raw_text)
    elif module == "ROLLBACK":
        p_fn = get_fn("parse_rollback_stage2", parse_rollback_stage2)
        parsed = p_fn(raw_text, context=context) if context else p_fn(raw_text)
    elif module in ("DIGEST", "MOTION"):
        parsed = raw_text
    else:
        p_fn = get_fn("parse_tasks_stage2", parse_tasks_stage2)
        parsed = p_fn(raw_text, context=context) if context else p_fn(raw_text)

    return module, parsed



def analyze_user_text_with_gemini(text: str) -> TaskAnalysis:
    """Legacy helper: calls Stage 2 TASKS parsing directly for backward compatibility."""
    import sys
    app_main = sys.modules.get("app.main")
    p_fn = getattr(app_main, "parse_tasks_stage2", parse_tasks_stage2) if app_main else parse_tasks_stage2
    return p_fn(text)

