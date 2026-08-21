from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# --- Stage 1: Module Classification Schema ---

class ModuleEnum(str, Enum):
    """Supported module categories for Stage 1 classification."""
    TASKS = "TASKS"
    TASK_ACTION = "TASK_ACTION"
    BATCH_TASK_ACTION = "BATCH_TASK_ACTION"
    DOCUMENT_APPEND = "DOCUMENT_APPEND"
    MEMORY_CONTROL = "MEMORY_CONTROL"
    MIND = "MIND"
    LEARNING = "LEARNING"
    LEETCODE = "LEETCODE"
    SEARCH = "SEARCH"
    DIGEST = "DIGEST"
    MOTION = "MOTION"


class ModuleClassification(BaseModel):
    """Stage 1: Lightweight classification schema deciding target module."""
    module: Literal[
        "TASKS",
        "TASK_ACTION",
        "BATCH_TASK_ACTION",
        "DOCUMENT_APPEND",
        "MEMORY_CONTROL",
        "MIND",
        "LEARNING",
        "LEETCODE",
        "SEARCH",
        "DIGEST",
        "MOTION",
    ] = Field(
        ...,
        description="The target module: TASKS, TASK_ACTION, BATCH_TASK_ACTION, DOCUMENT_APPEND, MEMORY_CONTROL, MIND, LEARNING, LEETCODE, SEARCH, DIGEST, or MOTION."
    )
    raw_text: str = Field(
        default="",
        description="The raw message text passed through."
    )


# --- Stage 2: Module-Specific Schemas ---

class TaskAnalysis(BaseModel):
    """Structured output schema for TASKS module (create/query/update tasks and logs)."""
    intent: Literal["CREATE_TASK", "DAILY_LOG", "QUERY_PENDING", "UPDATE_TASK", "QUERY_TODAY"]
    title: str = ""
    priority: Literal["High", "Medium", "Low"] = "Medium"
    tag: Literal[
        "Finances",
        "UMass Admin",
        "Writing",
        "Personal Site",
        "Substack",
        "Open Source",
        "Learning",
        "Leetcode",
        "Projects",
        "Schoolwork",
        "Miscellaneous",
    ] = "Miscellaneous"
    due_date: Optional[str] = None
    description: Optional[str] = None
    log_content: Optional[str] = None
    target_status: Optional[Literal["In progress", "Done", "Not started"]] = None
    new_due_date: Optional[str] = None
    priority_filter: Optional[Literal["High", "Medium", "Low"]] = None
    tag_filter: Optional[str] = None
    offset: int = 0
    limit: int = 5
    is_followup: bool = False


# Alias for backward compatibility
AgentAction = TaskAnalysis


class MindEntry(BaseModel):
    """Structured output schema for MIND module (substack drafts, rambling, daily logs)."""
    entry_type: Literal["DRAFT_SUBSTACK", "SUBSTACK_DRAFT", "RAMBLING", "DAILY_LOG"] = Field(
        default="DAILY_LOG",
        description="Type of mind entry: DRAFT_SUBSTACK (or SUBSTACK_DRAFT), RAMBLING, or DAILY_LOG."
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional headline or title for the entry or draft."
    )
    core_thesis: Optional[str] = Field(
        default=None,
        description="One-sentence core thesis extracted from the entry."
    )
    content: str = Field(
        default="",
        description="The main text, thoughts, rambling, or drafted content."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Brief summary or takeaway."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags or topics associated with the entry."
    )

    @property
    def sub_intent(self) -> str:
        """Alias for entry_type normalized to DRAFT_SUBSTACK, RAMBLING, or DAILY_LOG."""
        if self.entry_type == "SUBSTACK_DRAFT":
            return "DRAFT_SUBSTACK"
        return self.entry_type



class LearningRequest(BaseModel):
    """Structured output schema for LEARNING module (new study topic requests)."""
    topic: str = Field(
        default="",
        description="Subject or topic to study (e.g. Distributed Systems, Rust Borrow Checker)."
    )
    category: Optional[str] = Field(
        default=None,
        description="Broader subject domain (e.g. Computer Science, Math, Systems)."
    )
    goal: Optional[str] = Field(
        default=None,
        description="Specific learning objective, subtopic, or questions to answer."
    )
    proficiency_level: Optional[Literal["Beginner", "Intermediate", "Advanced"]] = Field(
        default=None,
        description="Target proficiency or difficulty level."
    )
    resources_requested: Optional[str] = Field(
        default=None,
        description="Requested study materials (e.g. roadmap, books, papers, tutorials)."
    )


# --- Learning Synthesis & Resource Schemas ---

LearningTagLiteral = Literal["Learning"]
LEARNING_TAG: LearningTagLiteral = "Learning"

ResourceTypeLiteral = Literal["Article", "Video", "Docs", "Paper"]


class VerifiedResource(BaseModel):
    """Schema representing a link verified via live HTTP liveness check."""
    name: str = Field(
        default="",
        description="Resource title or concise descriptive name."
    )
    url: str = Field(
        ...,
        description="Live, verified URL."
    )
    resource_type: ResourceTypeLiteral = Field(
        default="Article",
        description="Inferred resource type: Article, Video, Docs, or Paper."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Brief 1-sentence overview of what this resource covers."
    )


class LearningPlanSynthesis(BaseModel):
    """Synthesized learning plan compiled via Gemini with web search grounding."""
    subject_title: str = Field(
        default="",
        description="Subject title (e.g. Distributed Systems Fundamentals)."
    )
    curriculum_topics: List[str] = Field(
        default_factory=list,
        description="Flat continuous numbered list of novice-level topics without headers or groupings."
    )
    starter_tasks: List[str] = Field(
        default_factory=list,
        description="1-3 concrete, immediately actionable first steps to begin studying."
    )
    surfaced_resources: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Surfaced resource links with url and title from grounding."
    )


class LeetcodeReviewRequest(BaseModel):
    """Structured output schema for LEETCODE module (problem review requests)."""
    problem_name: str = Field(
        default="",
        description="Problem name or title (e.g. 'Two Sum', 'Trapping Rain Water')."
    )
    problem_number: Optional[int] = Field(
        default=None,
        description="LeetCode problem number if specified."
    )
    difficulty: Optional[Literal["Easy", "Medium", "Hard"]] = Field(
        default=None,
        description="Problem difficulty rating."
    )
    patterns: List[str] = Field(
        default_factory=list,
        description="Algorithmic patterns (e.g. ['Two Pointers', 'Dynamic Programming'])."
    )
    review_notes: Optional[str] = Field(
        default=None,
        description="Key takeaways, pitfalls, edge cases, time/space complexity or review notes."
    )
    status: Optional[Literal["Solved", "Review Needed", "Failed", "Mastered"]] = Field(
        default=None,
        description="Current mastery status."
    )


class LeetcodeCommitData(BaseModel):
    """Schema representing commit data pulled from LeetHub GitHub repo."""
    commit_sha: str = Field(default="", description="Latest commit SHA hash.")
    commit_message: str = Field(default="", description="Commit message generated by LeetHub.")
    problem_title: str = Field(default="", description="Problem title extracted from commit or README.")
    problem_slug: str = Field(default="", description="LeetCode URL slug for the problem.")
    problem_number: Optional[int] = Field(default=None, description="Problem number if parsed.")
    code: str = Field(default="", description="Extracted solution source code.")
    code_file_name: str = Field(default="", description="Name of the solution file.")
    readme_content: Optional[str] = Field(default=None, description="Extracted README content if present.")


class LeetcodeProblemDetails(BaseModel):
    """Schema representing metadata and constraints fetched from LeetCode GraphQL API."""
    title: str = Field(default="", description="Official LeetCode problem title.")
    title_slug: str = Field(default="", description="Official LeetCode problem slug.")
    difficulty: Optional[str] = Field(default=None, description="Difficulty (Easy, Medium, Hard).")
    constraints: List[str] = Field(default_factory=list, description="Extracted problem constraints list.")
    raw_content_html: Optional[str] = Field(default=None, description="Raw problem description HTML.")
    topic_tags: List[str] = Field(default_factory=list, description="Topic tags/categories.")


class LeetcodeReviewResult(BaseModel):
    """Schema representing the completed Gemini LeetCode code review."""
    problem_title: str = Field(default="", description="Problem title.")
    problem_slug: str = Field(default="", description="Problem slug.")
    problem_number: Optional[int] = Field(default=None, description="Problem number.")
    difficulty: Optional[str] = Field(default=None, description="Difficulty rating.")
    verdict: str = Field(default="Correct", description="Review verdict (e.g. Correct, Incorrect, Suboptimal).")
    time_complexity: Optional[str] = Field(default=None, description="Evaluated Time Complexity.")
    space_complexity: Optional[str] = Field(default=None, description="Evaluated Space Complexity.")
    is_optimal: bool = Field(default=True, description="Whether approach is optimal given constraints.")
    review_summary: str = Field(default="", description="Concise assessment and evaluation breakdown.")
    testing_questions: List[str] = Field(default_factory=list, description="Targeted logic-testing questions.")
    full_review_text: str = Field(default="", description="Full formatted review text for messaging/Notion.")
    fallback_mode: bool = Field(default=False, description="True if constraints were unavailable and reviewed code-only.")
    notion_page_url: Optional[str] = Field(default=None, description="Link to created Notion page.")


# --- Common & Telegram Schemas ---

class ReminderItem(BaseModel):
    """Schema representing a Notion task or reminder item."""
    page_id: str
    title: str
    due_date: Optional[str] = None
    status: Optional[str] = None
    details: Optional[str] = None


class TelegramMessage(BaseModel):
    """Schema representing an incoming Telegram message."""
    message_id: int
    chat_id: int
    text: Optional[str] = None
    sender_name: Optional[str] = None


class TelegramWebhookUpdate(BaseModel):
    """Schema representing a Telegram Webhook Update payload."""
    update_id: int
    message: Optional[Dict[str, Any]] = None


class WebhookResponse(BaseModel):
    """Standard API response for webhooks and health endpoints."""
    status: str = "ok"
    message: Optional[str] = None


# --- Ocean v2.2: Weekly Digest & Second-Brain Search Schemas ---

class WeeklyVelocityReport(BaseModel):
    """Structured output schema for Sunday Weekly Velocity & Momentum Digest."""
    velocity_score: int = Field(default=85, description="Momentum/velocity score out of 100.")
    verdict: str = Field(default="High Momentum", description="Punchy verdict phrase (e.g. High Momentum, Steady Execution, Rebalancing Needed).")
    headline: str = Field(default="", description="1-sentence executive summary headline.")
    tasks_completed_count: int = Field(default=0, description="Count of completed tasks.")
    tasks_pending_count: int = Field(default=0, description="Count of carryover/pending tasks.")
    completed_highlights: List[str] = Field(default_factory=list, description="Top milestones and tasks completed this week.")
    learning_progress: List[str] = Field(default_factory=list, description="Subjects studied, papers digested, and concepts mastered.")
    leetcode_summary: Optional[str] = Field(default=None, description="LeetCode problems and algorithmic patterns practiced.")
    bottlenecks: List[str] = Field(default_factory=list, description="Carryover tasks, blocked items, or areas needing attention.")
    next_week_priorities: List[str] = Field(default_factory=list, description="Top 3 high-leverage focus areas for the upcoming week.")
    full_digest_markdown: str = Field(default="", description="Full clean formatted text for WhatsApp/Telegram delivery.")
    notion_page_url: Optional[str] = Field(default=None, description="Link to created Notion Weekly Review page.")


class SearchQueryAnalysis(BaseModel):
    """Structured output schema for Stage 2 Second-Brain Search parser."""
    query: str = Field(..., description="The core search query or question.")
    target_domain: Optional[str] = Field(default=None, description="Optional domain tag filter (e.g. AI Research, System Design).")
    time_filter: Optional[str] = Field(default=None, description="Optional relative time filter (e.g. yesterday, past week, last month).")
    search_type: Literal["QUESTION", "FIND_NOTES", "LIST_SUBJECTS", "LIST_TASKS", "FOLDER_EXPLORE", "PAGE_INSPECT", "ARCHIVE_SUGGEST"] = Field(
        default="QUESTION",
        description="Type of knowledge inquiry."
    )
    container_name: Optional[str] = Field(default=None, description="Target folder/container if exploring (e.g. Notes, Miscellaneous, Archive, YouTube).")
    page_name: Optional[str] = Field(default=None, description="Specific document/page name if inspecting content (e.g. year one budget, finances for umass fall).")


class SearchResultItem(BaseModel):
    """Represents a retrieved document or page from the Notion second brain."""
    title: str
    url: str
    category: str  # Subject, Resource, Task, Mind, Daily Log, Page, Folder
    domain_tag: Optional[str] = None
    snippet: str = ""
    last_edited_time: Optional[str] = None
    breadcrumb: Optional[str] = None


# --- Ocean v2.3: Dynamic Workspace Hierarchy & Page Explorer Schemas ---

class WorkspacePageNode(BaseModel):
    """Represents a page or container node in the dynamic Notion workspace hierarchy graph."""
    id: str
    title: str
    url: str = ""
    parent_type: str = "workspace"  # workspace, page_id, database_id, block_id
    parent_id: Optional[str] = None
    parent_title: Optional[str] = None
    breadcrumb: str = ""
    is_container: bool = False
    children_pages: List[Dict[str, str]] = Field(default_factory=list)
    last_edited_time: Optional[str] = None


class FolderExploreResult(BaseModel):
    """Result of exploring a Notion container page or folder."""
    status: str = "ok"
    container_title: str
    container_url: str = ""
    breadcrumb: str = ""
    subpages: List[Dict[str, str]] = Field(default_factory=list)  # list of {'title': ..., 'url': ..., 'id': ..., 'type': ...}
    reply_text: str = ""


class PageInspectResult(BaseModel):
    """Result of deeply inspecting and reading a Notion page's block content."""
    status: str = "ok"
    page_title: str
    page_url: str = ""
    breadcrumb: str = ""
    block_count: int = 0
    extracted_text: str = ""
    synthesis: str = ""
    reply_text: str = ""


# --- Ocean v3.0: Task Action & Document Append Schemas ---

class TaskActionType(str, Enum):
    """Supported task modification actions."""
    MARK_DONE = "MARK_DONE"
    MARK_IN_PROGRESS = "MARK_IN_PROGRESS"
    UPDATE_DUE_DATE = "UPDATE_DUE_DATE"
    DELETE_TASK = "DELETE_TASK"


class TaskActionAnalysis(BaseModel):
    """Structured output schema for TASK_ACTION module (updating, rescheduling, completing, archiving tasks)."""
    action: Literal["MARK_DONE", "MARK_IN_PROGRESS", "UPDATE_DUE_DATE", "DELETE_TASK"] = Field(
        ...,
        description="The action type to perform on the task."
    )
    task_target_title: str = Field(
        default="",
        description="Target task title or keyword to match against active tasks."
    )
    new_due_date_iso: Optional[str] = Field(
        default=None,
        description="New due date resolved in YYYY-MM-DD format (if action is UPDATE_DUE_DATE)."
    )
    new_status_name: Optional[Literal["Done", "In progress", "Not started"]] = Field(
        default=None,
        description="New status name to set."
    )
    ordinal_index: Optional[int] = Field(
        default=None,
        description="1-based ordinal index if user referenced a task by position from recent query (e.g., 1 for 'first task', 2 for 'second')."
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence level of intent parsing."
    )


class DocumentAppendAnalysis(BaseModel):
    """Structured output schema for DOCUMENT_APPEND module (inserting blocks/bullets into existing notes)."""
    target_document_title: str = Field(
        ...,
        description="Title of the target Notion document or container page (e.g. 'Ideas for projects', 'Year 1 Budget', 'Finances')."
    )
    content_to_append: str = Field(
        ...,
        description="Text content, bullet point, idea, or to-do item to append into the document."
    )
    block_type: Literal["bulleted_list_item", "to_do", "paragraph", "callout"] = Field(
        default="bulleted_list_item",
        description="Type of block to append: bulleted_list_item, to_do, paragraph, or callout."
    )


# --- Ocean v3.1: Memory Governance & Batch Action Schemas ---

class MemoryGovernanceAnalysis(BaseModel):
    """Structured output schema for MEMORY_CONTROL module (forget, update memory, inspect memory)."""
    command: Literal["FORGET", "UPDATE_STATUS", "INSPECT_MEMORY"] = Field(
        ...,
        description="Memory governance action: FORGET (soft-delete entity), UPDATE_STATUS (supersede old status), INSPECT_MEMORY (view active nodes)."
    )
    target_entity: str = Field(
        default="",
        description="Target entity name or keyword to forget, update, or inspect."
    )
    new_state_summary: Optional[str] = Field(
        default=None,
        description="Updated thesis or status summary if updating memory state."
    )


class BatchTaskActionAnalysis(BaseModel):
    """Structured output schema for BATCH_TASK_ACTION module (multi-task actions across tags, priorities, or queries)."""
    action: Literal["MARK_DONE", "MARK_IN_PROGRESS", "UPDATE_DUE_DATE", "DELETE_TASK"] = Field(
        ...,
        description="The action type to perform across matching tasks."
    )
    tag_filter: Optional[str] = Field(
        default=None,
        description="Optional tag filter (e.g. 'UMass Admin', 'Leetcode', 'Finances')."
    )
    priority_filter: Optional[Literal["High", "Medium", "Low"]] = Field(
        default=None,
        description="Optional priority filter (High/Medium/Low)."
    )
    target_query: Optional[str] = Field(
        default=None,
        description="Optional title substring query matching multiple tasks."
    )
    new_due_date_iso: Optional[str] = Field(
        default=None,
        description="New due date resolved in YYYY-MM-DD format if action is UPDATE_DUE_DATE."
    )
    new_status_name: Optional[Literal["Done", "In progress", "Not started"]] = Field(
        default=None,
        description="New status name to set."
    )






