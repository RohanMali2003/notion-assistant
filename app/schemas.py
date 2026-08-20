from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# --- Stage 1: Module Classification Schema ---

class ModuleEnum(str, Enum):
    """Supported module categories for Stage 1 classification."""
    TASKS = "TASKS"
    MIND = "MIND"
    LEARNING = "LEARNING"
    LEETCODE = "LEETCODE"
    SEARCH = "SEARCH"
    DIGEST = "DIGEST"


class ModuleClassification(BaseModel):
    """Stage 1: Lightweight classification schema deciding target module."""
    module: Literal["TASKS", "MIND", "LEARNING", "LEETCODE", "SEARCH", "DIGEST"] = Field(
        ...,
        description="The target module: TASKS, MIND, LEARNING, LEETCODE, SEARCH, or DIGEST."
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
    search_type: Literal["QUESTION", "FIND_NOTES", "LIST_SUBJECTS", "LIST_TASKS"] = Field(
        default="QUESTION",
        description="Type of knowledge inquiry."
    )


class SearchResultItem(BaseModel):
    """Represents a retrieved document or page from the Notion second brain."""
    title: str
    url: str
    category: str  # Subject, Resource, Task, Mind, Daily Log
    domain_tag: Optional[str] = None
    snippet: str = ""
    last_edited_time: Optional[str] = None



