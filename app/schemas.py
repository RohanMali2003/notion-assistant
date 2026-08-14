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


class ModuleClassification(BaseModel):
    """Stage 1: Lightweight classification schema deciding target module."""
    module: Literal["TASKS", "MIND", "LEARNING", "LEETCODE"] = Field(
        ...,
        description="The target module: TASKS, MIND, LEARNING, or LEETCODE."
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


