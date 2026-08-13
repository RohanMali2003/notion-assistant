from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class TaskAnalysis(BaseModel):
    """Structured output schema for Gemini intent classification and task details."""
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


# Alias for AgentAction schema
AgentAction = TaskAnalysis



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

