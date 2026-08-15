import pytest
from pydantic import ValidationError
from app.schemas import (
    AgentAction,
    LearningRequest,
    LeetcodeReviewRequest,
    MindEntry,
    ModuleClassification,
    ModuleEnum,
    ReminderItem,
    TaskAnalysis,
    TelegramWebhookUpdate,
    WebhookResponse,
)


# --- Stage 1: ModuleClassification Tests ---

@pytest.mark.parametrize("module_val", ["TASKS", "MIND", "LEARNING", "LEETCODE"])
def test_module_classification_valid_modules(module_val):
    """Test ModuleClassification with all supported module types."""
    classification = ModuleClassification(module=module_val, raw_text="Test message")
    assert classification.module == module_val
    assert classification.raw_text == "Test message"


def test_module_classification_default_raw_text():
    """Test ModuleClassification default empty string for raw_text."""
    classification = ModuleClassification(module="TASKS")
    assert classification.module == "TASKS"
    assert classification.raw_text == ""


def test_module_classification_invalid_module():
    """Test that an invalid module value raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ModuleClassification(module="UNKNOWN_MODULE")
    assert "module" in str(exc_info.value)


def test_module_classification_missing_module():
    """Test that missing the module field raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ModuleClassification.model_validate({"raw_text": "no module specified"})
    assert "module" in str(exc_info.value)


def test_module_enum_members():
    """Test ModuleEnum string enum members."""
    assert ModuleEnum.TASKS.value == "TASKS"
    assert ModuleEnum.MIND.value == "MIND"
    assert ModuleEnum.LEARNING.value == "LEARNING"
    assert ModuleEnum.LEETCODE.value == "LEETCODE"


# --- Stage 2: MindEntry Tests ---

def test_mind_entry_full_payload():
    """Test MindEntry with all fields populated including core_thesis."""
    entry = MindEntry(
        entry_type="DRAFT_SUBSTACK",
        title="Why Simple Systems Scale",
        core_thesis="Simple systems scale because they eliminate unnecessary coordination overhead.",
        content="Long-form essay draft discussing minimalism in distributed architecture...",
        summary="A draft exploring system simplicity.",
        tags=["Architecture", "Essays", "Substack"],
    )
    assert entry.entry_type == "DRAFT_SUBSTACK"
    assert entry.sub_intent == "DRAFT_SUBSTACK"
    assert entry.title == "Why Simple Systems Scale"
    assert entry.core_thesis == "Simple systems scale because they eliminate unnecessary coordination overhead."
    assert "distributed architecture" in entry.content
    assert entry.summary == "A draft exploring system simplicity."
    assert entry.tags == ["Architecture", "Essays", "Substack"]


def test_mind_entry_defaults():
    """Test MindEntry defaults."""
    entry = MindEntry(content="Random thought during my walk.")
    assert entry.entry_type == "DAILY_LOG"
    assert entry.sub_intent == "DAILY_LOG"
    assert entry.title is None
    assert entry.core_thesis is None
    assert entry.content == "Random thought during my walk."
    assert entry.summary is None
    assert entry.tags == []


def test_mind_entry_substack_alias():
    """Test MindEntry sub_intent normalizes SUBSTACK_DRAFT to DRAFT_SUBSTACK."""
    entry = MindEntry(entry_type="SUBSTACK_DRAFT", content="Draft")
    assert entry.entry_type == "SUBSTACK_DRAFT"
    assert entry.sub_intent == "DRAFT_SUBSTACK"


@pytest.mark.parametrize("entry_type_val", ["DRAFT_SUBSTACK", "SUBSTACK_DRAFT", "RAMBLING", "DAILY_LOG"])
def test_mind_entry_valid_types(entry_type_val):
    """Test all supported MindEntry entry_types."""
    entry = MindEntry(entry_type=entry_type_val, content="Some thought")
    assert entry.entry_type == entry_type_val


def test_mind_entry_invalid_type():
    """Test that an unsupported entry_type raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        MindEntry(entry_type="TWEET_THREAD", content="Thread text")
    assert "entry_type" in str(exc_info.value)


# --- Stage 2: LearningRequest Tests ---

def test_learning_request_full_payload():
    """Test LearningRequest with full payload."""
    req = LearningRequest(
        topic="Distributed Consensus (Raft)",
        category="Computer Science",
        goal="Understand leader election and log replication invariants.",
        proficiency_level="Intermediate",
        resources_requested="Original Ongaro paper and visualization guide.",
    )
    assert req.topic == "Distributed Consensus (Raft)"
    assert req.category == "Computer Science"
    assert req.goal == "Understand leader election and log replication invariants."
    assert req.proficiency_level == "Intermediate"
    assert "Ongaro paper" in req.resources_requested


def test_learning_request_defaults():
    """Test LearningRequest minimal instantiation and defaults."""
    req = LearningRequest(topic="Rust Concurrency")
    assert req.topic == "Rust Concurrency"
    assert req.category is None
    assert req.goal is None
    assert req.proficiency_level is None
    assert req.resources_requested is None


@pytest.mark.parametrize("level", ["Beginner", "Intermediate", "Advanced"])
def test_learning_request_valid_proficiency_levels(level):
    """Test all supported proficiency levels."""
    req = LearningRequest(topic="Quantum Computing", proficiency_level=level)
    assert req.proficiency_level == level


def test_learning_request_invalid_proficiency_level():
    """Test that an invalid proficiency level raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        LearningRequest(topic="Math", proficiency_level="GodMode")
    assert "proficiency_level" in str(exc_info.value)


# --- Stage 2: LeetcodeReviewRequest Tests ---

def test_leetcode_review_request_full_payload():
    """Test LeetcodeReviewRequest with full payload."""
    req = LeetcodeReviewRequest(
        problem_name="Trapping Rain Water",
        problem_number=42,
        difficulty="Hard",
        patterns=["Two Pointers", "Monotonic Stack"],
        review_notes="Use two pointers to maintain left_max and right_max in O(1) space.",
        status="Review Needed",
    )
    assert req.problem_name == "Trapping Rain Water"
    assert req.problem_number == 42
    assert req.difficulty == "Hard"
    assert req.patterns == ["Two Pointers", "Monotonic Stack"]
    assert "two pointers" in req.review_notes
    assert req.status == "Review Needed"


def test_leetcode_review_request_defaults():
    """Test LeetcodeReviewRequest minimal instantiation and defaults."""
    req = LeetcodeReviewRequest(problem_name="Two Sum")
    assert req.problem_name == "Two Sum"
    assert req.problem_number is None
    assert req.difficulty is None
    assert req.patterns == []
    assert req.review_notes is None
    assert req.status is None


@pytest.mark.parametrize("diff", ["Easy", "Medium", "Hard"])
def test_leetcode_review_request_valid_difficulties(diff):
    """Test all supported difficulties."""
    req = LeetcodeReviewRequest(problem_name="Reverse Linked List", difficulty=diff)
    assert req.difficulty == diff


def test_leetcode_review_request_invalid_difficulty():
    """Test that an invalid difficulty raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        LeetcodeReviewRequest(problem_name="Test", difficulty="Extreme")
    assert "difficulty" in str(exc_info.value)


@pytest.mark.parametrize("status_val", ["Solved", "Review Needed", "Failed", "Mastered"])
def test_leetcode_review_request_valid_statuses(status_val):
    """Test all supported status values."""
    req = LeetcodeReviewRequest(problem_name="Test", status=status_val)
    assert req.status == status_val


def test_leetcode_review_request_invalid_status():
    """Test that an invalid status raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        LeetcodeReviewRequest(problem_name="Test", status="Unknown")
    assert "status" in str(exc_info.value)


# --- AgentAction / TaskAnalysis Tests (Preserved) ---

def test_agent_action_valid_full_payload():
    """Test instantiating AgentAction with a full set of valid fields."""
    action = AgentAction(
        intent="CREATE_TASK",
        title="Complete quarterly report",
        priority="High",
        tag="Projects",
        due_date="2026-08-25",
        description="Write and finalize Q3 summary",
        log_content=None,
    )
    assert action.intent == "CREATE_TASK"
    assert action.title == "Complete quarterly report"
    assert action.priority == "High"
    assert action.tag == "Projects"
    assert action.due_date == "2026-08-25"
    assert action.description == "Write and finalize Q3 summary"
    assert action.log_content is None


def test_agent_action_valid_minimal_payload():
    """Test instantiating AgentAction with only the required intent field."""
    action = AgentAction(intent="CREATE_TASK")
    assert action.intent == "CREATE_TASK"
    assert action.title == ""
    assert action.priority == "Medium"
    assert action.tag == "Miscellaneous"
    assert action.due_date is None
    assert action.description is None
    assert action.log_content is None


@pytest.mark.parametrize("intent_val", ["CREATE_TASK", "DAILY_LOG", "QUERY_PENDING", "UPDATE_TASK", "QUERY_TODAY"])
def test_agent_action_valid_intents(intent_val):
    """Test that all supported intents are valid."""
    action = AgentAction(intent=intent_val)
    assert action.intent == intent_val


def test_agent_action_update_task_fields():
    """Test AgentAction with UPDATE_TASK intent and target_status field."""
    action = AgentAction(
        intent="UPDATE_TASK",
        title="pack for college",
        target_status="In progress",
        new_due_date="2026-08-28",
    )
    assert action.intent == "UPDATE_TASK"
    assert action.title == "pack for college"
    assert action.target_status == "In progress"
    assert action.new_due_date == "2026-08-28"


@pytest.mark.parametrize("priority_val", ["High", "Medium", "Low"])
def test_agent_action_valid_priorities(priority_val):
    """Test that all supported priorities are valid."""
    action = AgentAction(intent="CREATE_TASK", priority=priority_val)
    assert action.priority == priority_val


@pytest.mark.parametrize("tag_val", [
    "Finances", "UMass Admin", "Writing", "Personal Site", "Substack",
    "Open Source", "Learning", "Leetcode", "Projects", "Schoolwork", "Miscellaneous"
])
def test_agent_action_valid_tags(tag_val):
    """Test that all supported tags are valid."""
    action = AgentAction(intent="CREATE_TASK", tag=tag_val)
    assert action.tag == tag_val


def test_agent_action_invalid_intent():
    """Test that an unsupported intent raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        AgentAction(intent="DELETE_TASK")
    assert "intent" in str(exc_info.value)


def test_agent_action_invalid_priority():
    """Test that an unsupported priority raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        AgentAction(intent="CREATE_TASK", priority="Urgent")
    assert "priority" in str(exc_info.value)


def test_agent_action_invalid_tag():
    """Test that an unsupported tag raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        AgentAction(intent="CREATE_TASK", tag="NonExistentTag")
    assert "tag" in str(exc_info.value)


def test_agent_action_missing_required_intent():
    """Test that omitting the required intent field raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        AgentAction.model_validate({"title": "No intent provided"})
    assert "intent" in str(exc_info.value)


def test_agent_action_invalid_field_type():
    """Test that passing an invalid type for a field raises a ValidationError."""
    with pytest.raises(ValidationError):
        AgentAction(intent="CREATE_TASK", title={"invalid": "type"})


# --- ReminderItem Tests ---

def test_reminder_item_valid():
    item = ReminderItem(
        page_id="page-123",
        title="Buy groceries",
        due_date="2026-08-15",
        status="Pending"
    )
    assert item.page_id == "page-123"
    assert item.title == "Buy groceries"
    assert item.due_date == "2026-08-15"
    assert item.status == "Pending"


def test_reminder_item_minimal():
    item = ReminderItem(
        page_id="page-456",
        title="Call mom"
    )
    assert item.page_id == "page-456"
    assert item.title == "Call mom"
    assert item.due_date is None
    assert item.status is None


# --- Webhook Response & Telegram Update Tests ---

def test_webhook_response_default():
    resp = WebhookResponse()
    assert resp.status == "ok"
    assert resp.message is None


def test_telegram_update_schema():
    update = TelegramWebhookUpdate(
        update_id=1001,
        message={"id": 1, "text": "/start", "chat": {"id": 999}}
    )
    assert update.update_id == 1001
    assert update.message["text"] == "/start"


