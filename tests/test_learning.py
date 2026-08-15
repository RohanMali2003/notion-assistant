from unittest.mock import MagicMock, patch
import pytest
import httpx
from app.learning_service import (
    compile_learning_curriculum,
    execute_learning_background_pipeline,
    infer_resource_type,
    verify_link_liveness,
)
from app.schemas import LearningPlanSynthesis, LearningRequest, VerifiedResource


# --- Link Verification & Type Inference Tests ---

@patch("httpx.Client.head")
def test_verify_link_liveness_head_success(mock_head):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_head.return_value = mock_resp

    is_valid, status, err = verify_link_liveness("https://doc.rust-lang.org/book/")
    assert is_valid is True
    assert status == 200
    assert err is None


@patch("httpx.Client.get")
@patch("httpx.Client.head")
def test_verify_link_liveness_head_blocked_get_fallback(mock_head, mock_get):
    mock_head_resp = MagicMock()
    mock_head_resp.status_code = 405
    mock_head.return_value = mock_head_resp

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get.return_value = mock_get_resp

    is_valid, status, err = verify_link_liveness("https://example.com/blocked-head")
    assert is_valid is True
    assert status == 200
    mock_head.assert_called_once()
    mock_get.assert_called_once()


@patch("httpx.Client.get")
@patch("httpx.Client.head")
def test_verify_link_liveness_dropped_404(mock_head, mock_get):
    mock_head_resp = MagicMock()
    mock_head_resp.status_code = 404
    mock_head.return_value = mock_head_resp

    is_valid, status, err = verify_link_liveness("https://example.com/not-found")
    assert is_valid is False
    assert status == 404


def test_verify_link_liveness_invalid_scheme():
    is_valid, status, err = verify_link_liveness("ftp://invalid.com/file")
    assert is_valid is False
    assert status is None


def test_infer_resource_type():
    assert infer_resource_type("https://www.youtube.com/watch?v=123") == "Video"
    assert infer_resource_type("https://youtu.be/abc") == "Video"
    assert infer_resource_type("https://arxiv.org/abs/2301.00001") == "Paper"
    assert infer_resource_type("https://example.com/research_paper.pdf") == "Paper"
    assert infer_resource_type("https://doc.rust-lang.org/book/") == "Docs"
    assert infer_resource_type("https://fastapi.tiangolo.com/tutorial/") == "Docs"
    assert infer_resource_type("https://martinfowler.com/articles/patterns-of-distributed-systems/") == "Article"
    assert infer_resource_type("https://medium.com/@author/learning-rust") == "Article"


# --- Curriculum Compilation Tests ---

@patch("app.learning_service.get_gemini_client")
def test_compile_learning_curriculum_success(mock_get_client):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = """
SUBJECT TITLE: Rust Memory Safety and Borrowing

CURRICULUM TOPICS:
1. Understanding stack and heap memory
2. Ownership rules and move semantics
3. References and the Borrow Checker
4. Mutable vs immutable borrowing
5. Slices and string views
6. Lifetimes basics and annotations

STARTER TASKS:
1. Install Rust via rustup and verify cargo installation
2. Complete Chapter 4 Ownership in The Rust Book
"""
    # Simulate grounding chunk
    mock_chunk = MagicMock()
    mock_chunk.web.uri = "https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html"
    mock_chunk.web.title = "The Rust Programming Language - Ownership"
    mock_resp.candidates = [MagicMock()]
    mock_resp.candidates[0].grounding_metadata.grounding_chunks = [mock_chunk]

    mock_client.models.generate_content.return_value = mock_resp
    mock_get_client.return_value = mock_client

    req = LearningRequest(topic="Rust Ownership and Lifetimes")
    synthesis = compile_learning_curriculum(req)

    assert synthesis.subject_title == "Rust Memory Safety and Borrowing"
    assert len(synthesis.curriculum_topics) == 6
    assert synthesis.curriculum_topics[0] == "Understanding stack and heap memory"
    assert len(synthesis.starter_tasks) == 2
    assert "Install Rust" in synthesis.starter_tasks[0]
    assert len(synthesis.surfaced_resources) >= 1
    assert synthesis.surfaced_resources[0]["url"] == "https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html"


# --- Background Pipeline Execution Tests ---

@patch("app.learning_service.verify_link_liveness")
@patch("app.learning_service.compile_learning_curriculum")
def test_execute_learning_background_pipeline_all_success(mock_compile, mock_verify):
    mock_compile.return_value = LearningPlanSynthesis(
        subject_title="Distributed Systems Foundations",
        curriculum_topics=[
            "System Models and Failure Modes",
            "RPC and Communication",
            "Consensus and Raft Algorithm",
        ],
        starter_tasks=[
            "Read Diego Ongaro's Raft Paper",
            "Run a local 3-node Raft simulation",
        ],
        surfaced_resources=[
            {"url": "https://raft.github.io/docs/", "title": "Raft Consensus Algorithm Docs"},
            {"url": "https://youtube.com/watch?v=v1", "title": "Raft Explained Video"},
        ],
    )
    mock_verify.return_value = (True, 200, None)

    mock_notion = MagicMock()
    mock_notion.create_subject_page.return_value = {
        "id": "subj-page-123",
        "url": "https://notion.so/subj-page-123",
    }
    mock_notion.create_resource_row.return_value = {"id": "res-1"}
    mock_notion.create_starter_task.return_value = {"id": "task-1"}

    mock_wa = MagicMock()

    req = LearningRequest(topic="Distributed Systems")
    result = execute_learning_background_pipeline(
        learning_req=req,
        to_phone="+1234567890",
        notion_client=mock_notion,
        whatsapp_client=mock_wa,
    )

    assert result["status"] == "ok"
    assert result["subject_title"] == "Distributed Systems Foundations"
    assert result["resources_logged"] == 2
    assert result["resources_failed"] == 0
    assert result["tasks_added"] == 2
    assert result["tasks_failed"] == 0

    # Verify Notion sequence:
    # 1. create_subject_page with numbered_list_item children
    mock_notion.create_subject_page.assert_called_once_with(
        title="Distributed Systems Foundations",
        curriculum_topics=[
            "System Models and Failure Modes",
            "RPC and Communication",
            "Consensus and Raft Algorithm",
        ],
    )
    # 2. create_resource_row linked to subject page id
    assert mock_notion.create_resource_row.call_count == 2
    mock_notion.create_resource_row.assert_any_call(
        name="Raft Consensus Algorithm Docs",
        url="https://raft.github.io/docs/",
        resource_type="Docs",
        subject_page_id="subj-page-123",
    )
    mock_notion.create_resource_row.assert_any_call(
        name="Raft Explained Video",
        url="https://youtube.com/watch?v=v1",
        resource_type="Video",
        subject_page_id="subj-page-123",
    )
    # 3. create_starter_task with subject relation
    assert mock_notion.create_starter_task.call_count == 2
    mock_notion.create_starter_task.assert_any_call(
        title="Read Diego Ongaro's Raft Paper",
        subject_page_id="subj-page-123",
    )

    # 4. WhatsApp completion notification
    mock_wa.send_message.assert_called_once()
    sent_text = mock_wa.send_message.call_args[1]["text"]
    assert "https://notion.so/subj-page-123" in sent_text
    assert "2/2 resources logged" in sent_text
    assert "2 starter tasks added" in sent_text


@patch("app.learning_service.verify_link_liveness")
@patch("app.learning_service.compile_learning_curriculum")
def test_execute_learning_background_pipeline_partial_failure(mock_compile, mock_verify):
    """Test that a failed resource write or task write does not abort the batch

    and that the completion message explicitly reports partial failures (e.g. 4/5 logged, 1 failed).
    """
    mock_compile.return_value = LearningPlanSynthesis(
        subject_title="Advanced Rust",
        curriculum_topics=["Topic 1", "Topic 2"],
        starter_tasks=["Task 1", "Task 2"],
        surfaced_resources=[
            {"url": "https://rust-lang.org/1", "title": "Resource 1"},
            {"url": "https://rust-lang.org/2", "title": "Resource 2"},
            {"url": "https://rust-lang.org/3", "title": "Resource 3"},
            {"url": "https://rust-lang.org/4", "title": "Resource 4"},
            {"url": "https://rust-lang.org/5", "title": "Resource 5"},
        ],
    )
    mock_verify.return_value = (True, 200, None)

    mock_notion = MagicMock()
    mock_notion.create_subject_page.return_value = {
        "id": "subj-page-456",
        "url": "https://notion.so/subj-page-456",
    }
    # Simulate 1 resource failing due to rate limiting (429) and others succeeding
    mock_notion.create_resource_row.side_effect = [
        {"id": "res-1"},
        Exception("429 rate_limited Notion API"),
        {"id": "res-3"},
        {"id": "res-4"},
        {"id": "res-5"},
    ]
    # Starter tasks: 2 tasks succeed
    mock_notion.create_starter_task.side_effect = [
        {"id": "task-1"},
        {"id": "task-2"},
    ]

    mock_wa = MagicMock()

    req = LearningRequest(topic="Advanced Rust")
    result = execute_learning_background_pipeline(
        learning_req=req,
        to_phone="1234567890",
        notion_client=mock_notion,
        whatsapp_client=mock_wa,
    )

    assert result["resources_logged"] == 4
    assert result["resources_failed"] == 1
    assert result["tasks_added"] == 2

    # Check that the summary message reflects partial failure
    mock_wa.send_message.assert_called_once()
    sent_text = mock_wa.send_message.call_args[1]["text"]
    assert "4/5 resources logged" in sent_text
    assert "1 failed (rate limited)" in sent_text
    assert "2 starter tasks added" in sent_text


@patch("app.learning_service.verify_link_liveness")
@patch("app.learning_service.compile_learning_curriculum")
def test_execute_learning_background_pipeline_dropped_invalid_links(mock_compile, mock_verify):
    """Test that links failing live liveness check are dropped and not written to Notion."""
    mock_compile.return_value = LearningPlanSynthesis(
        subject_title="Distributed Systems",
        curriculum_topics=["Topic 1"],
        starter_tasks=["Task 1"],
        surfaced_resources=[
            {"url": "https://deadlink.com/broken", "title": "Dead Link"},
            {"url": "https://valid.com/guide", "title": "Valid Link"},
        ],
    )
    # First link is 404, second is 200
    mock_verify.side_effect = [
        (False, 404, "HTTP 404"),
        (True, 200, None),
    ]

    mock_notion = MagicMock()
    mock_notion.create_subject_page.return_value = {
        "id": "subj-789",
        "url": "https://notion.so/subj-789",
    }
    mock_notion.create_resource_row.return_value = {"id": "res-1"}
    mock_notion.create_starter_task.return_value = {"id": "task-1"}

    mock_wa = MagicMock()

    req = LearningRequest(topic="Distributed Systems")
    result = execute_learning_background_pipeline(
        learning_req=req,
        to_phone="1234567890",
        notion_client=mock_notion,
        whatsapp_client=mock_wa,
    )

    # Only 1 valid resource should be written to Notion
    assert mock_notion.create_resource_row.call_count == 1
    assert result["resources_logged"] == 1
    assert result["resources_failed"] == 0
