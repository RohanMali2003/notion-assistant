import json
from unittest.mock import MagicMock, patch
import pytest

from app.leetcode_service import (
    _extract_constraints_from_html,
    execute_leetcode_background_pipeline,
    fetch_latest_leethub_commit,
    fetch_leetcode_problem_details,
    generate_leetcode_review,
    parse_problem_title_and_slug,
    slugify_title,
)
from app.schemas import (
    LeetcodeCommitData,
    LeetcodeProblemDetails,
    LeetcodeReviewRequest,
    LeetcodeReviewResult,
)


# --- Unit Tests: Title & Slug Parsing ---

def test_slugify_title():
    assert slugify_title("0001-two-sum") == "two-sum"
    assert slugify_title("1. Two Sum") == "two-sum"
    assert slugify_title("Trapping Rain Water (Hard)") == "trapping-rain-water"
    assert slugify_title("3Sum Closest [Medium]") == "3sum-closest"
    assert slugify_title("Longest Substring Without Repeating Characters") == "longest-substring-without-repeating-characters"
    assert slugify_title("") == ""


def test_parse_problem_title_from_readme():
    readme = """
    <h2><a href="https://leetcode.com/problems/two-sum/">1. Two Sum</a></h2>
    <h3>Easy</h3>
    <hr>
    <p>Given an array of integers <code>nums</code> and an integer <code>target</code>...</p>
    """
    title, slug, num = parse_problem_title_and_slug(
        commit_message="Time: 45 ms | Memory: 15 MB - LeetHub",
        readme_content=readme,
        file_paths=["0001-two-sum/0001-two-sum.py", "0001-two-sum/README.md"],
    )
    assert slug == "two-sum"
    assert num == 1
    assert "Two Sum" in title


def test_parse_problem_title_from_file_paths():
    title, slug, num = parse_problem_title_and_slug(
        commit_message="Time: O(N) - LeetHub",
        readme_content=None,
        file_paths=["0042-trapping-rain-water/trapping-rain-water.py"],
    )
    assert slug == "trapping-rain-water"
    assert num == 42
    assert "Trapping Rain Water" in title


def test_parse_problem_title_from_commit_message():
    title, slug, num = parse_problem_title_and_slug(
        commit_message="Added 0200-number-of-islands [Time: 12ms]",
        readme_content=None,
        file_paths=[],
    )
    assert slug == "number-of-islands"
    assert num == 200
    assert "Number Of Islands" in title


# --- Unit Tests: GitHub Commit Fetching ---

@patch("httpx.Client")
def test_fetch_latest_leethub_commit_success(mock_client_cls, monkeypatch):
    monkeypatch.setenv("GITHUB_LEETHUB_REPO", "octocat/leetcode")
    monkeypatch.setenv("GITHUB_PAT", "fake_pat_token")

    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client

    # 1. Commits list response
    resp_commits = MagicMock()
    resp_commits.status_code = 200
    resp_commits.json.return_value = [
        {"sha": "abc123456789", "commit": {"message": "Added 0001-two-sum [Time: 12ms]"}}
    ]

    # 2. Commit detail response
    resp_detail = MagicMock()
    resp_detail.status_code = 200
    resp_detail.json.return_value = {
        "sha": "abc123456789",
        "files": [
            {
                "filename": "0001-two-sum/0001-two-sum.py",
                "raw_url": "https://raw.githubusercontent.com/octocat/leetcode/main/0001-two-sum/0001-two-sum.py",
            },
            {
                "filename": "0001-two-sum/README.md",
                "raw_url": "https://raw.githubusercontent.com/octocat/leetcode/main/0001-two-sum/README.md",
            }
        ]
    }

    # 3. Raw file responses
    resp_code = MagicMock()
    resp_code.status_code = 200
    resp_code.text = "class Solution:\n    def twoSum(self, nums, target):\n        return []"

    resp_readme = MagicMock()
    resp_readme.status_code = 200
    resp_readme.text = '<h2><a href="https://leetcode.com/problems/two-sum/">1. Two Sum</a></h2>'

    mock_client.get.side_effect = [resp_commits, resp_detail, resp_code, resp_readme]

    commit_data = fetch_latest_leethub_commit(repo="octocat/leetcode", pat="fake_pat_token")
    assert commit_data.commit_sha == "abc123456789"
    assert commit_data.problem_slug == "two-sum"
    assert "class Solution" in commit_data.code
    assert commit_data.code_file_name == "0001-two-sum.py"


def test_fetch_latest_leethub_commit_missing_repo(monkeypatch):
    monkeypatch.delenv("GITHUB_LEETHUB_REPO", raising=False)
    with patch("app.leetcode_service.settings.GITHUB_LEETHUB_REPO", ""):
        with pytest.raises(ValueError, match="GITHUB_LEETHUB_REPO is not configured"):
            fetch_latest_leethub_commit(repo=None)
        with pytest.raises(ValueError, match="GITHUB_LEETHUB_REPO is not configured"):
            fetch_latest_leethub_commit(repo="")


@patch("httpx.Client")
def test_fetch_latest_leethub_commit_http_error(mock_client_cls, monkeypatch):
    monkeypatch.setenv("GITHUB_LEETHUB_REPO", "octocat/leetcode")
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client

    resp_err = MagicMock()
    resp_err.status_code = 404
    resp_err.text = "Not Found"
    mock_client.get.return_value = resp_err

    with pytest.raises(RuntimeError, match="GitHub API returned 404"):
        fetch_latest_leethub_commit(repo="octocat/leetcode")


# --- Unit Tests: LeetCode GraphQL Client & Constraints ---

def test_extract_constraints_from_html():
    raw_html = """
    <p>Given an array of integers <code>nums</code>.</p>
    <p><strong>Constraints:</strong></p>
    <ul>
        <li><code>2 &lt;= nums.length &lt;= 10<sup>4</sup></code></li>
        <li><code>-10<sup>9</sup> &lt;= nums[i] &lt;= 10<sup>9</sup></code></li>
        <li><code>-10<sup>9</sup> &lt;= target &lt;= 10<sup>9</sup></code></li>
        <li><strong>Only one valid answer exists.</strong></li>
    </ul>
    """
    constraints = _extract_constraints_from_html(raw_html)
    assert len(constraints) == 4
    assert "2 <= nums.length <= 104" in constraints[0]
    assert "-109 <= nums[i] <= 109" in constraints[1]


@patch("httpx.Client")
def test_fetch_leetcode_problem_details_success(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {
            "question": {
                "questionId": "1",
                "questionFrontendId": "1",
                "title": "Two Sum",
                "titleSlug": "two-sum",
                "difficulty": "Easy",
                "content": "<p><strong>Constraints:</strong></p><ul><li>2 <= nums.length <= 10^4</li></ul>",
                "topicTags": [{"name": "Array"}, {"name": "Hash Table"}],
            }
        }
    }
    mock_client.post.return_value = resp

    details = fetch_leetcode_problem_details("two-sum")
    assert details is not None
    assert details.title == "Two Sum"
    assert details.difficulty == "Easy"
    assert details.constraints == ["2 <= nums.length <= 10^4"]
    assert details.topic_tags == ["Array", "Hash Table"]


@patch("httpx.Client")
def test_fetch_leetcode_problem_details_not_found(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": {"question": None}}
    mock_client.post.return_value = resp

    details = fetch_leetcode_problem_details("non-existent-problem-slug-xyz")
    assert details is None


@patch("httpx.Client")
def test_fetch_leetcode_problem_details_network_error(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client
    mock_client.post.side_effect = Exception("Connection timeout")

    details = fetch_leetcode_problem_details("two-sum")
    assert details is None


# --- Unit Tests: Gemini Review Generation ---

@patch("app.leetcode_service.genai.Client")
def test_generate_leetcode_review_with_constraints(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = """
    VERDICT: Correct
    TIME COMPLEXITY: O(N)
    SPACE COMPLEXITY: O(N)
    IS OPTIMAL: Yes

    ANALYSIS:
    The single-pass hash map approach is optimal given N <= 10^4. Looking up complements runs in O(1) average time.

    TARGETED TESTING QUESTIONS:
    - What happens if target is formed by adding an element to itself when no duplicate exists?
    - How does your dictionary initialization behave on negative values?
    """
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    commit = LeetcodeCommitData(
        commit_sha="12345",
        problem_title="Two Sum",
        problem_slug="two-sum",
        code="class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        for i, n in enumerate(nums):\n            if target - n in seen:\n                return [seen[target - n], i]\n            seen[n] = i",
    )
    details = LeetcodeProblemDetails(
        title="Two Sum",
        title_slug="two-sum",
        difficulty="Easy",
        constraints=["2 <= nums.length <= 10^4", "-10^9 <= nums[i] <= 10^9"],
    )

    review = generate_leetcode_review(commit, details)
    assert review.verdict == "Correct"
    assert review.time_complexity == "O(N)"
    assert review.space_complexity == "O(N)"
    assert review.is_optimal is True
    assert review.fallback_mode is False
    assert len(review.testing_questions) >= 2


@patch("app.leetcode_service.genai.Client")
def test_generate_leetcode_review_fallback_mode_without_constraints(mock_genai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = """
    VERDICT: Suboptimal
    TIME COMPLEXITY: O(N^2)
    SPACE COMPLEXITY: O(1)
    IS OPTIMAL: No

    ANALYSIS:
    Nested loop approach detected. Time complexity is quadratic.

    TARGETED TESTING QUESTIONS:
    - How will this behave on arrays containing 10^5 elements?
    - What happens if all elements in nums are equal?
    """
    mock_client.models.generate_content.return_value = mock_resp
    mock_genai_cls.return_value = mock_client

    commit = LeetcodeCommitData(
        commit_sha="12345",
        problem_title="Two Sum",
        problem_slug="two-sum",
        code="def twoSum(nums, target): pass",
    )

    # Calling with problem_details=None (fallback mode)
    review = generate_leetcode_review(commit, problem_details=None)
    assert review.fallback_mode is True
    assert review.verdict == "Suboptimal"
    assert review.is_optimal is False


# --- Unit Tests: Background Pipeline Orchestrator ---

@patch("app.leetcode_service.fetch_latest_leethub_commit")
@patch("app.leetcode_service.fetch_leetcode_problem_details")
@patch("app.leetcode_service.generate_leetcode_review")
def test_execute_leetcode_background_pipeline_full_success(
    mock_review_gen,
    mock_fetch_details,
    mock_fetch_commit,
):
    mock_fetch_commit.return_value = LeetcodeCommitData(
        commit_sha="sha987",
        problem_title="Two Sum",
        problem_slug="two-sum",
        problem_number=1,
        code="def twoSum(nums, target): return []",
    )
    mock_fetch_details.return_value = LeetcodeProblemDetails(
        title="Two Sum",
        title_slug="two-sum",
        difficulty="Easy",
        constraints=["2 <= nums.length <= 10^4"],
    )
    mock_review_gen.return_value = LeetcodeReviewResult(
        problem_title="Two Sum",
        problem_slug="two-sum",
        problem_number=1,
        difficulty="Easy",
        verdict="Correct",
        time_complexity="O(N)",
        space_complexity="O(N)",
        is_optimal=True,
        review_summary="Excellent hash map approach.",
        testing_questions=["What happens if nums contains duplicate elements?"],
        fallback_mode=False,
    )

    mock_notion = MagicMock()
    mock_notion.create_leetcode_log_row.return_value = {
        "id": "notion-page-123",
        "url": "https://notion.so/notion-page-123",
    }
    mock_wa = MagicMock()
    mock_tg = MagicMock()

    result = execute_leetcode_background_pipeline(
        leetcode_req=LeetcodeReviewRequest(problem_name="Two Sum"),
        to_phone="15551234567",
        chat_id="123456",
        notion_client=mock_notion,
        whatsapp_client=mock_wa,
        telegram_client=mock_tg,
    )

    assert result["status"] == "ok"
    assert result["problem_title"] == "Two Sum"
    assert result["verdict"] == "Correct"
    assert result["fallback_mode"] is False

    # Verify Notion write
    mock_notion.create_leetcode_log_row.assert_called_once()

    # Verify WhatsApp and Telegram follow-up calls
    mock_wa.send_message.assert_called_once()
    wa_msg = mock_wa.send_message.call_args[1]["text"]
    assert "Two Sum" in wa_msg
    assert "Correct" in wa_msg
    assert "Optimal" in wa_msg

    mock_tg.send_message.assert_called_once()
    tg_msg = mock_tg.send_message.call_args[1]["text"]
    assert "Two Sum" in tg_msg


@patch("app.leetcode_service.fetch_latest_leethub_commit")
@patch("app.leetcode_service.fetch_leetcode_problem_details")
@patch("app.leetcode_service.generate_leetcode_review")
def test_execute_leetcode_background_pipeline_fallback_warning_in_message(
    mock_review_gen,
    mock_fetch_details,
    mock_fetch_commit,
):
    mock_fetch_commit.return_value = LeetcodeCommitData(
        commit_sha="sha987",
        problem_title="Custom Problem",
        problem_slug="custom-problem",
        code="def solve(): pass",
    )
    # GraphQL returns None -> triggers fallback
    mock_fetch_details.return_value = None

    mock_review_gen.return_value = LeetcodeReviewResult(
        problem_title="Custom Problem",
        problem_slug="custom-problem",
        difficulty="Unknown",
        verdict="Correct",
        time_complexity="O(N)",
        space_complexity="O(1)",
        is_optimal=True,
        review_summary="Reviewed code standalone.",
        testing_questions=["What happens on edge case inputs?"],
        fallback_mode=True,
    )

    mock_notion = MagicMock()
    mock_notion.create_leetcode_log_row.return_value = {"id": "page-123"}
    mock_wa = MagicMock()

    result = execute_leetcode_background_pipeline(
        leetcode_req=LeetcodeReviewRequest(problem_name="Custom Problem"),
        to_phone="15551234567",
        notion_client=mock_notion,
        whatsapp_client=mock_wa,
    )

    assert result["status"] == "ok"
    assert result["fallback_mode"] is True

    # Verify fallback notice is explicitly in WhatsApp message
    mock_wa.send_message.assert_called_once()
    wa_msg = mock_wa.send_message.call_args[1]["text"]
    assert "Could not fetch constraints from LeetCode GraphQL API" in wa_msg


@patch("app.leetcode_service.fetch_latest_leethub_commit")
def test_execute_leetcode_background_pipeline_github_error(mock_fetch_commit):
    mock_fetch_commit.side_effect = RuntimeError("Repo not found")
    mock_wa = MagicMock()
    mock_tg = MagicMock()

    result = execute_leetcode_background_pipeline(
        leetcode_req=LeetcodeReviewRequest(problem_name="Two Sum"),
        to_phone="15551234567",
        chat_id="123456",
        whatsapp_client=mock_wa,
        telegram_client=mock_tg,
    )

    assert result["status"] == "error"
    assert result["stage"] == "github_fetch"

    # Verify error notification sent
    mock_wa.send_message.assert_called_once()
    assert "Failed to pull solution from GitHub" in mock_wa.send_message.call_args[1]["text"]
    mock_tg.send_message.assert_called_once()
    assert "Failed to pull solution from GitHub" in mock_tg.send_message.call_args[1]["text"]


def test_clean_math_and_markdown():
    from app.leetcode_service import clean_math_and_markdown

    raw_text = "⏱️ Time: $O(K)$, where $K$ is the number of seats. | 💾 Space: $O(K)$, stored in $unordered_map$ with $n = 10^9$. **Logic**: Tested."
    
    # WhatsApp cleaning
    wa_cleaned = clean_math_and_markdown(raw_text, for_whatsapp=True)
    assert "$O(K)$" not in wa_cleaned
    assert "O(K)" in wa_cleaned
    assert "$n = 10^9$" not in wa_cleaned
    assert "n = 10^9" in wa_cleaned
    assert "**Logic**" not in wa_cleaned
    assert "*Logic*" in wa_cleaned

    # Notion cleaning
    notion_cleaned = clean_math_and_markdown(raw_text, for_whatsapp=False)
    assert "$" not in notion_cleaned
    assert "O(K)" in notion_cleaned
    assert "**Logic**" not in notion_cleaned
    assert "Logic: Tested." in notion_cleaned
