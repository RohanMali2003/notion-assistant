from unittest.mock import MagicMock, patch
import pytest
from app.schemas import ReminderItem


def test_fetch_pending_reminders():
    mock_client_inst = MagicMock()
    mock_client_inst.databases.query.return_value = {
        "results": [
            {
                "id": "page-001",
                "properties": {
                    "Name": {"title": [{"plain_text": "Finish documentation"}]},
                    "Status": {"status": {"name": "Pending"}},
                    "Due Date": {"date": {"start": "2026-08-20"}}
                }
            }
        ]
    }

    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")
        reminders = client.fetch_pending_reminders()

        assert len(reminders) == 1
        assert isinstance(reminders[0], ReminderItem)
        assert reminders[0].page_id == "page-001"
        assert reminders[0].title == "Finish documentation"
        assert reminders[0].due_date == "2026-08-20"


def test_mark_reminder_notified():
    mock_client_inst = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")
        client.mark_reminder_notified("page-001")

        mock_client_inst.pages.update.assert_called_once_with(
            page_id="page-001",
            properties={
                "Status": {
                    "status": {
                        "name": "Notified"
                    }
                }
            }
        )


def test_create_task_properties_omission():
    """Assert that properties dictionary omits 'Due date' and 'Description' when not provided."""
    mock_client_inst = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")

        client.create_task(
            title="Buy Groceries",
            priority="High",
            tag="Finances",
            due_date=None,
            description=None,
        )

        mock_client_inst.pages.create.assert_called_once()
        call_kwargs = mock_client_inst.pages.create.call_args[1]
        assert call_kwargs["parent"] == {"database_id": "fake_db_id"}

        props = call_kwargs["properties"]
        assert props["Name"] == {"title": [{"text": {"content": "Buy Groceries"}}]}
        assert props["Status"] == {"status": {"name": "Not started"}}
        assert props["Priority"] == {"select": {"name": "High"}}
        assert props["Tag"] == {"select": {"name": "Finances"}}
        # Verify conditional omission
        assert "Due date" not in props
        assert "Description" not in props


def test_create_task_properties_full():
    """Assert that properties dictionary includes 'Due date' and 'Description' when provided."""
    mock_client_inst = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")

        client.create_task(
            title="Submit Report",
            priority="Medium",
            tag="Projects",
            due_date="2026-08-20",
            description="Detailed Q3 summary report",
        )

        mock_client_inst.pages.create.assert_called_once()
        call_kwargs = mock_client_inst.pages.create.call_args[1]
        props = call_kwargs["properties"]

        assert props["Name"] == {"title": [{"text": {"content": "Submit Report"}}]}
        assert props["Status"] == {"status": {"name": "Not started"}}
        assert props["Priority"] == {"select": {"name": "Medium"}}
        assert props["Tag"] == {"select": {"name": "Projects"}}
        assert props["Due date"] == {"date": {"start": "2026-08-20"}}
        assert props["Description"] == {
            "rich_text": [{"text": {"content": "Detailed Q3 summary report"}}]
        }



def test_get_pending():
    mock_client_inst = MagicMock()
    mock_client_inst.databases.query.return_value = {
        "results": [
            {
                "id": "p1",
                "properties": {
                    "Name": {"title": [{"plain_text": "Task A"}]},
                    "Due date": {"date": {"start": "2026-08-15"}},
                    "Priority": {"select": {"name": "High"}},
                    "Tag": {"select": {"name": "Work"}}
                }
            },
            {
                "id": "p2",
                "properties": {
                    "Name": {"title": [{"plain_text": "Task B"}]},
                    "Due date": None,
                    "Priority": {"select": {"name": "Low"}},
                    "Tag": {"select": {"name": "Personal"}}
                }
            }
        ]
    }
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")
        tasks = client.get_pending(limit=5)

        assert len(tasks) == 2
        assert tasks[0]["title"] == "Task A"
        assert tasks[0]["due_date"] == "2026-08-15"
        assert tasks[0]["priority"] == "High"
        assert tasks[0]["tag"] == "Work"
        assert tasks[1]["title"] == "Task B"
        assert tasks[1]["due_date"] is None

        mock_client_inst.databases.query.assert_called_once_with(
            database_id="fake_db_id",
            page_size=5,
            filter={"property": "Status", "status": {"does_not_equal": "Done"}},
            sorts=[{"property": "Due date", "direction": "ascending"}]
        )


def test_get_reminder_candidates():
    from datetime import datetime, timezone, timedelta
    two_days_later = (datetime.now(timezone.utc).date() + timedelta(days=2)).isoformat()
    five_days_later = (datetime.now(timezone.utc).date() + timedelta(days=5)).isoformat()

    mock_client_inst = MagicMock()
    mock_client_inst.databases.query.return_value = {
        "results": [
            {
                "id": "p1",
                "properties": {
                    "Name": {"title": [{"plain_text": "Due Soon"}]},
                    "Due date": {"date": {"start": two_days_later}},
                    "Status": {"status": {"name": "In progress"}}
                }
            },
            {
                "id": "p2",
                "properties": {
                    "Name": {"title": [{"plain_text": "Due Far Away"}]},
                    "Due date": {"date": {"start": five_days_later}},
                    "Status": {"status": {"name": "In progress"}}
                }
            },
            {
                "id": "p3",
                "properties": {
                    "Name": {"title": [{"plain_text": "High Priority No Date"}]},
                    "Due date": None,
                    "Priority": {"select": {"name": "High"}},
                    "Status": {"status": {"name": "Not started"}}
                }
            },
            {
                "id": "p4",
                "properties": {
                    "Name": {"title": [{"plain_text": "Low Priority No Date"}]},
                    "Due date": None,
                    "Priority": {"select": {"name": "Low"}},
                    "Status": {"status": {"name": "Not started"}}
                }
            }
        ],
        "has_more": False
    }
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")
        list_a, list_b = client.get_reminder_candidates()

        assert len(list_a) == 1
        assert list_a[0]["title"] == "Due Soon"

        assert len(list_b) == 1
        assert list_b[0]["title"] == "High Priority No Date"


def test_retry_on_429_rate_limit():
    mock_client_inst = MagicMock()

    class Fake429Exception(Exception):
        status = 429

    mock_client_inst.databases.query.side_effect = [
        Fake429Exception("Rate limited"),
        Fake429Exception("Rate limited"),
        {"results": []}
    ]
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls), patch("time.sleep") as mock_sleep:
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")
        res = client.get_pending()
        assert res == []
        assert mock_client_inst.databases.query.call_count == 3
        assert mock_sleep.call_count == 2


def test_raise_on_400_validation_error():
    mock_client_inst = MagicMock()

    class Fake400Exception(Exception):
        status = 400
        message = "body failed validation: properties.Due date.date should be defined"

    mock_client_inst.pages.create.side_effect = Fake400Exception()
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient, NotionValidationError
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")

        with pytest.raises(NotionValidationError) as exc_info:
            client.create_task(title="Bad task")

        assert "Due date" in str(exc_info.value)
        assert exc_info.value.property_name == "Due date"


def test_update_task_status_success():
    """Test update_task_status matching a task title and sending page update."""
    mock_client_inst = MagicMock()
    mock_client_inst.databases.query.return_value = {
        "results": [
            {
                "id": "page-999",
                "properties": {
                    "Name": {"title": [{"plain_text": "Pack for College"}]},
                    "Status": {"status": {"name": "Not started"}}
                }
            }
        ]
    }
    mock_client_inst.pages.update.return_value = {"id": "page-999"}
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")

        success, matched_title, page = client.update_task_status("pack for college", status_name="In progress")

        assert success is True
        assert matched_title == "Pack for College"
        assert page == {"id": "page-999"}
        mock_client_inst.pages.update.assert_called_once_with(
            page_id="page-999",
            properties={"Status": {"status": {"name": "In progress"}}}
        )


def test_update_task_status_not_found():
    """Test update_task_status returning failure tuple when no task matches query."""
    mock_client_inst = MagicMock()
    mock_client_inst.databases.query.return_value = {"results": []}
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")

        success, matched_title, page = client.update_task_status("Nonexistent task", status_name="Done")

        assert success is False
        assert matched_title == "Nonexistent task"
        assert page is None
        mock_client_inst.pages.update.assert_not_called()


def test_get_today_tasks():
    """Test get_today_tasks filtering tasks due today."""
    from datetime import datetime, timezone
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    mock_client_inst = MagicMock()
    mock_client_inst.databases.query.return_value = {
        "results": [
            {
                "id": "t1",
                "properties": {
                    "Name": {"title": [{"plain_text": "Today Task"}]},
                    "Due date": {"date": {"start": today_str}}
                }
            },
            {
                "id": "t2",
                "properties": {
                    "Name": {"title": [{"plain_text": "Future Task"}]},
                    "Due date": {"date": {"start": "2099-01-01"}}
                }
            }
        ]
    }
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(token="fake_token", database_id="fake_db_id")

        today_tasks = client.get_today_tasks()
        assert len(today_tasks) == 1
        assert today_tasks[0]["title"] == "Today Task"


# --- MIND Module Notion Client Tests ---

def test_create_mind_entry_substack():
    """Test create_mind_entry for DRAFT_SUBSTACK sets Status='Idea' and targets NOTION_SUBSTACK_ID."""
    mock_client_inst = MagicMock()
    mock_client_inst.databases.retrieve.return_value = {
        "properties": {
            "Title": {"type": "title"},
            "Status": {"type": "status"},
            "Tags": {"type": "multi_select"},
        }
    }
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(
            token="fake_token",
            tasks_db_id="tasks_db_123",
            substack_db_id="substack_db_456",
            ramblings_db_id="ramblings_db_789",
            daily_logs_db_id="daily_logs_db_000",
        )

        res = client.create_mind_entry(
            entry_type="DRAFT_SUBSTACK",
            title="The Future of AI Coding",
            content="Full essay text explaining agentic coding workflows.\nSecond paragraph on developer velocity.",
            core_thesis="Agentic coding shifts the developer role from writing syntax to directing architectural intent.",
            tags=["AI", "Substack"],
        )

        mock_client_inst.pages.create.assert_called_once()
        call_kwargs = mock_client_inst.pages.create.call_args[1]
        assert call_kwargs["parent"] == {"database_id": "substack_db_456"}

        props = call_kwargs["properties"]
        assert props["Title"] == {"title": [{"text": {"content": "The Future of AI Coding"}}]}
        assert props["Status"] == {"status": {"name": "Idea"}}
        assert props["Tags"] == {"multi_select": [{"name": "AI"}, {"name": "Substack"}]}

        children = call_kwargs["children"]
        assert len(children) >= 3
        # First block is core thesis
        assert children[0]["paragraph"]["rich_text"][0]["text"]["content"] == (
            "Agentic coding shifts the developer role from writing syntax to directing architectural intent."
        )
        # Subsequent blocks are full text paragraphs
        assert "Full essay text" in children[1]["paragraph"]["rich_text"][0]["text"]["content"]
        assert "Second paragraph" in children[2]["paragraph"]["rich_text"][0]["text"]["content"]


def test_create_mind_entry_rambling():
    """Test create_mind_entry for RAMBLING writes to NOTION_RAMBLINGS_ID with child blocks."""
    mock_client_inst = MagicMock()
    mock_client_inst.databases.retrieve.return_value = {
        "properties": {
            "Name": {"type": "title"},
        }
    }
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(
            token="fake_token",
            tasks_db_id="tasks_db_123",
            substack_db_id="substack_db_456",
            ramblings_db_id="ramblings_db_789",
            daily_logs_db_id="daily_logs_db_000",
        )

        client.create_mind_entry(
            entry_type="RAMBLING",
            title="Evening Walk Thoughts",
            content="Thinking about compilers, AST transformations, and memory allocators.",
            core_thesis="AST-based rewriting can optimize declarative pipelines.",
        )

        mock_client_inst.pages.create.assert_called_once()
        call_kwargs = mock_client_inst.pages.create.call_args[1]
        assert call_kwargs["parent"] == {"database_id": "ramblings_db_789"}

        props = call_kwargs["properties"]
        assert props["Name"] == {"title": [{"text": {"content": "Evening Walk Thoughts"}}]}

        children = call_kwargs["children"]
        assert children[0]["paragraph"]["rich_text"][0]["text"]["content"] == (
            "AST-based rewriting can optimize declarative pipelines."
        )
        assert "compilers" in children[1]["paragraph"]["rich_text"][0]["text"]["content"]


def test_create_mind_entry_daily_log():
    """Test create_mind_entry for DAILY_LOG sets date property to today and writes to NOTION_DAILY_LOGS_ID."""
    from datetime import datetime, timezone
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    mock_client_inst = MagicMock()
    mock_client_inst.databases.retrieve.return_value = {
        "properties": {
            "Name": {"type": "title"},
            "Date": {"type": "date"},
        }
    }
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(
            token="fake_token",
            tasks_db_id="tasks_db_123",
            substack_db_id="substack_db_456",
            ramblings_db_id="ramblings_db_789",
            daily_logs_db_id="daily_logs_db_000",
        )

        client.create_mind_entry(
            entry_type="DAILY_LOG",
            title="Daily Reflection 2026-08-14",
            content="Completed Phase B implementation and tested all endpoints.",
            core_thesis="Steady iterative progress yields high reliability.",
        )

        mock_client_inst.pages.create.assert_called_once()
        call_kwargs = mock_client_inst.pages.create.call_args[1]
        assert call_kwargs["parent"] == {"database_id": "daily_logs_db_000"}

        props = call_kwargs["properties"]
        assert props["Name"] == {"title": [{"text": {"content": "Daily Reflection 2026-08-14"}}]}
        assert props["Date"] == {"date": {"start": today_str}}


def test_create_mind_entry_tasks_db_collision_rejected():
    """Test that create_mind_entry rejects writing into NOTION_TASKS_DB_ID."""
    mock_client_inst = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(
            token="fake_token",
            tasks_db_id="shared_db_id",
            substack_db_id="shared_db_id",  # Collision with tasks db
        )

        with pytest.raises(ValueError) as exc_info:
            client.create_mind_entry(
                entry_type="DRAFT_SUBSTACK",
                title="Collision Test",
                content="This should fail",
            )
        assert "NOTION_TASKS_DB_ID" in str(exc_info.value)


def test_create_mind_entry_unconfigured_db_raises_error():
    """Test that create_mind_entry raises ValueError if database ID is not configured."""
    mock_client_inst = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(
            token="fake_token",
            substack_db_id="",
        )

        with pytest.raises(ValueError) as exc_info:
            client.create_mind_entry(
                entry_type="DRAFT_SUBSTACK",
                title="Missing DB",
                content="This should fail",
            )
        assert "not configured" in str(exc_info.value)


def test_build_mind_blocks_large_content_chunking():
    """Test that _build_mind_blocks properly chunks paragraphs longer than 2000 characters."""
    from app.notion_client import NotionAssistantClient
    client = NotionAssistantClient(token="fake", tasks_db_id="db")

    large_text = "A" * 4500
    blocks = client._build_mind_blocks(core_thesis="Core thesis", content=large_text)

    # 1 block for thesis, plus ceil(4500/2000) = 3 blocks for content
    assert len(blocks) == 4
    assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "Core thesis"
    assert len(blocks[1]["paragraph"]["rich_text"][0]["text"]["content"]) == 2000
    assert len(blocks[2]["paragraph"]["rich_text"][0]["text"]["content"]) == 2000
    assert len(blocks[3]["paragraph"]["rich_text"][0]["text"]["content"]) == 500


# --- Learning Module Notion Client Tests ---

def test_create_subject_page_success():
    """Test creating a Subject page in NOTION_SUBJECTS_DB_ID with numbered_list_item children."""
    mock_client_inst = MagicMock()
    mock_client_inst.databases.retrieve.return_value = {
        "properties": {
            "Subject": {"type": "title"},
            "Completed tasks": {"type": "rollup"},
            "% Completed": {"type": "rollup"},
        }
    }
    mock_client_inst.pages.create.return_value = {
        "id": "subj-123",
        "url": "https://notion.so/subj-123",
    }
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(
            token="fake_token",
            subjects_db_id="subjects_db_123",
        )

        res = client.create_subject_page(
            title="Distributed Systems",
            curriculum_topics=[
                "1. Foundations of Consensus",
                "2. Raft Protocol in Depth",
            ],
        )

        assert res["id"] == "subj-123"
        mock_client_inst.pages.create.assert_called_once()
        create_kwargs = mock_client_inst.pages.create.call_args[1]

        # Verify parent DB
        assert create_kwargs["parent"] == {"database_id": "subjects_db_123"}

        # Verify title
        assert create_kwargs["properties"]["Subject"]["title"][0]["text"]["content"] == "Distributed Systems"

        # Verify children are numbered_list_item blocks
        children = create_kwargs["children"]
        assert len(children) == 2
        assert children[0]["type"] == "numbered_list_item"
        assert children[0]["numbered_list_item"]["rich_text"][0]["text"]["content"] == "1. Foundations of Consensus"
        assert children[1]["numbered_list_item"]["rich_text"][0]["text"]["content"] == "2. Raft Protocol in Depth"

        # Verify rollups are not touched
        assert "Completed tasks" not in create_kwargs["properties"]
        assert "% Completed" not in create_kwargs["properties"]


def test_create_resource_row_success():
    """Test creating a Resource row in NOTION_RESOURCES_DB_ID with relation to Subject."""
    mock_client_inst = MagicMock()
    mock_client_inst.databases.retrieve.return_value = {
        "properties": {
            "Resource Name": {"type": "title"},
            "Type": {"type": "select"},
            "URL": {"type": "url"},
            "Subjects": {"type": "relation"},
        }
    }
    mock_client_inst.pages.create.return_value = {"id": "res-999"}
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(
            token="fake_token",
            resources_db_id="resources_db_123",
        )

        res = client.create_resource_row(
            name="Raft Paper",
            url="https://raft.github.io/raft.pdf",
            resource_type="Paper",
            subject_page_id="subj-123",
        )

        assert res["id"] == "res-999"
        mock_client_inst.pages.create.assert_called_once()
        create_kwargs = mock_client_inst.pages.create.call_args[1]
        props = create_kwargs["properties"]

        assert props["Resource Name"]["title"][0]["text"]["content"] == "Raft Paper"
        assert props["Type"]["select"]["name"] == "Paper"
        assert props["URL"]["url"] == "https://raft.github.io/raft.pdf"
        assert props["Subjects"]["relation"][0]["id"] == "subj-123"


def test_create_starter_task_with_learning_tag():
    """Test creating a starter task with Tag='Learning' Literal."""
    mock_client_inst = MagicMock()
    mock_client_inst.databases.retrieve.return_value = {
        "properties": {
            "Task name": {"type": "title"},
            "Status": {"type": "status"},
            "Tags": {"type": "multi_select"},
            "Subject": {"type": "relation"},
        }
    }
    mock_client_inst.pages.create.return_value = {"id": "task-555"}
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(
            token="fake_token",
            tasks_db_id="tasks_db_123",
            subjects_db_id="subjects_db_123",
        )

        res = client.create_starter_task(
            title="Read Chapter 1",
            subject_page_id="subj-123",
        )

        assert res["id"] == "task-555"
        mock_client_inst.pages.create.assert_called_once()
        create_kwargs = mock_client_inst.pages.create.call_args[1]
        props = create_kwargs["properties"]

        assert props["Task name"]["title"][0]["text"]["content"] == "Read Chapter 1"
        assert props["Status"]["status"]["name"] == "Not started"
        assert props["Tags"]["multi_select"][0]["name"] == "Learning"
        assert props["Subject"]["relation"][0]["id"] == "subj-123"


def test_create_leetcode_log_row_success():
    """Test creating a LeetCode problem review row in NOTION_LEETCODE_LOG_DB_ID."""
    mock_client_inst = MagicMock()
    mock_client_inst.databases.retrieve.return_value = {
        "properties": {
            "Problem": {"type": "title"},
            "Difficulty": {"type": "select"},
            "Verdict": {"type": "select"},
            "Time Complexity": {"type": "rich_text"},
            "Space Complexity": {"type": "rich_text"},
            "Date": {"type": "date"},
            "URL": {"type": "url"},
            "Patterns": {"type": "multi_select"},
        }
    }
    mock_client_inst.pages.create.return_value = {
        "id": "lc-row-456",
        "url": "https://notion.so/lc-row-456",
    }
    mock_client_cls = MagicMock(return_value=mock_client_inst)

    with patch("app.notion_client.Client", mock_client_cls):
        from app.notion_client import NotionAssistantClient
        client = NotionAssistantClient(
            token="fake_token",
            leetcode_log_db_id="lc_db_123",
        )

        res = client.create_leetcode_log_row(
            problem_title="Two Sum",
            difficulty="Easy",
            verdict="Correct",
            time_complexity="O(N)",
            space_complexity="O(N)",
            is_optimal=True,
            review_text="Optimal hash map approach.",
            testing_questions=["What if nums contains duplicate elements?"],
            code="def twoSum(nums, target): pass",
            problem_url="https://leetcode.com/problems/two-sum/",
            patterns=["Array", "Hash Table"],
        )

        assert res["id"] == "lc-row-456"
        mock_client_inst.pages.create.assert_called_once()
        create_kwargs = mock_client_inst.pages.create.call_args[1]
        assert create_kwargs["parent"] == {"database_id": "lc_db_123"}
        props = create_kwargs["properties"]

        assert props["Problem"]["title"][0]["text"]["content"] == "Two Sum"
        assert props["Difficulty"]["select"]["name"] == "Easy"
        assert props["Verdict"]["select"]["name"] == "Correct"
        assert props["Time Complexity"]["rich_text"][0]["text"]["content"] == "O(N)"
        assert props["Space Complexity"]["rich_text"][0]["text"]["content"] == "O(N)"
        assert props["URL"]["url"] == "https://leetcode.com/problems/two-sum/"
        assert props["Patterns"]["multi_select"][0]["name"] == "Array"

        # Verify child blocks
        children = create_kwargs["children"]
        assert len(children) >= 4
        # 1st block is callout
        assert children[0]["type"] == "callout"
        # Includes code block
        assert any(c["type"] == "code" for c in children)





