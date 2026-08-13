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


