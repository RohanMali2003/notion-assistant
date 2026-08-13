# Notion Assistant

A Python application for managing Notion database tasks and sending notifications via Telegram Bot API.

## Project Structure

```
notion-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI webhook app
│   ├── notion_client.py   # thin wrapper around notion-client calls
│   ├── telegram_client.py # thin wrapper around Telegram Bot API calls
│   ├── schemas.py         # pydantic models
│   └── config.py          # reads env vars, fails fast if any are missing
├── cron/
│   └── check_reminders.py # standalone script, no FastAPI dependency
├── tests/
│   ├── test_notion_client.py
│   └── test_schemas.py
├── .github/workflows/
│   └── reminders.yml
├── requirements.txt
├── .env.example
├── .gitignore              # excludes .env, __pycache__, .venv
├── Procfile
└── README.md
```

## Features

- **FastAPI Webhook Service**: Endpoint `/webhook/telegram` to handle incoming messages from Telegram.
- **Fail-Fast Configuration**: Validates environment variables at startup and lists all missing required keys.
- **Local Dev Support**: Uses `python-dotenv` only when a local `.env` file exists.
- **Standalone Cron Execution**: `cron/check_reminders.py` runs independently without loading FastAPI.
- **GitHub Actions Scheduled Workflow**: Automatically executes reminder checks on a 15-minute schedule.

## Environment Variables

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `NOTION_TOKEN` | Yes | Notion integration secret token |
| `NOTION_DATABASE_ID` | Yes | Target Notion database ID |
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram bot API token |
| `TELEGRAM_CHAT_ID` | Yes | Target Telegram chat ID |
| `GEMINI_API_KEY` | No | Google Gemini API key (for AI intent classification) |
| `APP_ENV` | No | Environment (`development` / `production`). Defaults to `production`. |
| `PORT` | No | Server port (default: 8000) |
| `WEBHOOK_SECRET` | No | Optional secret header for webhook authentication |

> **Note:** If any required environment variable (`NOTION_TOKEN`, `NOTION_DATABASE_ID`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) is missing, `app/config.py` will raise a clear startup error listing the missing variables by name.

## Getting Started

### Local Setup

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

4. Run the standalone cron script manually:
   ```bash
   python -m cron.check_reminders
   ```

### Running Tests

Execute pytest suite:

```bash
pytest
```

## Deployment

### Render

This project includes a `Procfile` ready for Render deployment. Set your environment variables in the Render Dashboard.

### GitHub Actions

The repository includes a scheduled workflow in `.github/workflows/reminders.yml`. Configure the following repository secrets under GitHub Settings > Secrets and variables > Actions:
- `NOTION_TOKEN`
- `NOTION_DATABASE_ID`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
