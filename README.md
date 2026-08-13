# Notion Personal Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Gemini AI](https://img.shields.io/badge/Gemini%20AI-Structured%20Output-8E7CC3.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **$0/month personal assistant** that transforms Telegram messages into structured Notion tasks, journal entries, and status queries — plus a daily reminder digest for anything due soon or missing a due date.

> *Turn informal voice notes or text messages into organized, tagged, and scheduled Notion database entries powered by Google Gemini AI.*

---

## 📸 Demo Preview

```
Telegram User: "Remind me to submit the UMass admin report by tomorrow 5pm #Schoolwork"
     ↓
Bot (via Gemini AI): Parses task title, due date, priority, and tag
     ↓
Notion Database: Adds new task "Submit the UMass admin report" (Due: Tomorrow 5 PM, Tag: Schoolwork, Status: Not started)
     ↓
Telegram Bot Reply: "✅ Task created: Submit the UMass admin report (Due: 2026-08-14 17:00)"
```

---

## 🌟 Key Features

- 💬 **Natural Language Task Creation**: Send natural messages (typed or voice-to-text) to Telegram. Google Gemini parses task details using structured output models.
- 📋 **Notion Task Tracker Sync**: Automatically creates, updates (`In progress`, `Done`), and queries database tasks in your Notion workspace.
- 📓 **Journal & Quick Logging**: Automatically logs daily entries, thoughts, or progress updates directly into Notion.
- ⏰ **Proactive Daily Digest**: A scheduled GitHub Actions job checks Notion every morning for upcoming deadlines or high-priority tasks missing a due date and sends a concise Telegram digest.
- ⚡ **Dual-Path Architecture**: High-speed reactive webhook server on Render for instant messaging + completely decoupled proactive GitHub Actions cron job for reminders.
- 💰 **100% Free Tier Stack**: Leverages Render (Free Web Service), GitHub Actions, Google AI Studio (Gemini Free Tier), and Notion API.
- 🧪 **Fully Tested**: Extensive unit test suite mocking Notion, Telegram, and Gemini APIs.

---

## 🏗️ Architecture

```
                     ┌────────────────────────────┐
   Telegram (you)  ─▶│  FastAPI webhook (Render)  │─▶  Gemini (intent parsing)
                     └────────────┬───────────────┘
                                  │
                                  ▼
                        Notion Tasks Tracker DB
                                  ▲
                                  │
                     ┌────────────┴───────────────┐
                     │ GitHub Actions (daily cron) │─▶  Telegram (reminder digest)
                     └─────────────────────────────┘
```

The system operates via two independent paths:
- **Reactive** — You message the bot on Telegram. The FastAPI server on Render receives the webhook update, parses intent (`CREATE_TASK`, `DAILY_LOG`, `QUERY_PENDING`, `QUERY_TODAY`, `UPDATE_TASK`) via Gemini AI structured output, and updates your Notion database.
- **Proactive** — A scheduled GitHub Actions job runs `cron/check_reminders.py` daily to search Notion for tasks due soon or high-priority tasks with no due date, pushing a digest directly to Telegram. Has zero Render dependency.

---

## 📁 Repository Structure

```
notion-assistant/
├── app/
│   ├── main.py            # FastAPI webhook app & Gemini intent processing (reactive path)
│   ├── notion_client.py   # Notion API wrapper with error extraction & automatic retry logic
│   ├── telegram_client.py # Telegram Bot API wrapper for sending messages
│   ├── schemas.py         # Pydantic models for Gemini structured output & API payloads
│   └── config.py          # Environment variable loading & startup fail-fast validation
├── cron/
│   └── check_reminders.py # Standalone reminder script (proactive path)
├── tests/
│   ├── conftest.py        # Pytest fixtures and environment setup
│   ├── test_main.py       # FastAPI endpoint & webhook tests
│   ├── test_notion_client.py # Notion client unit tests
│   ├── test_schemas.py    # Pydantic schema validation tests
│   ├── test_config.py     # Configuration & env var tests
│   └── test_check_reminders.py # Daily reminder script unit tests
├── .github/workflows/
│   └── reminders.yml      # Daily scheduled GitHub Actions workflow
├── requirements.txt       # Production dependencies
├── .env.example           # Local environment variable template
├── Procfile               # Render deployment start command
└── README.md
```

---

## 🗄️ Notion Database Setup

Create a database in Notion (e.g. named **Tasks Tracker**) and ensure it includes the following properties:

| Property Name | Notion Type | Options / Values |
|---|---|---|
| **Name** (or **Task name**) | Title | Task title |
| **Status** | Status | `Not started`, `In progress`, `Done` |
| **Priority** | Select | `High`, `Medium`, `Low` |
| **Tag** (or **Tags**) | Select / Multi-Select | `Finances`, `UMass Admin`, `Writing`, `Personal Site`, `Substack`, `Open Source`, `Learning`, `Leetcode`, `Projects`, `Schoolwork`, `Miscellaneous` |
| **Due date** (or **Due Date**) | Date | Task deadline |
| **Description** | Rich Text | Task details or journal content |

> 🔑 **Important**: Remember to share/invite your Notion Internal Integration to your database so the assistant has read/write permissions.

---

## 🔑 Environment Variables

The application relies on the following environment variables (defined in `.env.example`):

| Variable | Used By | Required | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | app, cron | Yes | Bot token generated from `@BotFather` |
| `TELEGRAM_CHAT_ID` | app, cron | Yes | Your personal Telegram Chat ID |
| `NOTION_API_KEY` / `NOTION_TOKEN` | app, cron | Yes | Notion Internal Integration Secret |
| `NOTION_TASKS_DB_ID` / `NOTION_DATABASE_ID` | app, cron | Yes | 32-character ID from your Notion database URL |
| `GEMINI_API_KEY` | app | Yes | Google AI Studio API key |
| `TELEGRAM_WEBHOOK_SECRET` / `WEBHOOK_SECRET` | app | Optional | Random secret string verifying incoming Telegram webhooks |
| `APP_ENV` | app | Optional | `development` or `production` (default: `production`) |
| `PORT` | app | Optional | Web server listening port (default: `8000`) |

Copy `.env.example` to `.env` for local development:
```bash
cp .env.example .env
```
> Never commit your `.env` file — it is included in `.gitignore`.

---

## 💻 Local Setup

1. **Clone repository & activate virtual environment**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/notion-assistant.git
   cd notion-assistant
   python -m venv .venv

   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI local development server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

4. **Expose local server via ngrok**:
   In a second terminal:
   ```bash
   ngrok http 8000
   ```

5. **Point Telegram bot at your local tunnel**:
   ```bash
   curl -X POST https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook \
     -d "url=https://<your-ngrok-url>.ngrok-free.app/webhook" \
     -d "secret_token=<TELEGRAM_WEBHOOK_SECRET>"
   ```

6. **Test the assistant**:
   Send a message to your Telegram bot (e.g., *"Finish reading Chapter 3 by tomorrow #Schoolwork"*) and verify that a new page appears in your Notion database and a confirmation reply is returned to Telegram.

---

## ☁️ Deployment (Render)

1. Push this repository to GitHub and connect it as a new **Web Service** in the [Render Dashboard](https://dashboard.render.com).
2. Configure deployment parameters:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}` (or default from `Procfile`)
3. Add all environment variables in Render's dashboard (**Environment** tab).
4. Point your Telegram Bot webhook to your live Render application:
   ```bash
   curl -X POST https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook \
     -d "url=https://<your-app-name>.onrender.com/webhook" \
     -d "secret_token=<TELEGRAM_WEBHOOK_SECRET>"
   ```

---

## ⏰ Reminder Automation (GitHub Actions)

The workflow `.github/workflows/reminders.yml` runs `cron/check_reminders.py` automatically on a daily schedule (defaults to 8:00 AM IST / 2:30 AM UTC).

### Setup Instructions:
1. Navigate to **Repo → Settings → Secrets and variables → Actions**.
2. Add the required repository secrets:
   - `NOTION_API_KEY` (or `NOTION_TOKEN`)
   - `NOTION_TASKS_DB_ID` (or `NOTION_DATABASE_ID`)
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Trigger the workflow manually via the **Actions** tab (`workflow_dispatch`) to verify execution.

> ℹ️ The script outputs no message if there are no reminders due soon or high-priority items without due dates, preventing daily notification clutter.

---

## 🧪 Testing

Run unit tests locally using pytest:

```bash
python -m pytest
```

All external API interactions with Notion, Telegram, and Gemini are fully mocked, allowing offline test execution without consuming API quota.

---

## 🛠️ Troubleshooting

If the bot is silent or failing to process tasks, check this quick checklist:

1. **Webhook 401 Unauthorized**: Ensure `TELEGRAM_WEBHOOK_SECRET` matches the `secret_token` passed in the `setWebhook` curl command.
2. **Render Cold Starts**: Render's free tier spins down after inactivity. The first message after a delay may take a few seconds to process while the container wakes up.
3. **Notion Schema Mismatch**: Verify that Notion database property names match expected names (`Name`/`Task name`, `Status`, `Priority`, `Tag`/`Tags`, `Due date`/`Due Date`, `Description`).
4. **Integration Permissions**: Ensure the Notion integration token has been explicitly shared with the target database.
5. **Check Logs**:
   - Check Render service logs for the webhook path.
   - Check GitHub Actions run logs for the daily reminder cron path.

---

## 🚀 Roadmap (v2 & Future Scope)

- 💬 **WhatsApp Integration (v2 Flagship)**: Native WhatsApp Business API & Twilio gateway support — send tasks, journal notes, and query status directly from WhatsApp!
- 🎙️ **Voice Memo Support**: Direct audio transcription of voice notes using Gemini Multimodal Audio models.
- 🔔 **Notion Webhook Integration**: Bi-directional sync when tasks are marked complete directly inside Notion.
- 📊 **Weekly Productivity Digest**: Summary report sent every Sunday reviewing completed tasks and habits.
- 📅 **Google Calendar Sync**: Two-way synchronization of Notion task due dates with Google Calendar.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - open source and available for personal or commercial use.
