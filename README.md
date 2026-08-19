# 🌊 Ocean v2.0 — The Intelligent Life & Study Operating System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Gemini 2.5/3.5](https://img.shields.io/badge/Google%20Gemini-Structured%20Output-8E7CC3.svg)](https://ai.google.dev/)
[![WhatsApp Cloud API](https://img.shields.io/badge/WhatsApp-Cloud%20API-25D366.svg)](https://developers.facebook.com/)
[![Telegram Bot](https://img.shields.io/badge/Telegram-Bot%20API-2CA5E0.svg)](https://core.telegram.org/bots/api)
[![Notion API](https://img.shields.io/badge/Notion-Workspace%20OS-000000.svg)](https://developers.notion.com/)
[![Tests](https://img.shields.io/badge/tests-174%20passed-success.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *"A $0/month context-aware, multimodal second brain powered by Google Gemini, Meta WhatsApp, Telegram, and Notion."*

---

## ⚡ What is Ocean?

**Ocean** is an intelligent, context-aware AI assistant designed to run your life, research, study roadmaps, tasks, and writing directly from your favorite messaging apps (**WhatsApp** & **Telegram**). 

Instead of dealing with clunky Notion interfaces on mobile, you simply talk to Ocean in natural human language. Ocean remembers your recent chat context, routes your intent across specialized intelligence modules, researches canonical papers and documentation on the fly, and organizes everything into a structured, interconnected Notion workspace.

```
You (on WhatsApp): "Hey, I want to learn about the foundational papers for Gemini AI and Gemma models"
     ↓
Ocean (Immediate Ack): "Building your study plan..."
     ↓
Gemini AI + Live Grounding: Researches seminal ArXiv papers, technical reports, and docs
     ↓
Link Verification: Performs live HTTP health checks & infers types (Paper, Docs, Video, Article)
     ↓
Notion Workspace Auto-Population:
  ├─ 🏛️ Subjects DB: Creates "Gemini AI & Gemma Architecture" with detailed syllabus
  │    └─ In-Page Resources: Embeds direct clickable paper links & 1-sentence summaries!
  ├─ 📚 Resources DB: Logs individual verified resource rows linked to the Subject
  └─ 📋 Tasks Tracker DB: Adds starter tasks linked to the Subject for % Completed rollups
     ↓
Ocean (WhatsApp): "✅ Study Plan Ready: Gemini AI & Gemma Models (4/4 resources logged)
                   🔗 https://app.notion.com/p/Gemini-AI-Gemma-3c138af8cb5881..."
```

---

## 🌟 Superpowers & Modules

```
                               ┌────────────────────────┐
                               │   🌊 Ocean v2.0 Core   │
                               └───────────┬────────────┘
                                           │
         ┌──────────────────┬──────────────┴─────────────┬──────────────────┐
         ▼                  ▼                            ▼                  ▼
  📋 TASKS ENGINE    🏛️ LEARNING ENGINE           💭 MIND & ESSAYS   💻 LEETCODE REVIEW
  • Conversational   • Deep syllabus synthesis    • Substack drafts  • O(N) complexity
  • Priority query   • In-page clickable links    • Core thesis      • Space analysis
  • Smart pagination • Live link verification     • Brain dumps      • Edge cases & Qs
  • Multi-tagging    • Relational % rollups       • Daily logs       • Direct log links
```

### 1. 🧠 Short-Term Conversational Memory & Contextual Routing
* **Rolling Context Buffer:** Remembers recent conversation turns (per WhatsApp phone number or Telegram chat ID) with automatic 30-minute session TTL.
* **Natural Follow-ups:** Ask *"What are my high priority tasks?"* followed by *"others?"*, *"more"*, or *"next"*, and Ocean seamlessly paginates through items without repeating.
* **Anti-Rambling Guardrails:** Short follow-up questions ($\le 4$ words) are protected from being accidentally saved as philosophical essays or journal entries.

### 2. 🏛️ Learning & Research Curriculum Compiler
* **Deep Knowledge & Search Grounding:** Turn any exploratory topic (e.g. *"Mixture of Experts architecture"*, *"Distributed Consensus with Raft"*) into a comprehensive syllabus.
* **Direct In-Page Clickable Resources:** Every Subject page in Notion is built with **embedded clickable paper/documentation links, type badges (`[Paper]`, `[Docs]`, `[Video]`), and 1-sentence takeaways**.
* **3-Way Database Orchestration:** Automatically synchronizes `Subjects`, `Resources`, and `Tasks Tracker` so that checking off study tasks updates the visual `% Completed` progress bar.

### 3. 🧹 Automated Duplicate Detection & Cleanup Review Dashboard
* **Hybrid Fuzzy Similarity Engine:** Scans `Subjects`, `Tasks`, and `Resources` using `difflib.SequenceMatcher`, token sort ratio, and Jaccard containment ($\ge 70\%$).
* **Dedicated Review Page in Notion:** Publishes a clean, clickable **[Notion Cleanup & Duplicate Review](https://app.notion.com/p/Notion-Cleanup-Duplicate-Review-3c138af8cb5881598ae9cbab0f882fa2)** dashboard.
* **100% Non-Destructive:** Never deletes or alters pages automatically—gives you pairwise links, timestamps, and actionable advice to resolve duplicates with a single click.

### 4. ✍️ Mind, Substack Drafts & Philosophical Ramblings
* **Substack Mode:** Formats long-form thoughts into Substack-ready drafts with automatic title generation, core thesis extraction, and tags.
* **Ramblings & Journaling:** Captures unformatted brain dumps, reflections, and streams of consciousness into a dedicated Notion database.

### 5. 💻 LeetCode Problem Practice & Review
* **Algorithm & Complexity Evaluator:** Reviews problem practice notes, calculates time and space complexity ($O(N)$, $O(1)$), and generates targeted edge-case interview testing questions.

### 6. 🔗 Clickable Notion Deep-Links Everywhere
* Every task created, subject compiled, note logged, or query returned includes a direct `🔗 https://app.notion.com/...` deep-link so you never have to browse manually.

---

## 🏗️ Architecture & Dual-Path Engine

```
 ┌───────────────────────────┐         ┌────────────────────────────────┐
 │ WhatsApp (Meta Cloud API) │──┐      │  FastAPI Webhook Server        │
 └───────────────────────────┘  │      │  (Hosted 24/7 on Render)       │
                                ├─────▶│  • 2-Stage Gemini Routing      │──▶  Notion Workspace
 ┌───────────────────────────┐  │      │  • Conversational Memory       │     • Tasks Tracker DB
 │ Telegram Bot API          │──┘      │  • Background Task Worker      │     • Subjects DB
 └───────────────────────────┘         └────────────────────────────────┘     • Resources DB
                                                                              • Substack & Mind DB
                                                                              • LeetCode Log DB
                                       ┌────────────────────────────────┐
                                       │  GitHub Actions (Proactive)    │
                                       │  • 08:00 AM Reminder Digest    │──▶  Telegram Digest
                                       │  • Daily LeetCode Cleanup      │
                                       │  • Automated Duplicate Audit   │
                                       └────────────────────────────────┘
```

1. **Reactive Path (Real-time Messaging):**
   * WhatsApp/Telegram payloads arrive at the FastAPI server on Render.
   * Stage 1: Fast Gemini module classification (`TASKS`, `MIND`, `LEARNING`, `LEETCODE`).
   * Stage 2: Deep structured Pydantic schema extraction.
   * Long-running operations (like Grounding + Link Verification + Multi-DB writes) execute in background threadpools with immediate chat acknowledgments.
2. **Proactive Path (GitHub Actions Cron):**
   * Runs daily independent of Render to scan deadlines, clean up expired practice items, audit duplicate entries, and push morning digests.

---

## 📁 Repository Map

```
notion-assistant/
├── app/
│   ├── main.py               # FastAPI application & 2-stage Gemini router
│   ├── memory.py             # Sliding-window conversational memory & state tracker
│   ├── learning_service.py   # Grounding curriculum compiler & resource engine
│   ├── duplicate_detector.py # Fuzzy token similarity & duplicate cluster engine
│   ├── cleanup_reporter.py   # Notion Cleanup & Review dashboard generator
│   ├── notion_client.py      # Resilient Notion API client with automatic retries
│   ├── whatsapp_client.py    # Meta WhatsApp Cloud API wrapper
│   ├── telegram_client.py    # Telegram Bot API client
│   ├── schemas.py            # Pydantic models for structured output & validation
│   └── config.py             # Environment configuration & fail-fast validator
├── cron/
│   ├── check_reminders.py    # Daily morning reminder digest script
│   ├── cleanup_leetcode.py   # Automatic LeetCode practice status cleaner
│   └── find_duplicates.py    # Duplicate detection audit & reporter script
├── tests/
│   ├── test_main.py          # Webhook & endpoint integration tests
│   ├── test_memory.py        # Conversational memory & pagination tests
│   ├── test_learning.py      # Learning curriculum & resource pipeline tests
│   ├── test_duplicate_detector.py # Similarity & cleanup dashboard tests
│   ├── test_notion_client.py # Notion client unit tests
│   └── test_whatsapp_client.py # WhatsApp integration tests
├── .github/workflows/
│   ├── reminders.yml         # Daily scheduled GitHub Actions cron job
│   └── keep_alive.yml        # Render free-tier keep-alive pinger
├── requirements.txt          # Production dependencies
├── Procfile                  # Render start command
└── README.md
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Install
```bash
git clone https://github.com/RohanMali2003/notion-assistant.git
cd notion-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory:

```env
# Notion Credentials & Database IDs
NOTION_TOKEN=secret_your_notion_integration_token
NOTION_TASKS_DB_ID=your_tasks_database_id
NOTION_SUBJECTS_DB_ID=your_subjects_database_id
NOTION_RESOURCES_DB_ID=your_resources_database_id
NOTION_SUBSTACK_ID=your_substack_database_id
NOTION_RAMBLINGS_ID=your_ramblings_database_id
NOTION_DAILY_LOGS_ID=your_daily_logs_database_id
NOTION_LEETCODE_LOG_DB_ID=your_leetcode_database_id

# Google Gemini AI
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite

# WhatsApp Cloud API
WHATSAPP_TOKEN=your_meta_system_user_token
WHATSAPP_PHONE_NUMBER_ID=your_whatsapp_phone_number_id
WHATSAPP_VERIFY_TOKEN=your_custom_webhook_verify_token

# Telegram Bot API
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# App Environment
APP_ENV=production
```

### 3. Run Locally
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Run Test Suite
```bash
pytest
```
*All 174 unit tests pass with zero external network dependencies!*

---

## 🧪 Testing Duplicate Audit & Cron Jobs Locally

You can manually trigger any of Ocean's maintenance scripts anytime:

```bash
# Run duplicate audit across Notion
python cron/find_duplicates.py

# Run morning deadline check & digest
python cron/check_reminders.py

# Run LeetCode practice cleanup
python cron/cleanup_leetcode.py
```

---

## 💰 The $0/Month Stack

| Component | Provider | Tier | Cost |
| :--- | :--- | :--- | :--- |
| **Messaging** | Meta WhatsApp Cloud API | Free (1,000 conversations/month) | **$0.00** |
| **Messaging** | Telegram Bot API | Unlimited Free Tier | **$0.00** |
| **Compute / API** | Render Web Service | Free Instance Tier | **$0.00** |
| **Intelligence** | Google AI Studio (Gemini 2.5 / 3.5) | Free Tier (1M TPM / 15 RPM) | **$0.00** |
| **Database** | Notion Official API | Free Integration Tier | **$0.00** |
| **Proactive Cron** | GitHub Actions | 2,000 free workflow minutes/month | **$0.00** |
| **Total** | | | **$0.00 / month** |

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.

---

<p align="center">
  <b>Ocean v2.0</b> — Crafted with ❤️ by <a href="https://github.com/RohanMali2003">Rohan Mali</a> & powered by Google Gemini.
</p>
