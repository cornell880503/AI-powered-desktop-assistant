# Gemini Command Center

An agentic AI desktop assistant built with Electron + React + Python, powered by Google Gemini. This repository serves as both a working prototype and a PM-ready product strategy demonstration.

The entire prototype was designed, scoped, and built using vibe coding (Claude Code + GitHub Copilot) — no prior Electron or React experience required.

---

## Product Vision

Knowledge workers spend hours switching between Gmail, Drive, Docs, Sheets, and Calendar. Each context switch is lost momentum. Gemini Command Center eliminates that friction by embedding a single agentic interface into the OS that can understand natural language, retrieve context from local files and open documents, and execute real actions — draft, save, send, schedule — without leaving your workflow.

This is a demonstration of what becomes possible when Gemini models are linked directly to practical, workflow-level actions on the desktop.

---

## What It Does

### Core Actions

| Action | What happens |
|---|---|
| **Analyze documents** | Reads open Excel / CSV / Word / PDF, writes real pandas / matplotlib code, runs it, streams the actual output |
| **Generate charts** | Bar charts, pie charts, histograms, scatter plots — saved to Downloads automatically |
| **Draft and save** | Writes any output as a real `.docx` Word file to `~/Downloads/` with heading and bullet formatting |
| **Send email** | Drafts with editable fields, attaches a generated chart or doc, sends via SMTP (Gmail-compatible) |
| **Schedule meeting** | Natural language → `.ics` calendar event → opens in system calendar |
| **Search local files** | RAG-based semantic search across Downloads / Documents / Desktop / Sheets |
| **Delete files** | AI ranks candidate files by relevance; user confirms before deletion |
| **Generate images** | Calls Gemini image model with an auto-enhanced prompt; result shown inline |
| **Open apps** | Launches any macOS app by name or opens a browser search |
| **Undo** | Full-stack undo for file saves, deletes, and calendar events — Cmd+Z or button |

### Agentic Architecture

- **Two-stage Gemini routing**: `gemini-2.5-flash` fast router decides action type, whether a screenshot is needed, and whether RAG is needed — before the execution model runs. Cuts average latency significantly.
- **Multi-action orchestration**: the fast router outputs an ordered `actions: []` list; the execution pipeline chains them in sequence with carry-forward context (e.g., draft → save → send in one command).
- **Tool-calling layer**: 10 discrete tools mapped to real OS and network actions. Each tool has a defined schema, confirmation UI, and undo path.
- **Memory management**: conversation context persists across sessions via disk + ChromaDB vector store; new files indexed immediately after every save.
- **Evaluation pipeline**: Demo Mode surfaces the full CTX / RAG / THINK reasoning trace per turn. All pipeline logs persist across app restarts for analysis.
- **Self-healing code**: on Python execution error, stderr is fed back to Gemini which rewrites and re-runs automatically.
- **Model fallback**: primary `gemini-3-flash-preview` → `gemini-2.5-flash` on 503 overload.

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Electron + React UI                    │
│  Demo Mode (pipeline trace) │ User Mode (clean chat)   │
└────────────────────┬────────────────────────────────────┘
                     │ SSE stream
┌────────────────────▼────────────────────────────────────┐
│              FastAPI sidecar (Python 3.12)              │
│                                                         │
│  Fast Router (gemini-2.5-flash)                         │
│    → decides: action[], screenshot?, RAG?               │
│                                                         │
│  Execution Model (gemini-3-flash-preview)               │
│    → document context + RAG results + memory           │
│    → action payload with typed schema                   │
│                                                         │
│  Action Executor                                        │
│    → Python sandbox (pandas/matplotlib)                 │
│    → SMTP email sender                                  │
│    → .ics calendar writer                               │
│    → .docx file writer                                  │
│    → Gemini image model                                 │
│    → ChromaDB semantic search                           │
│    → ActionLedger (undo stack)                          │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop shell | Electron 28 + React 18 + Vite |
| Backend | FastAPI on Python 3.12 |
| AI models | Gemini `gemini-3-flash-preview` (execution), `gemini-2.5-flash` (routing + fallback) |
| Image generation | Gemini `gemini-3.1-flash-image-preview` |
| Vector search | ChromaDB + `sentence-transformers/all-MiniLM-L6-v2` |
| Document parsing | LangChain + `pypdf`, `openpyxl`, `python-docx` |
| Data analysis | `pandas`, `matplotlib` (runtime code generation) |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/cornell880503/AI-powered-desktop-assistant
cd AI-powered-desktop-assistant
npm install
```

### 2. Configure backend

```bash
cd server
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_FALLBACK_MODEL=gemini-2.5-flash

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Add documents to knowledge base (optional)

Drop PDF, TXT, CSV, Excel, or Word files into `server/local_data/`. They are indexed on startup. The agent also scans `~/Downloads/`, `~/Documents/`, and `~/Desktop/` by default.

---

## Running

```bash
# Terminal 1: Python sidecar
cd server && bash start.sh

# Terminal 2: Electron app
npm run dev
```

---

## Agent Pipeline

```
1. Detect active app (OS context)
2. Fast router → action list + screenshot flag + RAG flag
3. Extract live document text (Excel, Numbers, CSV, Word, PDF)
4. Capture screenshot only if visual grounding needed
5. Semantic search only if RAG flagged
6. Execution model → typed action payload
7. Execute action (Python code, SMTP, .ics, .docx, Gemini image, ChromaDB)
8. On Python error: stderr → Gemini repair → retry
9. For multi-action plans: chain with carry-forward context
10. Stream full pipeline trace back to UI via SSE
11. Persist conversation, pipeline logs, and RAG index to disk
```

---

## Product Strategy Docs

- [Product Strategy](docs/product-strategy.md) — vision, user segments, roadmap
- [Metrics](docs/metrics.md) — north star, guardrails, experiment design
- [System Design](docs/system-design.md) — architecture and scaling considerations
- [Experiment Plan](docs/experiment-plan.md) — A/B test design for proactive suggestions
- [Privacy Design](docs/privacy-design.md) — data handling, consent, enterprise controls

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Gemini API key from Google AI Studio |
| `GEMINI_MODEL` | Primary model (default: `gemini-3-flash-preview`) |
| `GEMINI_FALLBACK_MODEL` | Fallback on 503 (default: `gemini-2.5-flash`) |
| `GEMINI_IMAGE_MODEL` | Image model (default: `gemini-3.1-flash-image-preview`) |
| `SMTP_HOST` | SMTP server (e.g. `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port (default: `587`) |
| `SMTP_USER` | Gmail address |
| `SMTP_PASS` | Gmail app password |
| `EXTRA_DATA_DIRS` | Additional directories for RAG indexing |

---

## Project Structure

```
AI-powered-desktop-assistant/
├── docs/                      # PM strategy, metrics, experiment plan, privacy design
├── electron/                  # Electron main process
├── server/
│   ├── agent.py               # Two-stage routing, multi-action orchestration, SSE streaming
│   ├── main.py                # FastAPI endpoints, action execution, pipeline persistence
│   ├── action_ledger.py       # LIFO undo stack: snapshot/restore for file/calendar actions
│   ├── data_analytics.py      # Deterministic fast-path analytics (no codegen)
│   ├── window_context.py      # Active app detection, document extraction, browser fetch
│   ├── prompt_builder.py      # Context-aware prompt assembly
│   ├── rag/                   # ChromaDB indexer and semantic searcher
│   ├── memory_store.py        # Conversation persistence
│   └── start.sh               # Sidecar startup script
└── src/                       # React + Electron frontend
```
