"""
Copilot Agent — orchestrates the full pipeline:
  1. Fetch active window context
  2. Semantic search over local docs
  3. Stream step-by-step thoughts via SSE
  4. Call Gemini Flash and emit structured result
  5. If GENERATE_IMAGE: auto-augment prompt + call Gemini image model → base64

SSE event shape:
  { "step": "context"|"search"|"think"|"result"|"image"|"action_card"|"error", ... }
"""

import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

from google import genai
from google.genai import types

from action_ledger import ledger
from context_provider import ContextProvider
from data_analytics import run_deterministic_analysis, build_spreadsheet_report
from memory_store import MemoryStore
from privacy_mode import ENHANCED_MODE, load_privacy_mode
from prompt_builder import PromptBuilder
from query_normalizer import normalize_query
from rag.searcher import search_docs
from suggestion_engine import SuggestionEngine
from window_context import get_active_window_title, capture_screen_base64, get_active_document_content, get_browser_page_content, get_last_capture_mode

logger = logging.getLogger(__name__)

DEFAULT_MODEL        = "gemini-3-flash-preview"
FALLBACK_MODEL       = "gemini-2.5-flash"   # fallback when primary is overloaded
ROUTER_MODEL         = "gemini-2.5-flash"
ROUTER_FALLBACK_MODEL = "gemini-3-flash-preview"
IMAGE_MODEL          = "gemini-3.1-flash-image-preview"
IMAGE_MODEL_FALLBACK = "gemini-2.0-flash-exp-image-generation"

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are Copilot, an intelligent desktop AI assistant embedded in the user's workflow.

You will receive:
- The user's currently active application window title
- A screenshot of the user's screen (when available) — you CAN see what is on their screen
- The user's query
- Optionally: relevant excerpts from their local knowledge base

Your job is to analyze this context and choose the most useful action.

You MUST respond with ONLY a single valid JSON object — no markdown, no explanation, no code fences:
{
  "thought": "<1–2 sentence reasoning about what the user needs and why you chose this action>",
  "action": "<exactly one of: SEARCH_LOCAL_DOCS | DRAFT_CONTENT | GENERATE_IMAGE | SEND_EMAIL | SAVE_FILE | DELETE_FILE | SCHEDULE_MEETING | OPEN_APP | EXECUTE_PYTHON | UNDO_ACTION>",
  "payload": "<for SEND_EMAIL/SAVE_FILE/SCHEDULE_MEETING/OPEN_APP: a nested JSON object with the required fields; for all others: a plain string>"
}

Action selection rules:
- SEARCH_LOCAL_DOCS  → user wants to find, locate, or browse files/documents on their computer — always use this when user asks to "find", "search for", "look for", or "show me" a file
- DRAFT_CONTENT      → user wants text written, summarized, explained, or analyzed
- GENERATE_IMAGE     → user explicitly asks to create, describe, or visualize an image
- SEND_EMAIL         → user wants to compose and send an email to someone
- SAVE_FILE          → user wants to save content to a file on their computer
- SCHEDULE_MEETING   → user wants to create a calendar event or schedule a meeting
- OPEN_APP           → user explicitly asks to LAUNCH a specific application by name (e.g. "open Spotify", "open Chrome"); NOT for finding files
- DELETE_FILE        → user wants to delete or remove a local file by name (e.g. "delete X", "刪掉X", "remove X")
- UNDO_ACTION        → user wants to undo, revert, or reverse the last action (e.g. "undo that", "revert", "cancel what you did", "撤銷", "復原", "取消剛才")
- EXECUTE_PYTHON     → user wants to analyze, count, calculate, or process data from the active document using Python — write and run actual code (use when user says "用python", "analyze", "calculate", "count", "分析", "計算")

IMPORTANT: Never use OPEN_APP to open a file — use SEARCH_LOCAL_DOCS to find it first, then the user will choose to open it themselves.
IMPORTANT: When user asks to analyze/calculate data from a file (e.g. count ratios, statistics), always use EXECUTE_PYTHON — NOT DRAFT_CONTENT.

For EXECUTE_PYTHON, set payload to exactly the string "GENERATE_CODE" — the system will handle code generation separately.

For SEND_EMAIL, set payload to a nested JSON object (NOT a string):
{"to":"...","subject":"...","body":"...","attachment_path":null}
CRITICAL rules for the body field:
- Write as the sender (user), addressed TO the recipient — a complete professional email ready to send
- NEVER mention file paths, system paths, or technical details in the body
- NEVER ask for clarification or write meta-commentary — just write the email
- attachment_path must always be null (the system handles attachments automatically)

For SAVE_FILE, set payload to a nested JSON object (NOT a string):
{"filename":"...","content":"..."}
- Default to .docx for document summaries, notes, and reports (only use .txt for raw data or logs)
- filename should be descriptive and in the user's language

For DELETE_FILE, set payload to a nested JSON object (NOT a string):
{"filename":"..."}
- filename is the exact name of the file the user wants to delete (search in ~/Downloads, ~/Documents, ~/Desktop)

For SCHEDULE_MEETING, set payload to a nested JSON object (NOT a string):
{"title":"...","attendees":"...","date":"YYYY-MM-DD","time":"HH:MM","duration_minutes":60,"location":"..."}
- date must be a real future date in YYYY-MM-DD format
- time must be 24-hour HH:MM format
- attendees is a comma-separated list of names or email addresses

For OPEN_APP, set payload to a nested JSON object (NOT a string):
{"app":"...","action":"..."}
- app is the exact macOS application name (e.g. "Google Chrome", "Spotify", "Numbers")
- action is what to do after opening (e.g. "search for Microsoft Copilot news", or "" if just opening)

Tailor your tone to the active application context.

If the user's request asks about their next meeting, meeting time, meeting schedule, or upcoming calendar event,
and calendar context is available, answer with DRAFT_CONTENT using the calendar context directly.
Do NOT use SEARCH_LOCAL_DOCS for calendar questions.
"""

IMAGE_AUGMENT_PROMPT = """\
The user is currently working in: {active_window}
Original image request: {original_prompt}

Create an enhanced, detailed image generation prompt that:
1. Incorporates the context of the active application
2. Adds artistic style, lighting, and composition details
3. Is optimized for AI image generation

Respond with ONLY the enhanced prompt text, no explanation.
"""

CALENDAR_SCREEN_EXTRACT_PROMPT = """\
You are validating a calendar screenshot for a factual schedule lookup.

You MUST extract only what is clearly visible on the screen.
Do NOT guess from partial text, icons, layout, or world knowledge.
Do NOT add holidays, festivals, or events unless they are explicitly visible.
If you cannot clearly verify both an event label and its time/date, return status="unverified".

Return ONLY a single valid JSON object:
{
  "status": "verified" | "unverified",
  "source": "screen_inferred_calendar",
  "answer": "<short answer for the user, only if verified>",
  "event_title": "<visible title or empty string>",
  "event_time": "<visible date/time text or empty string>",
  "evidence": ["<up to 3 short visible snippets from the screenshot>"]
}
"""

CALENDAR_SCREEN_WITH_OCR_HINT_PROMPT = """\
You are validating a calendar screenshot for a factual schedule lookup.

You will receive:
- a calendar screenshot
- OCR text extracted from that same screenshot

Use the OCR text as a hint for where to look, but only answer from details that are actually visible in the screenshot.
Do NOT guess from browser chrome, tabs, or unrelated UI.
Do NOT add holidays, festivals, or events unless they are explicitly visible.
Only return status="verified" when the screenshot clearly shows a matching event title and time/date.

Return ONLY a single valid JSON object:
{
  "status": "verified" | "unverified",
  "source": "screen_inferred_calendar",
  "answer": "<short answer for the user, only if verified>",
  "event_title": "<visible title or empty string>",
  "event_time": "<visible date/time text or empty string>",
  "evidence": ["<up to 3 short visible snippets from the screenshot>"]
}
"""

CALENDAR_OCR_EXTRACT_PROMPT = """\
You are validating OCR text extracted from a calendar screenshot for a factual schedule lookup.

Use ONLY the OCR text provided below.
Do NOT guess missing words, dates, or holidays.
Do NOT use world knowledge.
Only return status="verified" when the OCR text clearly contains a matching event title and time/date.

Return ONLY a single valid JSON object:
{
  "status": "verified" | "unverified",
  "source": "ocr_screen_calendar",
  "answer": "<short answer for the user, only if verified>",
  "event_title": "<visible title or empty string>",
  "event_time": "<visible date/time text or empty string>",
  "evidence": ["<up to 3 exact snippets copied from OCR text>"]
}
"""

INTENT_ROUTER_PROMPT = """\
You are a fast intent router for a desktop Copilot system.

Your job is to decide:
1. which action the system should take
2. whether the task needs a screenshot
3. whether the task needs local RAG search
4. the shortest useful execution plan

Return ONLY a single valid JSON object:
{
  "actions": ["<primary action>", "<optional second action>"],
  "needs_screenshot": true,
  "needs_rag": false,
  "reason": "short reason",
  "plan": "short execution plan"
}

The "actions" field is an ordered list of actions to execute in sequence.
Use a single-item list for most requests. Use two items only when the user explicitly asks for two distinct outcomes (e.g. "summarize AND send email", "analyze AND save to file", "run python AND send results by email").

Valid action values: SEARCH_LOCAL_DOCS | DRAFT_CONTENT | GENERATE_IMAGE | SEND_EMAIL | SAVE_FILE | DELETE_FILE | SCHEDULE_MEETING | OPEN_APP | EXECUTE_PYTHON | UNDO_ACTION

Common multi-action patterns:
- "summarize + send email"  → ["DRAFT_CONTENT", "SEND_EMAIL"]
- "analyze + send results"  → ["EXECUTE_PYTHON", "SEND_EMAIL"]
- "summarize + save file"   → ["DRAFT_CONTENT", "SAVE_FILE"]
- "analyze + save file"     → ["EXECUTE_PYTHON", "SAVE_FILE"]

Routing principles:
- Use EXECUTE_PYTHON for spreadsheet/file analysis, counting, chart selection, metrics analysis, and structured data work.
- Use DRAFT_CONTENT for normal writing, explanation, summarization, and interpretation.
- Prefer model-based intent understanding over brittle keyword shortcuts.
- Set needs_screenshot=true only when the visible screen content is likely necessary to answer correctly.
- Set needs_rag=true when local knowledge-base retrieval is likely relevant.
  Always set needs_rag=true when: the user references a specific document or file by name (e.g. "re-summarize the Cohere doc", "read the report"), or the query is about content from a previously saved/indexed file.
- Prefer low-latency routing. Do not request screenshot or RAG unless they are actually useful.
"""


@dataclass
class IntentPlan:
    actions: list  # ordered list of actions to execute
    needs_screenshot: bool
    needs_rag: bool
    reason: str
    plan: str
    source: str

    @property
    def action(self) -> str:
        """Primary (first) action — preserved for backward compatibility."""
        return self.actions[0] if self.actions else "DRAFT_CONTENT"


_VALID_ACTIONS = {
    "SEARCH_LOCAL_DOCS", "DRAFT_CONTENT", "GENERATE_IMAGE", "SEND_EMAIL",
    "SAVE_FILE", "DELETE_FILE", "SCHEDULE_MEETING", "OPEN_APP",
    "EXECUTE_PYTHON", "UNDO_ACTION",
}


def _coerce_intent_plan(payload: dict, source: str) -> IntentPlan:
    # Support both new "actions" array and legacy "action" string
    raw_actions = payload.get("actions") or [payload.get("action", "DRAFT_CONTENT")]
    if isinstance(raw_actions, str):
        raw_actions = [raw_actions]
    actions = [str(a).strip() for a in raw_actions if str(a).strip() in _VALID_ACTIONS]
    if not actions:
        actions = ["DRAFT_CONTENT"]
    return IntentPlan(
        actions=actions,
        needs_screenshot=bool(payload.get("needs_screenshot")),
        needs_rag=bool(payload.get("needs_rag")),
        reason=str(payload.get("reason", "")).strip() or "No reason provided.",
        plan=str(payload.get("plan", "")).strip() or "No execution plan provided.",
        source=source,
    )


def _route_intent_fast(active_window: str, user_input: str) -> IntentPlan:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return IntentPlan(
            action="DRAFT_CONTENT",
            needs_screenshot=False,
            needs_rag=False,
            reason="No API key available, so using a minimal default route.",
            plan="Answer directly with the context already available.",
            source="fallback",
        )

    client = genai.Client(api_key=api_key)
    router_model = os.getenv("GEMINI_ROUTER_MODEL", ROUTER_MODEL)
    fallback_model = os.getenv("GEMINI_ROUTER_FALLBACK_MODEL", ROUTER_FALLBACK_MODEL)
    response, used_model = _generate_with_fallback(
        client,
        router_model,
        fallback_model,
        contents=(
            f"Active window: {active_window or 'Unknown'}\n"
            f"User request: {user_input}"
        ),
        config=types.GenerateContentConfig(system_instruction=INTENT_ROUTER_PROMPT),
    )
    raw = _clean_json(response.text)
    result = json.loads(raw)
    return _coerce_intent_plan(result, f"model:{used_model}")


_LAST_IMAGE_PATH = Path(tempfile.gettempdir()) / "copilot_last_image.png"


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _is_next_meeting_query(user_input: str) -> bool:
    query = normalize_query(user_input)
    patterns = (
        "next meeting",
        "my next meeting",
        "when is my next meeting",
        "next meeting today",
        "upcoming meeting",
        "meeting time",
        "next calendar event",
        "下個會議",
        "下一個會議",
        "我的下一個會議",
        "下個 meeting",
        "會議是幾點",
        "下一場會議",
    )
    return any(pattern in query for pattern in patterns)


def _is_calendar_factual_query(user_input: str) -> bool:
    query = normalize_query(user_input)
    return _contains_any(
        query,
        (
            "next meeting",
            "meeting today",
            "when is my next meeting",
            "what time is my meeting",
            "meeting time",
            "calendar event",
            "下個會議",
            "下一個會議",
            "會議是幾點",
            "今天的會議",
        ),
    )


def _window_looks_calendar_related(active_window: str) -> bool:
    lower_window = (active_window or "").lower()
    return _contains_any(
        lower_window,
        (
            "calendar",
            "google calendar",
            "outlook calendar",
            "lark calendar",
            "feishu calendar",
            "agenda",
            "schedule",
            "meeting",
            "日曆",
            "行事曆",
            "會議",
        ),
    )


def _is_temporal_schedule_query(user_input: str) -> bool:
    query = normalize_query(user_input)
    return _contains_any(
        query,
        (
            "what will happen",
            "what's happening",
            "what is happening",
            "what do i have",
            "what is on my calendar",
            "what's on my calendar",
            "schedule tomorrow",
            "schedule today",
            "tomorrow",
            "tmrw",
            "tmr",
            "today",
            "next",
            "upcoming",
            "later today",
            "明天",
            "今天",
            "等一下",
            "接下來",
            "待會",
        ),
    )


def _requires_verified_calendar_data(user_input: str, active_window: str) -> bool:
    return _is_calendar_factual_query(user_input) or (
        _window_looks_calendar_related(active_window) and _is_temporal_schedule_query(user_input)
    )


def _build_unverified_calendar_response() -> str:
    return (
        "I couldn't verify that from structured calendar data yet. "
        "This build only treats local macOS Calendar events as verified calendar data, "
        "and it will not answer calendar facts from screenshots alone."
    )


def _format_screen_inferred_calendar_response(extraction: dict) -> str:
    answer = str(extraction.get("answer", "")).strip()
    event_title = str(extraction.get("event_title", "")).strip()
    event_time = str(extraction.get("event_time", "")).strip()
    evidence = extraction.get("evidence") or []
    evidence = [str(item).strip() for item in evidence if str(item).strip()]

    lines = []
    if answer:
        lines.append(f"Inferred from calendar screen: {answer}")
    else:
        parts = []
        if event_title:
            parts.append(f"event \"{event_title}\"")
        if event_time:
            parts.append(f"time {event_time}")
        if parts:
            lines.append("Inferred from calendar screen: " + ", ".join(parts))
        else:
            lines.append("Inferred from calendar screen: A calendar event appears visible, but the details are limited.")

    if evidence:
        lines.append("")
        lines.append("Visible evidence:")
        for item in evidence[:3]:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _ocr_screen_text_via_vision(screen_b64: str) -> str:
    raw_bytes = base64.b64decode(screen_b64)
    image_path = Path(tempfile.mktemp(suffix=".png"))
    script_path = Path(tempfile.mktemp(suffix=".swift"))
    image_path.write_bytes(raw_bytes)
    script_path.write_text(
        """
import AppKit
import Vision
import Foundation

guard CommandLine.arguments.count > 1 else {
    fputs("missing image path\\n", stderr)
    exit(1)
}

let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard let nsImage = NSImage(contentsOf: imageURL) else {
    fputs("could not load image\\n", stderr)
    exit(2)
}

var rect = NSRect(origin: .zero, size: nsImage.size)
guard let cgImage = nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
    fputs("could not decode cgImage\\n", stderr)
    exit(3)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["en-US", "zh-Hans", "zh-Hant"]

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try handler.perform([request])

let observations = (request.results ?? []).compactMap { observation -> String? in
    guard let candidate = observation.topCandidates(1).first else { return nil }
    return candidate.string
}

print(observations.joined(separator: "\\n"))
""",
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            ["swift", str(script_path), str(image_path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return result.stdout.strip()
    finally:
        image_path.unlink(missing_ok=True)
        script_path.unlink(missing_ok=True)


def _extract_calendar_from_ocr_text(
    client: genai.Client,
    model_name: str,
    fallback_model: str,
    user_input: str,
    active_window: str,
    ocr_text: str,
) -> dict:
    response, _ = _generate_with_fallback(
        client,
        model_name,
        fallback_model,
        contents=(
            f"Active window: {active_window or 'Unknown'}\n"
            f"User request: {user_input}\n\n"
            f"OCR text from the calendar screenshot:\n{ocr_text}"
        ),
        config=types.GenerateContentConfig(system_instruction=CALENDAR_OCR_EXTRACT_PROMPT),
    )
    raw = _clean_json(response.text)
    result = json.loads(raw)
    if result.get("status") not in {"verified", "unverified"}:
        result["status"] = "unverified"
    return result


def _build_screen_inferred_calendar_prompt(normalized_user_input: str) -> str:
    return (
        "This is a factual calendar lookup and there is no structured calendar event available.\n"
        "You may inspect the screenshot ONLY if you can clearly read a visible calendar or meeting entry.\n"
        "Do NOT guess from icons, layout, or partial text.\n"
        "Do NOT infer a meeting from unrelated browser tabs or generic productivity UIs.\n"
        "If you infer an answer from the screen, the payload MUST begin with 'Inferred from screen context:'\n"
        "If the screen is ambiguous, say you cannot verify the next meeting.\n"
        "Confidence threshold: only answer when the meeting title and time are both clearly visible.\n\n"
        f"User request:\n{normalized_user_input}"
    )


def _format_inferred_calendar_payload(payload: str) -> str:
    text = (payload or "").strip()
    if not text:
        return "Inferred from screen context: I couldn't confidently read the next meeting from the screen."
    if text.lower().startswith("inferred from screen context:"):
        return text
    return f"Inferred from screen context: {text}"


def _screen_calendar_fallback_allowed(active_window: str) -> bool:
    lower_window = (active_window or "").lower()
    calendar_window_hints = (
        "calendar",
        "google calendar",
        "outlook calendar",
        "lark calendar",
        "feishu calendar",
        "schedule",
        "agenda",
        "meeting",
        "行事曆",
        "日曆",
        "會議",
    )
    return any(token in lower_window for token in calendar_window_hints)


def _structured_calendar_items(context: dict) -> list[dict]:
    return [
        item for item in (context.get("calendar", []) or [])
        if item.get("source") == "local_calendar"
    ]


def _looks_like_lark_calendar_surface(active_window: str, screen_b64: str | None) -> bool:
    lower_window = (active_window or "").lower()
    if any(token in lower_window for token in ("lark", "feishu", "calendar", "agenda", "schedule", "meeting")):
        return True
    return bool(screen_b64)


def _ocr_text_looks_calendar_related(ocr_text: str) -> bool:
    text = normalize_query(ocr_text)
    positive = (
        "calendar",
        "agenda",
        "meeting",
        "schedule",
        "today",
        "tomorrow",
        "am",
        "pm",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "日曆",
        "行事曆",
        "會議",
        "今天",
        "明天",
    )
    negative = ("electron", "file", "edit", "view", "window", "help")
    return any(token in text for token in positive) and not all(token in text for token in negative)


def _screen_calendar_result_is_credible(result: dict) -> bool:
    if result.get("status") != "verified":
        return False
    title = str(result.get("event_title", "")).strip()
    event_time = str(result.get("event_time", "")).strip()
    evidence = [str(item).strip() for item in (result.get("evidence") or []) if str(item).strip()]
    if not title or not event_time:
        return False
    if len(title) < 4 or len(event_time) < 3:
        return False
    return len(evidence) >= 1


def _public_thought_for_action(action: str) -> str:
    mapping = {
        "DRAFT_CONTENT": "Generated a response based on the available context.",
        "GENERATE_IMAGE": "Generated an image based on the current request and app context.",
        "SEND_EMAIL": "Prepared an email draft for confirmation.",
        "SAVE_FILE": "Prepared file content for saving.",
        "SCHEDULE_MEETING": "Prepared a meeting draft for confirmation.",
        "OPEN_APP": "Prepared an app action based on the current request.",
        "EXECUTE_PYTHON": "Analyzed the active document with Python.",
        "SEARCH_LOCAL_DOCS": "Ranked the most relevant local files for the current request.",
    }
    return mapping.get(action, "Processed the request using the current context.")


def _code_attempts_package_install(code: str) -> bool:
    lowered = (code or "").lower()
    blocked_signals = (
        "pip install",
        "python -m pip",
        "subprocess.run([\"pip\"",
        "subprocess.run(['pip'",
        "pip3 install",
        "uv pip install",
    )
    return any(signal in lowered for signal in blocked_signals)


def _ocr_text_has_google_calendar_signal(ocr_text: str) -> bool:
    text = normalize_query(ocr_text)
    signals = (
        "calendar.google.com",
        "google calendar",
        "/week",
        "/day",
        "/month",
        "today",
        "week",
        "month",
    )
    return any(token in text for token in signals)


def _extract_calendar_from_screen(
    client: genai.Client,
    model_name: str,
    fallback_model: str,
    screen_b64: str,
    user_input: str,
    active_window: str,
) -> dict:
    contents = [{
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": "image/png", "data": screen_b64}},
            {"text": (
                f"Active window: {active_window or 'Unknown'}\n"
                f"User request: {user_input}\n"
                "Extract only clearly visible calendar information."
            )},
        ],
    }]
    response, _ = _generate_with_fallback(
        client,
        model_name,
        fallback_model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=CALENDAR_SCREEN_EXTRACT_PROMPT),
    )
    raw = _clean_json(response.text)
    result = json.loads(raw)
    if result.get("status") not in {"verified", "unverified"}:
        result["status"] = "unverified"
    return result


def _extract_calendar_from_screen_with_ocr_hint(
    client: genai.Client,
    model_name: str,
    fallback_model: str,
    screen_b64: str,
    user_input: str,
    active_window: str,
    ocr_text: str,
) -> dict:
    contents = [{
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": "image/png", "data": screen_b64}},
            {"text": (
                f"Active window: {active_window or 'Unknown'}\n"
                f"User request: {user_input}\n\n"
                "OCR text hint from the same screenshot:\n"
                f"{ocr_text[:4000]}\n\n"
                "Focus on the calendar content area, not the browser toolbar or tabs."
            )},
        ],
    }]
    response, _ = _generate_with_fallback(
        client,
        model_name,
        fallback_model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=CALENDAR_SCREEN_WITH_OCR_HINT_PROMPT),
    )
    raw = _clean_json(response.text)
    result = json.loads(raw)
    if result.get("status") not in {"verified", "unverified"}:
        result["status"] = "unverified"
    return result


def _classify_intent(user_input: str, context: dict) -> dict:
    """
    Hybrid decision layer:
    - Use deterministic routing only for high-confidence intents with strong context.
    - Otherwise defer to the LLM orchestrator.
    """
    query = normalize_query(user_input)
    calendar_items = context.get("calendar", [])
    email_items = context.get("emails", [])
    document_items = context.get("documents", [])

    if _is_next_meeting_query(user_input) and calendar_items:
        return {
            "intent": "next_meeting_lookup",
            "mode": "deterministic",
            "confidence": 0.96,
            "reason": "The user explicitly asked about their next meeting and calendar context is available.",
        }

    if _contains_any(query, ("delete memory", "forget this preference", "清除記憶", "刪除記憶")):
        return {
            "intent": "memory_management",
            "mode": "deterministic",
            "confidence": 0.93,
            "reason": "The user explicitly asked to manage stored memory.",
        }

    if _contains_any(query, ("save this as", "save to file", "另存", "存成檔案")):
        return {
            "intent": "save_file",
            "mode": "deterministic",
            "confidence": 0.88,
            "reason": "The user explicitly asked to save content to a file.",
        }

    if calendar_items and _contains_any(query, ("meeting", "mtg", "calendar", "schedule", "sync", "會議", "行程")):
        return {
            "intent": "calendar_related",
            "mode": "llm",
            "confidence": 0.72,
            "reason": "Calendar context is relevant, but the exact action still benefits from model judgment.",
        }

    if email_items and _contains_any(query, ("email", "reply", "mail", "client", "寄信", "回信")):
        return {
            "intent": "email_related",
            "mode": "llm",
            "confidence": 0.72,
            "reason": "Email context is relevant, but drafting versus summarizing should still be decided by the model.",
        }

    if document_items:
        return {
            "intent": "document_related",
            "mode": "llm",
            "confidence": 0.64,
            "reason": "Document context exists, but the user intent is not explicit enough for deterministic routing.",
        }

    return {
        "intent": "general",
        "mode": "llm",
        "confidence": 0.45,
        "reason": "No high-confidence deterministic route matched, so the model should decide.",
    }


def _format_next_meeting_response(calendar_items: list[dict]) -> str | None:
    if not calendar_items:
        return None

    next_item = calendar_items[0]
    title = next_item.get("title", "Upcoming meeting")
    summary = next_item.get("summary", "")
    timestamp = next_item.get("timestamp")
    metadata = next_item.get("metadata") or {}
    starts_in = metadata.get("starts_in_minutes")
    attendees = metadata.get("attendees") or []

    lines = [f"Your next meeting is **{title}**."]
    if timestamp:
        lines.append(f"Start time: {timestamp}")
    if starts_in is not None:
        lines.append(f"It starts in about {starts_in} minutes.")
    if attendees:
        lines.append(f"Attendees: {', '.join(attendees)}")
    if summary:
        lines.append(f"Context: {summary}")
    return "\n".join(lines)


def _find_cv_file() -> str | None:
    """
    Find the most recent CV/resume PDF by scanning known local directories.
    First tries files with cv/resume keywords in the name; falls back to
    the most recently modified PDF in Downloads.
    """
    cv_keywords = {"cv", "resume", "curriculum", "簡歷", "履歷", "profile"}
    search_dirs = [Path.home() / "Downloads", Path.home() / "Documents", Path.home() / "Desktop"]
    extra = os.getenv("EXTRA_DATA_DIRS", "")
    for p in extra.split(":"):
        if p.strip():
            search_dirs.append(Path(p.strip()).expanduser())

    # Pass 1: keyword match
    candidates = []
    all_pdfs = []
    for folder in search_dirs:
        if not folder.exists():
            continue
        for f in folder.glob("*.pdf"):
            all_pdfs.append(f)
            if any(kw in f.name.lower() for kw in cv_keywords):
                candidates.append(f)

    if candidates:
        candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        logger.info("Found CV by keyword: %s", [f.name for f in candidates[:3]])
        return str(candidates[0])

    # Pass 2: fallback — most recently modified PDF anywhere in search dirs
    if all_pdfs:
        all_pdfs.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        logger.info("No keyword match — falling back to most recent PDF: %s", all_pdfs[0].name)
        return str(all_pdfs[0])

    return None


def _generate_with_fallback(client, primary_model: str, fallback_model: str, **kwargs):
    """Call Gemini with automatic fallback on 503 overload errors."""
    try:
        return client.models.generate_content(model=primary_model, **kwargs), primary_model
    except Exception as e:
        if "503" in str(e) or "UNAVAILABLE" in str(e) or "overloaded" in str(e).lower():
            logger.warning("Model %s unavailable, falling back to %s: %s", primary_model, fallback_model, e)
            return client.models.generate_content(model=fallback_model, **kwargs), fallback_model
        raise


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _elapsed_text(start_time: float) -> str:
    return f"Prompt completed in {time.perf_counter() - start_time:.1f}s"


def _clean_json(raw: str) -> str:
    raw = (raw or "").strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    # If still not a bare JSON object, extract the first {...} block
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
    return raw


async def generate_suggestions(active_window: str, privacy_mode: str = "safe") -> list[str]:
    """
    Keep the frontend contract as a string array.
    Suggestions should be cheap and deterministic: use preset quick prompts
    first, then fall back to lightweight context heuristics.
    """
    if not active_window or active_window in ("Unknown", ""):
        return []

    provider = ContextProvider()
    engine = SuggestionEngine()
    preset = engine.preset_suggestions_for_window(active_window)
    if preset:
        result = [item.text for item in preset[:3]]
        logger.info("Suggestions for '%s' [%s]: preset %s", active_window, privacy_mode, result)
        return result

    context = provider.get_context("", active_window, None, None, [])
    triggers = engine.detect_triggers("", context)
    suggestions = engine.generate_suggestions(triggers, context)

    # Enhanced mode: if real/local context did not produce suggestions,
    # fall back to screen-derived inference.
    if not suggestions and privacy_mode == ENHANCED_MODE:
        screen_signal = capture_screen_base64()
        synthetic_query = ""
        if screen_signal:
            lower_window = active_window.lower()
            if "lark" in lower_window:
                synthetic_query = "meeting calendar agenda"
            elif "outlook" in lower_window:
                synthetic_query = "email calendar follow up"
            elif "calendar" in lower_window:
                synthetic_query = "meeting calendar"
            elif "mail" in lower_window:
                synthetic_query = "email inbox follow up"

        if synthetic_query:
            context = provider.get_context(synthetic_query, active_window, None, None, [])
            triggers = engine.detect_triggers(synthetic_query, context)
            suggestions = engine.generate_suggestions(triggers, context)
            if suggestions:
                logger.info(
                    "Suggestions for '%s' [%s]: using screen-derived fallback",
                    active_window,
                    privacy_mode,
                )

    result = [item.text for item in suggestions]
    logger.info("Suggestions for '%s' [%s]: %s", active_window, privacy_mode, result)
    return result


async def generate_chat_title(user_query: str) -> str:
    """Ask Gemini for a 4-6 word chat title based on the first user message."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model   = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    if not api_key:
        return user_query[:48]
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=(
                f"Give a 4-6 word chat title for a conversation that starts with this message. "
                f"Reply with ONLY the title, no punctuation, no quotes:\n\n{user_query[:300]}"
            ),
        )
        return (resp.text or "").strip()[:60] or user_query[:48]
    except Exception:
        return user_query[:48]


async def _generate_image(
    client: genai.Client,
    original_prompt: str,
    active_window: str,
    model_name: str,
) -> tuple[str | None, str]:
    """
    Auto-augment the prompt with window context, then generate image.
    Returns (base64_png_or_none, augmented_prompt).
    """
    # Step 1: augment the prompt
    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", FALLBACK_MODEL)
    aug_response, used_aug_model = _generate_with_fallback(
        client,
        model_name,
        fallback_model,
        contents=IMAGE_AUGMENT_PROMPT.format(
            active_window=active_window,
            original_prompt=original_prompt,
        ),
    )
    augmented = (aug_response.text or "").strip()
    if used_aug_model != model_name:
        logger.warning("Image prompt augmentation fallback: %s -> %s", model_name, used_aug_model)
    logger.info("Augmented image prompt: %s", augmented[:120])

    # Step 2: generate image with Gemini image model
    image_model = os.getenv("GEMINI_IMAGE_MODEL", IMAGE_MODEL)
    try:
        img_response = client.models.generate_content(
            model=image_model,
            contents=augmented,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in img_response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                raw_bytes = part.inline_data.data
                _LAST_IMAGE_PATH.write_bytes(raw_bytes)
                logger.info("Saved generated image to %s", _LAST_IMAGE_PATH)
                b64 = base64.b64encode(raw_bytes).decode("utf-8")
                mime = part.inline_data.mime_type
                return f"data:{mime};base64,{b64}", augmented
    except Exception as e:
        logger.error("Image generation failed with %s: %s", image_model, e)
        # Try fallback model
        try:
            img_response = client.models.generate_content(
                model=IMAGE_MODEL_FALLBACK,
                contents=augmented,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for part in img_response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    raw_bytes = part.inline_data.data
                    _LAST_IMAGE_PATH.write_bytes(raw_bytes)
                    logger.info("Saved generated image to %s", _LAST_IMAGE_PATH)
                    b64 = base64.b64encode(raw_bytes).decode("utf-8")
                    mime = part.inline_data.mime_type
                    return f"data:{mime};base64,{b64}", augmented
        except Exception as e2:
            logger.error("Fallback image model also failed: %s", e2)

    return None, augmented


async def run_agent_stream(user_input: str, history: list[dict] | None = None) -> AsyncGenerator[str, None]:
    start_time = time.perf_counter()
    context_provider = ContextProvider()
    memory_store = MemoryStore()
    prompt_builder = PromptBuilder()
    suggestion_engine = SuggestionEngine()
    privacy_mode = load_privacy_mode()["mode"]
    normalized_user_input = normalize_query(user_input)

    # ── Step 1: Window Context ─────────────────────────────────────────────
    yield _sse({"step": "context", "text": "Reading your active application..."})
    active_window = get_active_window_title() or "Unknown"
    yield _sse({"step": "context", "text": f"Active window: {active_window}"})

    # ── Step 1.5: Fast Intent Router ───────────────────────────────────────
    try:
        intent_plan = _route_intent_fast(active_window, user_input)
    except Exception as exc:
        logger.warning("Fast intent router failed: %s", exc)
        intent_plan = IntentPlan(
            actions=["DRAFT_CONTENT"],
            needs_screenshot=False,
            needs_rag=False,
            reason="Router failed, so falling back to the default path.",
            plan="Answer directly with the context already available.",
            source="fallback",
        )
    yield _sse({
        "step": "decision",
        "text": (
            f"Fast router: actions={intent_plan.actions}, "
            f"screenshot={'yes' if intent_plan.needs_screenshot else 'no'}, "
            f"rag={'yes' if intent_plan.needs_rag else 'no'}, "
            f"source={intent_plan.source}"
        ),
    })
    yield _sse({"step": "decision", "text": f"Fast router reason: {intent_plan.reason}"})
    yield _sse({"step": "decision", "text": f"Fast router plan: {intent_plan.plan}"})

    # Read active document context early because it can affect downstream execution,
    # especially for spreadsheet analysis routed to Python.
    yield _sse({"step": "context", "text": "Reading active document content..."})
    doc_text, doc_path = get_active_document_content()
    screen_b64 = None
    if doc_text:
        fname = Path(doc_path).name if doc_path else "document"
        yield _sse({"step": "context", "text": f"Extracted text from {fname} ({len(doc_text)} chars) — not uploaded anywhere"})
    else:
        yield _sse({"step": "context", "text": "No structured document detected in the active app"})

    # ── Auto-locate named file when no active local document ────────────────
    # Triggers when: no doc_path, OR doc_path is a URL (browser noise)
    _is_url_doc = doc_path and (doc_path.startswith("http://") or doc_path.startswith("https://"))
    if not doc_path or _is_url_doc:
        import glob as _glob, re as _re
        _DOC_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".xls", ".csv", ".pptx", ".ppt", ".numbers", ".pages"}
        _STOP_WORDS = {"the", "this", "that", "with", "from", "and", "for", "一個", "文檔", "文件", "重新", "總結", "創建", "一次", "新的", "新"}
        search_dirs = [Path.home() / "Downloads", Path.home() / "Documents", Path.home() / "Desktop"]

        # Strategy 1: quoted filenames or bare filenames with known extensions
        name_candidates = _re.findall(r'[「「"\'](.*?)[」」"\']', user_input)
        name_candidates += _re.findall(r'(\S+\.(?:docx?|xlsx?|csv|pdf|txt|md|pptx?|numbers|pages))', user_input, _re.IGNORECASE)

        # Strategy 2: keyword matching against local filenames (e.g. "cohere" → Cohere*.docx)
        if not name_candidates:
            # Insert spaces at Chinese↔ASCII transitions so "把cohere" → "把 cohere"
            _spaced = _re.sub(r'([a-zA-Z0-9])([^\x00-\x7F])', r'\1 \2', user_input)
            _spaced = _re.sub(r'([^\x00-\x7F])([a-zA-Z0-9])', r'\1 \2', _spaced)
            keywords = [w.lower() for w in _re.split(r'[\s,，、。！？\?\!&]+', _spaced)
                        if len(w) > 2 and w.lower() not in _STOP_WORDS]
            if keywords:
                seen_paths = set()
                for d in search_dirs:
                    if not d.exists():
                        continue
                    for f in sorted(d.iterdir(), key=lambda x: -x.stat().st_mtime):
                        if f.is_file() and f.suffix.lower() in _DOC_EXTS and str(f) not in seen_paths:
                            fname_lower = f.name.lower()
                            if any(kw in fname_lower for kw in keywords):
                                name_candidates.append(f.name)
                                seen_paths.add(str(f))

        if name_candidates:
            for candidate in name_candidates:
                candidate = candidate.strip()
                if not candidate:
                    continue
                for d in search_dirs:
                    for pattern in [candidate, f"*{candidate}*"]:
                        for hit in _glob.glob(str(d / "**" / pattern), recursive=True):
                            hit_path = Path(hit)
                            if hit_path.is_file() and hit_path.suffix.lower() in _DOC_EXTS:
                                from window_context import _extract_text_from_file
                                text = _extract_text_from_file(str(hit_path))
                                if text:
                                    doc_text = text
                                    doc_path = str(hit_path)
                                    yield _sse({"step": "context", "text": f"Found and read: {hit_path.name} ({len(text)} chars)"})
                                    break
                        if doc_path and not _is_url_doc:
                            break
                    if doc_path and not _is_url_doc:
                        break
                if doc_path and not _is_url_doc:
                    break

        # If URL doc was replaced by a local file, clear the URL noise
        if _is_url_doc and doc_path and not (doc_path.startswith("http://") or doc_path.startswith("https://")):
            pass  # successfully replaced URL with local file
        elif _is_url_doc and (not doc_path or doc_path.startswith("http")):
            # Still a URL — suppress it to avoid polluting context with browser noise
            doc_text = None
            doc_path = None

    if intent_plan.needs_screenshot:
        # Also fetch browser page content here (same intent gate as screenshot)
        if not doc_text:
            browser_text, browser_url = get_browser_page_content()
            if browser_text:
                doc_text = browser_text
                doc_path = browser_url
                yield _sse({"step": "context", "text": f"Read browser page: {len(browser_text)} chars"})
        yield _sse({"step": "context", "text": "Capturing current app window for visual context..."})
        screen_b64 = capture_screen_base64()
        if screen_b64:
            approx_bytes = int(len(screen_b64) * 3 / 4)
            yield _sse({"step": "context", "text": f"Screen capture status: success (mode={get_last_capture_mode()}, approx {approx_bytes} bytes base64-decoded) — sent to AI, not stored locally"})
        else:
            yield _sse({"step": "context", "text": "Screen capture status: unavailable"})
    else:
        yield _sse({"step": "context", "text": "Skipping screenshot capture for this intent"})

    # ── Step 2: Local RAG Search ───────────────────────────────────────────
    # Always run RAG for SEARCH_LOCAL_DOCS — the filesystem scan alone isn't enough
    if intent_plan.action == "SEARCH_LOCAL_DOCS":
        intent_plan.needs_rag = True
    rag_results = []
    if intent_plan.needs_rag:
        yield _sse({"step": "search", "text": "Searching local knowledge base..."})
        try:
            rag_results = search_docs(user_input, top_k=3)
            if rag_results:
                yield _sse({"step": "search", "text": f"Found {len(rag_results)} relevant document(s) (top score: {rag_results[0]['score']:.3f})"})
            else:
                yield _sse({"step": "heal", "text": "No local documents matched — AI expanding to general knowledge..."})
                rag_results = []
        except Exception as e:
            logger.warning("RAG search failed: %s", e)
            rag_results = []
            yield _sse({"step": "heal", "text": "Local knowledge base unavailable — AI is falling back to general reasoning..."})
    else:
        yield _sse({"step": "search", "text": "Skipping local knowledge-base search for this intent"})

    # ── Step 2.5: Context + Memory + Decision ─────────────────────────────
    context = context_provider.get_context(user_input, active_window, doc_text, doc_path, rag_results)
    structured_calendar = _structured_calendar_items(context)
    calendar_meta = context.get("calendar_meta", {}) or {}
    yield _sse({
        "step": "context",
        "text": (
            "Context loaded: "
            f"docs={len(context.get('documents', []) or [])}, "
            f"emails={len(context.get('emails', []) or [])}, "
            f"structured_calendar={len(structured_calendar)}"
        ),
    })
    yield _sse({
        "step": "context",
        "text": (
            "Calendar connector status: "
            f"structured_source={calendar_meta.get('structured_source', 'unknown')}, "
            f"lark_connector={'yes' if calendar_meta.get('supports_lark_connector') else 'no'}, "
            f"window_hint={'yes' if calendar_meta.get('window_hint') else 'no'}, "
            f"lark_hint={'yes' if calendar_meta.get('lark_hint') else 'no'}"
        ),
    })
    yield _sse({
        "step": "context",
        "text": "Structured calendar filters: excluding Birthdays and Siri Suggestions; holiday calendars are included.",
    })
    if calendar_meta.get("lark_hint") and not structured_calendar:
        yield _sse({
            "step": "context",
            "text": "Calendar note: the active window looks Lark-related, but this build does not have a Lark Calendar connector. Structured calendar data currently comes from macOS Calendar only.",
        })

    memory_updates = memory_store.maybe_update_from_user_input(user_input)
    memory = memory_store.get_preferences()
    if memory_updates:
        updates = ", ".join(f"{key}={value}" for key, value in memory_updates.items())
        yield _sse({"step": "memory", "text": f"Updated explicit memory: {updates}"})
    elif memory:
        memory_summary = ", ".join(f"{key}={value}" for key, value in memory.items())
        yield _sse({"step": "memory", "text": f"Loaded memory: {memory_summary}"})

    triggers = suggestion_engine.detect_triggers(user_input, context)
    structured_suggestions = suggestion_engine.generate_suggestions(triggers, context)
    decision_mode = suggestion_engine.decide_mode(structured_suggestions)
    yield _sse({"step": "decision", "text": f"Decision layer: {decision_mode} mode"})
    if structured_suggestions:
        yield _sse({
            "step": "suggestions",
            "mode": decision_mode,
            "suggestions": [item.to_dict() for item in structured_suggestions],
        })

    intent = _classify_intent(user_input, context)
    yield _sse({
        "step": "decision",
        "text": (
            f"Intent classifier: {intent['intent']} "
            f"(mode={intent['mode']}, confidence={intent['confidence']:.2f})"
        ),
    })

    if intent["mode"] == "deterministic" and intent["intent"] == "next_meeting_lookup":
        next_meeting_response = _format_next_meeting_response(structured_calendar)
        if next_meeting_response:
            yield _sse({"step": "decision", "text": "Hybrid routing: answering directly from calendar context"})
            yield _sse({
                "step": "result",
                "thought": (
                    "The intent classifier marked this as a high-confidence next-meeting lookup, "
                    "so the system answered directly from structured calendar context."
                ),
                "action": "DRAFT_CONTENT",
                "payload": next_meeting_response,
            })
            yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
            return

    screen_calendar_fallback = False
    screen_inferred_calendar = None
    if _requires_verified_calendar_data(user_input, active_window) and not structured_calendar:
        calendar_window_ok = _screen_calendar_fallback_allowed(active_window)
        yield _sse({
            "step": "decision",
            "text": (
                "Calendar fallback eligibility: "
                f"privacy_mode={privacy_mode}, "
                f"screen_capture={'yes' if bool(screen_b64) else 'no'}, "
                f"calendar_window_hint={'yes' if calendar_window_ok else 'no'}"
            ),
        })
        if privacy_mode == ENHANCED_MODE and screen_b64 and _looks_like_lark_calendar_surface(active_window, screen_b64):
            yield _sse({
                "step": "decision",
                "text": "Calendar fallback: no structured event found, so the system is attempting a dedicated screen extractor. It will only answer if the OCR preview or active window looks calendar-related, and any answer will be labeled as screen-inferred.",
            })
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                yield _sse({"step": "error", "text": "GEMINI_API_KEY not set. Add it to server/.env"})
                return
            model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
            fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", FALLBACK_MODEL)
            client = genai.Client(api_key=api_key)
            try:
                ocr_text = ""
                try:
                    yield _sse({"step": "think", "text": "OCR extractor: reading calendar text from the screenshot with macOS Vision..."})
                    ocr_text = _ocr_screen_text_via_vision(screen_b64)
                except Exception as exc:
                    logger.warning("Calendar OCR failed: %s", exc)
                    yield _sse({"step": "heal", "text": f"OCR extractor failed: {str(exc)[:180]}. Falling back to direct image extraction."})

                if ocr_text.strip():
                    ocr_preview = " | ".join(line.strip() for line in ocr_text.splitlines()[:8] if line.strip())[:500] or "none"
                    yield _sse({
                        "step": "context",
                        "text": f"OCR extractor result: {len(ocr_text.splitlines())} lines detected; preview={ocr_preview}",
                    })
                    if calendar_window_ok or _ocr_text_looks_calendar_related(ocr_text):
                        yield _sse({"step": "think", "text": f"OCR parser: validating extracted calendar text with {model_name}..."})
                        screen_inferred_calendar = _extract_calendar_from_ocr_text(
                            client=client,
                            model_name=model_name,
                            fallback_model=fallback_model,
                            user_input=user_input,
                            active_window=active_window,
                            ocr_text=ocr_text,
                        )
                        if (
                            screen_inferred_calendar.get("status") != "verified"
                            and _ocr_text_has_google_calendar_signal(ocr_text)
                        ):
                            yield _sse({
                                "step": "think",
                                "text": "OCR parser could not verify the event, so the system is running a second pass over the screenshot using the OCR text as a hint.",
                            })
                            screen_inferred_calendar = _extract_calendar_from_screen_with_ocr_hint(
                                client=client,
                                model_name=model_name,
                                fallback_model=fallback_model,
                                screen_b64=screen_b64,
                                user_input=user_input,
                                active_window=active_window,
                                ocr_text=ocr_text,
                            )
                        if screen_inferred_calendar.get("status") != "verified":
                            yield _sse({
                                "step": "think",
                                "text": "OCR parser did not verify a result, so the system is falling back to direct screenshot interpretation.",
                            })
                            screen_inferred_calendar = _extract_calendar_from_screen_with_ocr_hint(
                                client=client,
                                model_name=model_name,
                                fallback_model=fallback_model,
                                screen_b64=screen_b64,
                                user_input=user_input,
                                active_window=active_window,
                                ocr_text=ocr_text,
                            )
                    else:
                        yield _sse({
                            "step": "think",
                            "text": "OCR preview is noisy, so the system is skipping OCR parsing and using direct screenshot interpretation instead.",
                        })
                        screen_inferred_calendar = _extract_calendar_from_screen(
                            client=client,
                            model_name=model_name,
                            fallback_model=fallback_model,
                            screen_b64=screen_b64,
                            user_input=user_input,
                            active_window=active_window,
                        )
                else:
                    yield _sse({"step": "think", "text": f"Screen extractor: validating visible calendar details with {model_name}..."})
                    screen_inferred_calendar = _extract_calendar_from_screen(
                        client=client,
                        model_name=model_name,
                        fallback_model=fallback_model,
                        screen_b64=screen_b64,
                        user_input=user_input,
                        active_window=active_window,
                    )

                evidence = screen_inferred_calendar.get("evidence") or []
                evidence_text = " | ".join(str(item).strip() for item in evidence[:3] if str(item).strip()) or "none"
                yield _sse({
                    "step": "context",
                    "text": (
                        "Screen extractor result: "
                        f"status={screen_inferred_calendar.get('status', 'unverified')}, "
                        f"source={screen_inferred_calendar.get('source', 'unknown')}, "
                        f"event_title={screen_inferred_calendar.get('event_title', '') or '<empty>'}, "
                        f"event_time={screen_inferred_calendar.get('event_time', '') or '<empty>'}, "
                        f"evidence={evidence_text}"
                    ),
                })
                if screen_inferred_calendar and screen_inferred_calendar.get("status") == "verified":
                    yield _sse({
                        "step": "result",
                        "thought": "No structured calendar connector matched, so the system answered from the screenshot and labeled the result as inferred.",
                        "action": "DRAFT_CONTENT",
                        "payload": _format_screen_inferred_calendar_response(screen_inferred_calendar),
                    })
                    yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
                    return
            except Exception as exc:
                logger.warning("Calendar screen extraction failed: %s", exc)
                yield _sse({
                    "step": "heal",
                    "text": "Calendar screen extractor could not verify visible meeting details.",
                })

        yield _sse({
            "step": "decision",
            "text": "Calendar safety gate: even after screenshot interpretation, the system could not produce a calendar answer for this turn.",
        })
        yield _sse({
            "step": "result",
            "thought": "This request needs verified calendar data, and no structured calendar event is available.",
            "action": "DRAFT_CONTENT",
            "payload": _build_unverified_calendar_response(),
        })
        yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
        return

    # ── Step 3: Gemini Decision ────────────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        yield _sse({"step": "error", "text": "GEMINI_API_KEY not set. Add it to server/.env"})
        return

    model_name = os.getenv("GEMINI_EXECUTOR_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    yield _sse({"step": "think", "text": f"Consulting {model_name}..."})

    try:
        client = genai.Client(api_key=api_key)

        # Build multi-turn contents: previous history + current user turn
        contents = []
        for msg in (history or []):
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        # Current turn: include screenshot if available
        prompt_bundle = prompt_builder.build_augmented_prompt(
            user_input=user_input,
            active_window=active_window,
            context=context,
            memory=memory,
            rag_results=rag_results,
            doc_text=doc_text,
            doc_path=doc_path,
        )
        if prompt_bundle["context_used"]:
            yield _sse({"step": "context", "text": f"Prompt augmented with: {', '.join(prompt_bundle['context_used'])}"})
        else:
            yield _sse({"step": "context", "text": "Prompt using minimal context fallback"})
        user_text = prompt_bundle["augmented_prompt"]
        if screen_calendar_fallback:
            yield _sse({"step": "context", "text": "Calendar screen inference enabled for this turn: the screenshot is being provided to the model as fallback evidence"})
            user_text += "\n\n" + _build_screen_inferred_calendar_prompt(normalized_user_input)
        if screen_b64:
            current_parts = [
                {"inline_data": {"mime_type": "image/png", "data": screen_b64}},
                {"text": user_text},
            ]
        else:
            current_parts = [{"text": user_text}]
        contents.append({"role": "user", "parts": current_parts})

        fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", FALLBACK_MODEL)
        logger.info("Sending %d-turn conversation to Gemini", len(contents))

        _ACTION_CARD_ACTIONS = {"SEND_EMAIL", "SAVE_FILE", "DELETE_FILE", "SCHEDULE_MEETING", "OPEN_APP", "UNDO_ACTION"}

        # Optimise: DRAFT_CONTENT → ACTION_CARD is wasteful — skip the intermediate step.
        # Go straight to the action card so the model writes the real content in one shot.
        if (len(intent_plan.actions) > 1
                and intent_plan.action == "DRAFT_CONTENT"
                and intent_plan.actions[1] in _ACTION_CARD_ACTIONS):
            logger.info("Skipping intermediate DRAFT_CONTENT; jumping to %s directly", intent_plan.actions[1])
            intent_plan.actions = intent_plan.actions[1:]
        if intent_plan.action == "EXECUTE_PYTHON":
            result = {"thought": "Analyzing document with Python.", "action": "EXECUTE_PYTHON", "payload": "GENERATE_CODE"}
        elif intent_plan.action in _ACTION_CARD_ACTIONS:
            # Inject router decision as a constraint so the model fills payload correctly
            # but cannot silently downgrade to DRAFT_CONTENT.
            action_hint = (
                f'\n\nIMPORTANT: The fast router has already decided the action is "{intent_plan.action}". '
                f'You MUST set "action": "{intent_plan.action}" in your response. Do not change it to DRAFT_CONTENT.'
            )
            contents[-1]["parts"][-1]["text"] += action_hint
            response, used_model = _generate_with_fallback(
                client, model_name, fallback_model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            if used_model != model_name:
                yield _sse({"step": "heal", "text": f"{model_name} overloaded — switched to {used_model}"})
            raw = _clean_json(response.text)
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {"thought": "", "action": intent_plan.action, "payload": {}}
            # Hard-enforce the router's action even if model disobeyed
            if result.get("action") != intent_plan.action:
                logger.warning("Execution model overrode router action %s → %s; reverting", intent_plan.action, result.get("action"))
                result["action"] = intent_plan.action
        else:
            response, used_model = _generate_with_fallback(
                client, model_name, fallback_model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            if used_model != model_name:
                yield _sse({"step": "heal", "text": f"{model_name} overloaded — switched to {used_model}"})
            raw = _clean_json(response.text)
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                raw_text = response.text or ""
                # Fast path: if model clearly decided EXECUTE_PYTHON but embedded code in JSON,
                # skip the JSON battle and jump straight to code-generation step
                if '"action": "EXECUTE_PYTHON"' in raw_text or "'action': 'EXECUTE_PYTHON'" in raw_text:
                    logger.info("Detected EXECUTE_PYTHON in malformed JSON — bypassing parse, going to code-gen")
                    result = {"thought": "Analyzing document with Python.", "action": "EXECUTE_PYTHON", "payload": "GENERATE_CODE"}
                else:
                    # Self-healing: retry with an explicit re-prompt
                    yield _sse({"step": "heal", "text": "Response format error — AI is self-correcting and retrying..."})
                    logger.warning("Gemini returned non-JSON on first attempt, retrying: %s", raw_text[:200])
                    retry_contents = contents + [
                        {"role": "model", "parts": [{"text": raw_text}]},
                        {"role": "user", "parts": [{"text": (
                            "Your previous response was not valid JSON. "
                            "You MUST reply with ONLY a single valid JSON object using exactly this schema, "
                            "no markdown, no explanation:\n"
                            '{"thought":"...","action":"...","payload":"..."}'
                        )}]},
                    ]
                    response, _ = _generate_with_fallback(
                        client, model_name, fallback_model,
                        contents=retry_contents,
                        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                    )
                    raw = _clean_json(response.text)
                    result = json.loads(raw)

        for key in ("thought", "action", "payload"):
            if key not in result:
                raise ValueError(f"Missing key '{key}' in Gemini response")

        action = result["action"]
        result["thought"] = _public_thought_for_action(action)
        if screen_calendar_fallback and action == "DRAFT_CONTENT":
            result["payload"] = _format_inferred_calendar_payload(str(result.get("payload", "")))
        yield _sse({"step": "think", "text": f"Decision: {action}"})

        # ── Image Generation ───────────────────────────────────────────────
        if action == "GENERATE_IMAGE":
            yield _sse({"step": "think", "text": "Auto-augmenting image prompt with window context..."})
            image_data, augmented_prompt = await _generate_image(
                client, result["payload"], active_window, model_name
            )
            yield _sse({"step": "think", "text": f"Prompt: {augmented_prompt[:80]}..."})
            if image_data:
                yield _sse({
                    "step": "image",
                    "thought": result["thought"],
                    "action": action,
                    "image_data": image_data,
                    "prompt": augmented_prompt,
                })
            else:
                yield _sse({
                    "step": "result",
                    "thought": result["thought"],
                    "action": "DRAFT_CONTENT",
                    "payload": f"Image generation unavailable. Enhanced prompt:\n\n{augmented_prompt}",
                })
            yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
            return

        # ── Action Cards (SEND_EMAIL / SAVE_FILE / SCHEDULE_MEETING / OPEN_APP) ──
        # NOTE: DELETE_FILE is handled separately below (needs file search + candidate list)
        if action in ("SEND_EMAIL", "SAVE_FILE", "SCHEDULE_MEETING", "OPEN_APP"):
            try:
                action_payload = json.loads(result["payload"]) if isinstance(result["payload"], str) else result["payload"]
            except (json.JSONDecodeError, TypeError):
                action_payload = {"content": result["payload"]}

            # If SAVE_FILE payload parsing failed and filename is missing,
            # try to recover by re-parsing the content string as JSON
            if action == "SAVE_FILE" and "filename" not in action_payload:
                raw_content = action_payload.get("content", "")
                if isinstance(raw_content, str):
                    try:
                        recovered = json.loads(raw_content)
                        if isinstance(recovered, dict) and "filename" in recovered:
                            action_payload = recovered
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Force .docx for document saves — model often defaults to .txt
            if action == "SAVE_FILE":
                fname = action_payload.get("filename", "")
                if fname.lower().endswith(".txt"):
                    action_payload["filename"] = fname[:-4] + ".docx"

            # Auto-attach logic for SEND_EMAIL
            if action == "SEND_EMAIL":
                query = user_input.lower()

                # 1. Image attachment: if user mentions "this image / the image / the photo"
                image_keywords = {"this image", "the image", "this photo", "the photo",
                                   "this picture", "the picture", "this figure", "這張圖",
                                   "this chart", "the chart"}
                if any(kw in query for kw in image_keywords):
                    if _LAST_IMAGE_PATH.exists():
                        action_payload["attachment_path"] = str(_LAST_IMAGE_PATH)
                        logger.info("Auto-attaching last generated image: %s", _LAST_IMAGE_PATH)
                        yield _sse({"step": "search", "text": "Auto-attaching last generated image"})
                    else:
                        logger.warning("User mentioned image but no generated image found on disk")

                # 2. Spreadsheet report attachment: generate a real Word report with embedded chart.
                elif (
                    doc_path
                    and Path(doc_path).suffix.lower() in {".xlsx", ".xls", ".csv"}
                    and any(kw in query for kw in {"word", "docx", "document", "文檔", "文件", "報告", "report"})
                ):
                    try:
                        report_path = build_spreadsheet_report(doc_path, user_input)
                        action_payload["attachment_path"] = report_path
                        logger.info("Auto-attaching generated spreadsheet report: %s", report_path)
                        yield _sse({"step": "search", "text": f"Generated report attachment: {Path(report_path).name}"})
                    except Exception as exc:
                        logger.warning("Failed to generate spreadsheet report attachment: %s", exc)

                # 3. CV attachment: filesystem scan for resume PDFs
                elif any(kw in query for kw in {"cv", "resume", "curriculum vitae"}):
                    logger.info("CV email detected; action_payload attachment_path=%r", action_payload.get("attachment_path"))
                    pdf_path = _find_cv_file()
                    if pdf_path:
                        action_payload["attachment_path"] = pdf_path
                        logger.info("Overriding attachment_path with: %s", pdf_path)
                        yield _sse({"step": "search", "text": f"Auto-attaching CV: {Path(pdf_path).name}"})
                    else:
                        logger.warning("No CV file found on filesystem")

                # 4. Always clear any hallucinated path from Gemini
                elif not action_payload.get("attachment_path") or not Path(str(action_payload.get("attachment_path", ""))).exists():
                    action_payload["attachment_path"] = None

            yield _sse({
                "step": "action_card",
                "thought": result["thought"],
                "action": action,
                "payload": action_payload,
            })
            yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
            return

        # ── EXECUTE_PYTHON: two-step — separate code generation call, then execute ──
        if action == "EXECUTE_PYTHON":
            yield _sse({"step": "think", "text": "Generating Python analysis code..."})

            # If doc_path wasn't captured at query time, retry now
            if not doc_path:
                _, doc_path = get_active_document_content()
                if doc_path:
                    yield _sse({"step": "context", "text": f"Re-detected document: {Path(doc_path).name}"})

            # If doc_path is a URL (browser was active), it's not a local file — discard it
            if doc_path and (doc_path.startswith("http://") or doc_path.startswith("https://")):
                logger.info("doc_path is a URL, discarding for EXECUTE_PYTHON: %s", doc_path)
                doc_path = None
                doc_text = None

            # Fast path: deterministic analytics for common spreadsheet questions.
            try:
                deterministic = run_deterministic_analysis(doc_path, user_input)
            except Exception as analysis_error:
                logger.warning("Deterministic analytics failed, falling back to codegen: %s", analysis_error)
                deterministic = None

            if deterministic and deterministic.handled:
                yield _sse({"step": "think", "text": "Using deterministic analytics path for this spreadsheet request..."})
                det_carry_text = f"**{deterministic.title} 分析結果**\n\n{deterministic.body}"
                yield _sse({
                    "step": "result",
                    "thought": "Used the built-in analytics pipeline for a faster and more stable spreadsheet analysis result.",
                    "action": "DRAFT_CONTENT",
                    "payload": det_carry_text,
                })
                # Multi-action continuation
                det_pending = intent_plan.actions[1:]
                for next_action in det_pending:
                    yield _sse({"step": "think", "text": f"Continuing to next step: {next_action}..."})
                    followup_hint = (
                        f'\n\nThe analysis produced this result:\n"""\n{det_carry_text[:3000]}\n"""\n\n'
                        f'IMPORTANT: The next action is "{next_action}". '
                        f'You MUST set "action": "{next_action}". Use the analysis result above as the primary content.'
                    )
                    followup_contents = contents[:-1] + [{
                        "role": "user",
                        "parts": [{"text": contents[-1]["parts"][-1]["text"] + followup_hint}],
                    }]
                    fu_response, _ = _generate_with_fallback(
                        client, model_name, fallback_model,
                        contents=followup_contents,
                        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                    )
                    try:
                        fu_result = json.loads(_clean_json(fu_response.text))
                    except json.JSONDecodeError:
                        fu_result = {"thought": "", "action": next_action, "payload": {}}
                    fu_result["action"] = next_action
                    if next_action in ("SEND_EMAIL", "SAVE_FILE", "SCHEDULE_MEETING", "OPEN_APP"):
                        try:
                            fu_payload = json.loads(fu_result["payload"]) if isinstance(fu_result["payload"], str) else fu_result["payload"]
                        except (json.JSONDecodeError, TypeError):
                            fu_payload = {}
                        if next_action == "SEND_EMAIL" and not fu_payload.get("body"):
                            fu_payload["body"] = det_carry_text
                        if next_action == "SAVE_FILE":
                            if not fu_payload.get("content"):
                                fu_payload["content"] = det_carry_text
                            fn2 = fu_payload.get("filename", "")
                            if fn2.lower().endswith(".txt"):
                                fu_payload["filename"] = fn2[:-4] + ".docx"
                        yield _sse({
                            "step": "action_card",
                            "thought": _public_thought_for_action(next_action),
                            "action": next_action,
                            "payload": fu_payload,
                        })
                    else:
                        yield _sse({"step": "result", **fu_result})
                yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
                return

            # Step 2: dedicated code-generation call (plain text, no JSON wrapper)
            if doc_path:
                ext = Path(doc_path).suffix.lower()
                read_snippet = (
                    "pd.read_csv(os.environ['DOC_PATH'])" if ext == ".csv"
                    else "pd.read_excel(os.environ['DOC_PATH'])" if ext in (".xlsx", ".xls")
                    else "open(os.environ['DOC_PATH']).read()"
                )
                col_hint = ""
                if doc_text:
                    preview_lines = doc_text.splitlines()[:4]
                    col_hint = "File preview (header + first 3 rows):\n" + "\n".join(preview_lines) + "\n"
                file_context = (
                    f"File: {Path(doc_path).name} (full path in os.environ['DOC_PATH'])\n"
                    f"Read it with: {read_snippet}\n"
                    f"{col_hint}\n"
                )
            else:
                file_context = (
                    "No specific file is currently open. If the user mentions a filename,\n"
                    "search for it in these directories: ~/Downloads, ~/Documents, ~/Desktop\n"
                    "Example search:\n"
                    "  import glob, os\n"
                    "  target = 'filename.txt'\n"
                    "  dirs = [os.path.expanduser(d) for d in ['~/Downloads','~/Documents','~/Desktop']]\n"
                    "  matches = [p for d in dirs for p in glob.glob(os.path.join(d, '**', target), recursive=True)]\n"
                    "  path = matches[0] if matches else None\n"
                )
            code_resp, _ = _generate_with_fallback(
                client, model_name, fallback_model,
                contents=(
                    f"Write Python code to answer this request: {user_input}\n\n"
                    f"{file_context}"
                    "Rules:\n"
                    "- Import os and any needed libraries at the top\n"
                    "- NEVER install packages, call pip, use subprocess for package installation, or download dependencies\n"
                    "- If saving a file, ALWAYS save to os.path.expanduser('~/Downloads/'), never to /download or /Downloads\n"
                    "- Print results in friendly, human-readable Chinese if the query is in Chinese\n"
                    "- Use clear labels, counts AND percentages, e.g. 'majority: 26筆 (89.7%)'\n"
                    "- Always print a final user-facing summary; do not print setup logs\n"
                    "- NO code blocks, NO variable dumps — only clean human-readable output\n"
                    "- Output ONLY executable Python code, no markdown, no explanation"
                ),
            )
            code = (code_resp.text or "").strip()
            code = re.sub(r"^```python\s*", "", code)
            code = re.sub(r"\s*```$", "", code).strip()

            if _code_attempts_package_install(code):
                yield _sse({"step": "heal", "text": "Generated code tried to install packages — regenerating with stricter constraints..."})
                regen_resp, _ = _generate_with_fallback(
                    client, model_name, fallback_model,
                    contents=(
                        f"The previous Python code attempted to install packages, which is not allowed.\n\n"
                        f"User request: {user_input}\n"
                        f"{file_context}"
                        "Return replacement code that does NOT install anything.\n"
                        "If extra plotting libraries are unavailable, print a textual chart recommendation and the reason.\n"
                        "Always print a concise final analysis summary for the user.\n"
                        "Return ONLY executable Python code."
                    ),
                )
                code = (regen_resp.text or "").strip()
                code = re.sub(r"^```python\s*", "", code)
                code = re.sub(r"\s*```$", "", code).strip()

            yield _sse({"step": "think", "text": "Running Python analysis on document..."})
            logger.info("Executing Python code (doc_path=%s):\n%s", doc_path, code[:300])

            def _run_code(code_str: str) -> tuple[str, int]:
                tmp = Path(tempfile.mktemp(suffix=".py"))
                tmp.write_text(code_str, encoding="utf-8")
                exec_env = os.environ.copy()
                if doc_path:
                    exec_env["DOC_PATH"] = doc_path
                try:
                    p = __import__("subprocess").run(
                        [sys.executable, str(tmp)],
                        capture_output=True, text=True, timeout=60, env=exec_env,
                    )
                    return p.stdout.strip(), p.returncode, p.stderr.strip()
                finally:
                    tmp.unlink(missing_ok=True)

            try:
                stdout, returncode, stderr = _run_code(code)

                # Auto-fix: if code errored, send error back to AI and retry once
                if returncode != 0 and stderr:
                    yield _sse({"step": "heal", "text": "Code error detected — AI is fixing and retrying..."})
                    logger.warning("Python error, requesting fix:\n%s", stderr[:300])
                    fix_resp, _ = _generate_with_fallback(
                        client, model_name, fallback_model,
                        contents=(
                            f"This Python code produced an error. Fix it and return ONLY the corrected code:\n\n"
                            f"```python\n{code}\n```\n\n"
                            f"Error:\n{stderr}\n\n"
                            "Return ONLY executable Python code, no markdown, no explanation."
                        ),
                    )
                    fixed = (fix_resp.text or "").strip()
                    fixed = re.sub(r"^```python\s*", "", fixed)
                    fixed = re.sub(r"\s*```$", "", fixed).strip()
                    stdout, returncode, stderr = _run_code(fixed)
                    code = fixed

                # Auto-fix: if code runs but prints nothing, regenerate with explicit output requirements
                if returncode == 0 and not stdout.strip():
                    yield _sse({"step": "heal", "text": "Code ran without producing a user-facing answer — regenerating with explicit print instructions..."})
                    logger.warning("Python code produced no stdout, requesting regenerated output:\n%s", code[:300])
                    regen_output_resp, _ = _generate_with_fallback(
                        client, model_name, fallback_model,
                        contents=(
                            f"This Python code ran successfully but produced no user-facing output.\n\n"
                            f"```python\n{code}\n```\n\n"
                            f"User request: {user_input}\n"
                            f"{file_context}"
                            "Return replacement code that MUST print a concise final answer for the user.\n"
                            "Do not save files unless the user explicitly asked.\n"
                            "If recommending a chart, print the recommendation and short reasoning.\n"
                            "Return ONLY executable Python code."
                        ),
                    )
                    regenerated = (regen_output_resp.text or "").strip()
                    regenerated = re.sub(r"^```python\s*", "", regenerated)
                    regenerated = re.sub(r"\s*```$", "", regenerated).strip()
                    stdout, returncode, stderr = _run_code(regenerated)
                    code = regenerated

                output = stdout or (f"⚠️ Error:\n{stderr}" if stderr else "(No output produced)")
                fname = Path(doc_path).name
                py_carry_text = f"**{fname} 分析結果**\n\n{output}"
                yield _sse({
                    "step": "result",
                    "thought": result["thought"],
                    "action": "DRAFT_CONTENT",
                    "payload": py_carry_text,
                })

                # Multi-action: if next action is SEND_EMAIL or SAVE_FILE, continue
                py_pending = intent_plan.actions[1:]
                for next_action in py_pending:
                    yield _sse({"step": "think", "text": f"Continuing to next step: {next_action}..."})
                    followup_hint = (
                        f'\n\nThe Python analysis produced this result:\n"""\n{py_carry_text[:3000]}\n"""\n\n'
                        f'IMPORTANT: The next action is "{next_action}". '
                        f'You MUST set "action": "{next_action}". Use the analysis result above as the primary content.'
                    )
                    followup_contents = contents[:-1] + [{
                        "role": "user",
                        "parts": [{"text": contents[-1]["parts"][-1]["text"] + followup_hint}],
                    }]
                    fu_response, _ = _generate_with_fallback(
                        client, model_name, fallback_model,
                        contents=followup_contents,
                        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                    )
                    try:
                        fu_result = json.loads(_clean_json(fu_response.text))
                    except json.JSONDecodeError:
                        fu_result = {"thought": "", "action": next_action, "payload": {}}
                    fu_result["action"] = next_action
                    if next_action in ("SEND_EMAIL", "SAVE_FILE", "SCHEDULE_MEETING", "OPEN_APP"):
                        try:
                            fu_payload = json.loads(fu_result["payload"]) if isinstance(fu_result["payload"], str) else fu_result["payload"]
                        except (json.JSONDecodeError, TypeError):
                            fu_payload = {}
                        if next_action == "SEND_EMAIL" and not fu_payload.get("body"):
                            fu_payload["body"] = py_carry_text
                        if next_action == "SAVE_FILE":
                            if not fu_payload.get("content"):
                                fu_payload["content"] = py_carry_text
                            fname2 = fu_payload.get("filename", "")
                            if fname2.lower().endswith(".txt"):
                                fu_payload["filename"] = fname2[:-4] + ".docx"
                        yield _sse({
                            "step": "action_card",
                            "thought": _public_thought_for_action(next_action),
                            "action": next_action,
                            "payload": fu_payload,
                        })
                    else:
                        yield _sse({"step": "result", **fu_result})

                yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
            except Exception as exec_err:
                logger.error("EXECUTE_PYTHON failed: %s", exec_err)
                yield _sse({"step": "error", "text": f"Code execution failed: {exec_err}"})
            return

        # ── SEARCH_LOCAL_DOCS: scan filesystem for matching files ─────────
        if action == "SEARCH_LOCAL_DOCS":
            yield _sse({"step": "search", "text": "Scanning Downloads, Documents, Desktop..."})
            search_dirs = [
                Path.home() / "Downloads",
                Path.home() / "Documents",
                Path.home() / "Desktop",
            ]
            extra = os.getenv("EXTRA_DATA_DIRS", "")
            for p in extra.split(":"):
                if p.strip():
                    search_dirs.append(Path(p.strip()).expanduser())

            found = []
            for folder in search_dirs:
                if not folder.exists():
                    continue
                for ext in ("*.pdf", "*.docx", "*.doc", "*.txt", "*.md",
                            "*.xlsx", "*.xls", "*.csv", "*.pptx", "*.ppt",
                            "*.pages", "*.numbers", "*.key"):
                    for f in folder.glob(ext):
                        found.append(f)

            if found:
                from datetime import datetime as _dt
                found.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                yield _sse({"step": "search", "text": f"Found {len(found)} file(s) — ranking by relevance..."})

                file_list = "\n".join(
                    f"{i+1}. {f.name} | modified {_dt.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d')} | {f}"
                    for i, f in enumerate(found[:30])
                )
                pick_response = client.models.generate_content(
                    model=model_name,
                    contents=(
                        f"User asked: \"{user_input}\"\n\n"
                        f"Files found:\n{file_list}\n\n"
                        "Return the top 3-5 most relevant files ranked by how well they match the user's request. "
                        "Reply with ONLY valid JSON: "
                        '{"results": [{"path": "...", "name": "...", "reason": "<short reason>"}]}'
                    ),
                )
                try:
                    pick = json.loads(_clean_json(pick_response.text))
                    yield _sse({
                        "step": "file_results",
                        "thought": result["thought"],
                        "results": pick.get("results", []),
                    })
                except Exception:
                    # Fallback: show top 5 most recent
                    yield _sse({
                        "step": "file_results",
                        "thought": result["thought"],
                        "results": [
                            {"path": str(f), "name": f.name, "reason": "Most recently modified"}
                            for f in found[:5]
                        ],
                    })
                yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
                return
            else:
                yield _sse({"step": "heal", "text": "No files found in Downloads, Documents or Desktop."})

        # ── UNDO_ACTION: reverse the last state-mutating action (NL path) ───
        if action == "UNDO_ACTION":
            success, message = ledger.undo()
            yield _sse({
                "step": "result",
                "thought": "Reversing the last action.",
                "action": "DRAFT_CONTENT",
                "payload": message,
            })
            yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
            return

        # ── DELETE_FILE: search dirs, rank candidates, let user pick ─────────
        if action == "DELETE_FILE":
            yield _sse({"step": "search", "text": "Scanning Downloads, Documents, Desktop for matching files..."})
            search_dirs = [
                Path.home() / "Downloads",
                Path.home() / "Documents",
                Path.home() / "Desktop",
            ]
            extra = os.getenv("EXTRA_DATA_DIRS", "")
            for p in extra.split(":"):
                if p.strip():
                    search_dirs.append(Path(p.strip()).expanduser())

            found = []
            for folder in search_dirs:
                if not folder.exists():
                    continue
                for ext in ("*.pdf", "*.docx", "*.doc", "*.txt", "*.md",
                            "*.xlsx", "*.xls", "*.csv", "*.pptx", "*.ppt",
                            "*.pages", "*.numbers", "*.key", "*.ics", "*.png", "*.jpg"):
                    for f in folder.glob(ext):
                        found.append(f)

            if not found:
                yield _sse({"step": "heal", "text": "No files found in Downloads, Documents or Desktop."})
                yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
                return

            from datetime import datetime as _dt
            found.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            yield _sse({"step": "search", "text": f"Found {len(found)} file(s) — ranking by relevance to your request..."})

            file_list = "\n".join(
                f"{i+1}. {f.name} | modified {_dt.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d')} | {f}"
                for i, f in enumerate(found[:40])
            )
            # Fallback: most recent 6 files (always available)
            recent_fallback = [
                {"path": str(f), "name": f.name, "reason": "最近修改"}
                for f in found[:6]
            ]
            try:
                pick_response = client.models.generate_content(
                    model=model_name,
                    contents=(
                        f"User wants to delete a file. Their request: \"{user_input}\"\n\n"
                        f"Files available:\n{file_list}\n\n"
                        "Return EXACTLY 5 candidate files ranked by how likely the user wants to delete them. "
                        "Include both close matches AND recent files so the user has options. "
                        "reason must be SHORT (under 20 words). "
                        "Reply ONLY with valid JSON: "
                        '{"results": [{"path": "...", "name": "...", "reason": "..."}]}'
                    ),
                )
                pick = json.loads(_clean_json(pick_response.text))
                ai_candidates = pick.get("results", [])
                # Merge AI results with recent fallback; deduplicate by path
                seen = {c["path"] for c in ai_candidates}
                merged = ai_candidates + [c for c in recent_fallback if c["path"] not in seen]
                candidates = merged[:6]
            except Exception:
                candidates = recent_fallback

            yield _sse({
                "step": "action_card",
                "thought": result["thought"],
                "action": "DELETE_FILE",
                "payload": {"candidates": candidates},
            })
            yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
            return

        carry_text = str(result.get("payload", ""))
        yield _sse({"step": "result", **result})

        # ── Multi-action continuation ──────────────────────────────────────
        pending = intent_plan.actions[1:]
        for next_action in pending:
            yield _sse({"step": "think", "text": f"Continuing to next step: {next_action}..."})
            followup_hint = (
                f'\n\nThe previous step produced this result:\n"""\n{carry_text[:3000]}\n"""\n\n'
                f'IMPORTANT: The fast router has decided the next action is "{next_action}". '
                f'You MUST set "action": "{next_action}" in your response. '
                f'Use the result above as the primary content (e.g. email body or file content). '
                f'Do not change the action.'
            )
            followup_contents = contents[:-1] + [{
                "role": "user",
                "parts": [{"text": contents[-1]["parts"][-1]["text"] + followup_hint}],
            }]
            fu_response, fu_model = _generate_with_fallback(
                client, model_name, fallback_model,
                contents=followup_contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            fu_raw = _clean_json(fu_response.text)
            try:
                fu_result = json.loads(fu_raw)
            except json.JSONDecodeError:
                fu_result = {"thought": "", "action": next_action, "payload": {}}
            fu_result["action"] = next_action  # hard-enforce

            fu_action = next_action
            if fu_action in ("SEND_EMAIL", "SAVE_FILE", "SCHEDULE_MEETING", "OPEN_APP"):
                try:
                    fu_payload = json.loads(fu_result["payload"]) if isinstance(fu_result["payload"], str) else fu_result["payload"]
                except (json.JSONDecodeError, TypeError):
                    fu_payload = {"content": fu_result.get("payload", carry_text)}
                if fu_action == "SAVE_FILE":
                    fname = fu_payload.get("filename", "")
                    if fname.lower().endswith(".txt"):
                        fu_payload["filename"] = fname[:-4] + ".docx"
                    if not fu_payload.get("content"):
                        fu_payload["content"] = carry_text
                if fu_action == "SEND_EMAIL" and not fu_payload.get("body"):
                    fu_payload["body"] = carry_text
                yield _sse({
                    "step": "action_card",
                    "thought": _public_thought_for_action(fu_action),
                    "action": fu_action,
                    "payload": fu_payload,
                })
            else:
                carry_text = str(fu_result.get("payload", ""))
                yield _sse({"step": "result", **fu_result})

        yield _sse({"step": "timing", "text": _elapsed_text(start_time)})

    except json.JSONDecodeError:
        logger.error("Gemini returned non-JSON after retry: %s", response.text[:200])
        yield _sse({"step": "heal", "text": "AI self-correction did not fully resolve — presenting raw response."})
        yield _sse({
            "step": "result",
            "thought": "Response could not be structured after retry.",
            "action": "DRAFT_CONTENT",
            "payload": response.text,
        })
        yield _sse({"step": "timing", "text": _elapsed_text(start_time)})
    except Exception as e:
        logger.exception("Agent error")
        yield _sse({"step": "error", "text": f"Agent error: {e}"})
