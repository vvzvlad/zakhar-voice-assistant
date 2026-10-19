"""Per-device conversation context: a rolling JSONL window of recent messages.

Each line is one JSON object {"role": "user"|"assistant", "content": "<text>"}.
The pipeline loads the recent window to feed it back to the model so it remembers
the last few turns, and appends the new exchange after each run. A conversation
idle longer than ttl_seconds is considered finished and reset to empty, so an
unrelated later request starts fresh. Only the last max_turns * 2 messages are
kept: user+assistant exchanges, plus lone assistant replies that arrived later
(append_reply), so the window may start with an assistant message.
"""

import json
import os
import time

from loguru import logger


def _is_stale(context_path: str, ttl_seconds: int) -> bool:
    """Return True if the context file is missing/unreadable or idle past the TTL.

    Staleness is derived from the file mtime: a conversation untouched for longer
    than ttl_seconds is treated as finished.
    """
    try:
        age_seconds = time.time() - os.path.getmtime(context_path)
    except OSError:
        return True
    return age_seconds > ttl_seconds


def load_context(context_path: str, max_turns: int = 5, ttl_seconds: int = 300) -> list[dict]:
    """Load the recent conversation window as a list of {"role", "content"} dicts.

    Returns [] when the file is missing or stale (idle past the TTL). Blank lines
    and any line that is not a valid JSON message are skipped, which gracefully
    ignores leftover legacy USER:/GLADOS: lines. The result is trimmed to the last
    max_turns * 2 messages.
    """
    if not os.path.exists(context_path) or _is_stale(context_path, ttl_seconds):
        return []

    messages: list[dict] = []
    try:
        with open(context_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    continue  # skip malformed / legacy non-JSON lines
                if not isinstance(obj, dict):
                    continue
                role = obj.get("role")
                content = obj.get("content")
                if role in ("user", "assistant") and isinstance(content, str):
                    messages.append({"role": role, "content": content})
    except OSError as e:
        logger.error(f"Failed to read context {context_path}: {str(e)}")
        return []

    # Defensive trim even if the file grew beyond the configured window.
    return messages[-(max_turns * 2):]


def append_context(
    context_path: str,
    user_text: str,
    assistant_text: str,
    max_turns: int = 5,
    ttl_seconds: int = 300,
) -> None:
    """Append the user and assistant messages to the per-device context file.
    Failures are logged and never raised; the pipeline relies on that."""
    _append(context_path, [{"role": "user", "content": user_text},
                           {"role": "assistant", "content": assistant_text}],
            max_turns, ttl_seconds)


def append_reply(context_path: str, text: str, max_turns: int = 5, ttl_seconds: int = 300) -> None:
    """Append one assistant message that has no user turn of its own: a reply that
    arrived later and was spoken unprompted (a handed-off task's answer), so a
    follow-up question sees it. Failures are logged and never raised."""
    _append(context_path, [{"role": "assistant", "content": text}], max_turns, ttl_seconds)


def _append(context_path: str, new: list[dict], max_turns: int, ttl_seconds: int) -> None:
    """Load the current window (which already returns [] when missing/stale/unreadable
    and skips legacy lines), append `new`, trim to the last max_turns * 2 messages,
    and rewrite the whole file as JSONL. The full rewrite is intentional: it applies
    trimming and the staleness reset in one place."""
    messages = load_context(context_path, max_turns=max_turns, ttl_seconds=ttl_seconds)
    messages.extend(new)
    messages = messages[-(max_turns * 2):]

    try:
        os.makedirs(os.path.dirname(context_path) or ".", exist_ok=True)
        with open(context_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error(f"Failed to write to {context_path}: {str(e)}")
