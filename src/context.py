"""Per-device conversation context: a rolling JSONL window of recent messages.

Each line is one JSON object {"role": "user"|"assistant", "content": "<text>"}.
The pipeline loads the recent window to feed it back to the model so it remembers
the last few turns, and appends the new exchange after each run. A conversation
idle longer than ttl_seconds is considered finished and reset to empty, so an
unrelated later request starts fresh. Only the last max_turns * 2 messages are
kept: user+assistant exchanges, plus user-role notes of what the device said
unprompted (append_spoken), so two user-role messages may follow each other and the
trimmed window may start with an assistant message. A note of the tools an answer
called (append_context) rides along with that answer: it is not counted, and it is
dropped with the answer.
"""

import json
import os
import time

from loguru import logger

# Opens the note append_context writes after an answer that called tools.
_TOOL_NOTE = "[Этот ответ ассистент дал, вызвав инструменты: "


def _trim(messages: list[dict], max_turns: int) -> list[dict]:
    """The last max_turns * 2 messages, tool notes not counted; a tool note whose
    answer was cut off is dropped too."""
    kept, counted = [], 0
    for msg in reversed(messages):
        if not msg["content"].startswith(_TOOL_NOTE):
            if counted == max_turns * 2:
                break
            counted += 1
        kept.append(msg)
    kept.reverse()
    while kept and kept[0]["content"].startswith(_TOOL_NOTE):
        kept.pop(0)
    return kept


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
    ignores leftover legacy USER:/GLADOS: lines. The result is trimmed by _trim: the
    last max_turns * 2 messages, tool notes not counted.
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
    return _trim(messages, max_turns)


def append_context(
    context_path: str,
    user_text: str,
    assistant_text: str,
    max_turns: int = 5,
    ttl_seconds: int = 300,
    calls: list[dict] | None = None,
) -> None:
    """Append the user and assistant messages to the per-device context file, and,
    when the answer ran tools (`calls`: the rounds' {name, args, ...} dicts), a note
    naming them. Failures are logged and never raised; the pipeline relies on that.

    The history holds text only, so without the note an answer like «Напоминание
    поставлено» reads as if saying it was enough, and after a few such turns the model
    answers the next request in words without calling the tool. The note is a
    user-role message in the third person, as append_spoken's, so the model does not
    copy it into its replies."""
    new = [{"role": "user", "content": user_text},
           {"role": "assistant", "content": assistant_text}]
    if calls:
        made = "; ".join(
            c["name"] + "(" + ", ".join(
                f"{k}={json.dumps(v, ensure_ascii=False)}"
                for k, v in (c.get("args") or {}).items() if v is not None) + ")"
            for c in calls)
        new.append({"role": "user", "content": f"{_TOOL_NOTE}{made}]"})
    _append(context_path, new, max_turns, ttl_seconds)


def append_spoken(context_path: str, text: str, max_turns: int = 5, ttl_seconds: int = 300) -> None:
    """Append a note that the device said `text` unprompted (an agent's say, a
    reminder, a handed-off task's answer), so a follow-up question sees it. Failures
    are logged and never raised.

    The note is a user-role message in the third person, not an assistant message:
    given a lone assistant message, the model answers "what was that?" with its
    previous reply and ignores the phrase."""
    note = f"[На колонке была произнесена фраза: «{text}»]"
    _append(context_path, [{"role": "user", "content": note}], max_turns, ttl_seconds)


def _append(context_path: str, new: list[dict], max_turns: int, ttl_seconds: int) -> None:
    """Load the current window (which already returns [] when missing/stale/unreadable
    and skips legacy lines), append `new`, trim it with _trim,
    and rewrite the whole file as JSONL. The full rewrite is intentional: it applies
    trimming and the staleness reset in one place."""
    messages = load_context(context_path, max_turns=max_turns, ttl_seconds=ttl_seconds)
    messages.extend(new)
    messages = _trim(messages, max_turns)

    try:
        os.makedirs(os.path.dirname(context_path) or ".", exist_ok=True)
        with open(context_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error(f"Failed to write to {context_path}: {str(e)}")
