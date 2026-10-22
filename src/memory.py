"""The assistant's long-term memory: short facts it saved with its memory tool
(src/builtin_mcp/memory.py), substituted into the system prompt at every request.

A JSON list of strings in data/memory.json. Every call reads the file anew, so a hand
edit of it applies on the next request; writes are atomic (unique temp file in the same
directory + fsync + os.replace), as in src/config_store.py.
"""

import json
import os
import tempfile


def _norm(fact: str) -> str:
    """A fact as stored: stripped, without the «- » it is listed with in the prompt."""
    return fact.strip().removeprefix("- ").strip()


class MemoryStore:
    def __init__(self, path: str):
        self.path = path

    def facts(self) -> list[str]:
        """The saved facts in the order they were added; a missing file is no facts."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def add(self, fact: str) -> bool:
        """Append `fact` (normalized); False when it is already saved, ignoring case."""
        fact = _norm(fact)
        facts = self.facts()
        if fact.casefold() in {f.casefold() for f in facts}:
            return False
        self._write(facts + [fact])
        return True

    def remove(self, fact: str) -> bool:
        """Drop the fact whose text equals `fact` (normalized), ignoring case; False
        when there is none."""
        key = _norm(fact).casefold()
        facts = self.facts()
        kept = [f for f in facts if f.casefold() != key]
        if len(kept) == len(facts):
            return False
        self._write(kept)
        return True

    def _write(self, facts: list[str]) -> None:
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        # Temp file in the SAME directory, so os.replace stays atomic (same fs).
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=os.path.basename(self.path) + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(facts, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
