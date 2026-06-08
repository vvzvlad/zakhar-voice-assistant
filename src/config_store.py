"""Atomic JSON config storage (data/config.json).

Single source of truth for the config document. Writes are atomic (temp file +
fsync + os.replace), keep a .bak of the previous file, and chmod the result to
0o600 because secrets live in this file as plain SecretStr values.
"""

import json
import os
import shutil

DEFAULT_PATH = "data/config.json"


def load(path: str = DEFAULT_PATH) -> dict:
    """Return the parsed config document, or {} if the file is missing.

    Invalid JSON propagates (json.JSONDecodeError) — a corrupt config must fail
    loudly rather than silently reset to defaults.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save(doc: dict, path: str = DEFAULT_PATH) -> None:
    """Atomically write the config document to `path`.

    Steps: create the parent dir if needed, write to <path>.tmp and fsync it, copy
    the existing file to <path>.bak (if any), then os.replace(tmp, path). The final
    file is chmod'd to 0o600 since it may contain secrets.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")

    os.replace(tmp, path)
    os.chmod(path, 0o600)
