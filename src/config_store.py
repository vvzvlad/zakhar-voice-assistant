"""Atomic JSON config storage (data/config.json).

Single source of truth for the config document. Writes are atomic (unique temp
file + fsync + os.replace + directory fsync), keep a .bak of the previous file, and
chmod the result to 0o600 because secrets live in this file as plain SecretStr values.
"""

import json
import os
import shutil
import tempfile

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

    Steps: create the parent dir if needed, write JSON to a UNIQUE temp file in the
    same directory and fsync it, copy the existing file to <path>.bak (if any), then
    os.replace(tmp, path) and fsync the containing directory so the rename survives a
    power failure. The final file is chmod'd to 0o600 since it may contain secrets.

    The temp file is unique (tempfile.mkstemp) so two concurrent saves cannot race on
    a shared `<path>.tmp` name. On any failure the temp file is removed (no leftover
    garbage) and the existing good file is left untouched, then the error is re-raised.
    """
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)

    # Unique temp file in the SAME directory so os.replace stays atomic (same fs) and
    # concurrent saves never collide on a fixed name.
    basename = os.path.basename(path)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=basename + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Preserve the OLD content as .bak before it is replaced.
        if os.path.exists(path):
            shutil.copy2(path, path + ".bak")

        os.replace(tmp, path)
        os.chmod(path, 0o600)

        # Persist the rename metadata by fsyncing the directory. Best-effort: some
        # platforms cannot open a directory (no O_DIRECTORY), so guard against OSError.
        try:
            dir_fd = os.open(parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        # On any failure remove the temp file (if os.replace did not consume it) and
        # re-raise; the existing good file is never touched.
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
