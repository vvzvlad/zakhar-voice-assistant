"""Per-device conversation context persistence."""

import os
from datetime import datetime

from loguru import logger


def append_context(context_path: str, user_text: str, assistant_text: str) -> None:
    """Append the user and assistant messages to the per-device context file.

    The path is supplied by the caller (the pipeline passes data/context_<name>.txt).
    If the file's last modification time is older than 60 seconds, clear it first.
    """
    try:
        # Decide whether to truncate the file based on its age
        truncate_file = False
        try:
            if os.path.exists(context_path):
                mtime = os.path.getmtime(context_path)
                age_seconds = datetime.now().timestamp() - mtime
                if age_seconds > 60:
                    truncate_file = True
        except OSError:
            # If we cannot stat the file for any reason, prefer recreating it
            truncate_file = True

        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        mode = "w" if truncate_file else "a"
        with open(context_path, mode, encoding="utf-8") as f:
            f.write(f"USER: {user_text}\n")
            f.write(f"GLADOS: {assistant_text}\n")
    except OSError as e:
        logger.error(f"Failed to write to {context_path}: {str(e)}")
