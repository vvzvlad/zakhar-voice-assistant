"""Single source of truth for the application version.

The version is derived from the Git tag. Resolution order:

1. `git describe --tags --always --dirty` run from the repository root — used
   when running from a Git checkout (development).
2. A build-stamped `_version.txt` written next to this file at image build time
   (see the Dockerfile `ARG VERSION`) — used inside the Docker image, where
   `.git` is stripped by .dockerignore and the `git` binary is absent.
3. A hardcoded fallback, so the app always reports *some* version.
"""

import subprocess
from pathlib import Path

_FALLBACK = "0.0.0+unknown"
_ROOT = Path(__file__).resolve().parent.parent
_STAMP = Path(__file__).resolve().parent / "_version.txt"


def _from_git() -> "str | None":
    """Return `git describe` output, or None when not in a usable Git checkout."""
    # No repository here → skip the subprocess entirely. This avoids a doomed
    # spawn inside the Docker image (no `.git`) and stops git from walking up to
    # an unrelated parent repository. `.git` may be a dir (normal) or a file
    # (worktrees/submodules), so `exists()` covers both.
    if not (_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _from_stamp() -> "str | None":
    """Return the build-stamped version file contents, or None when absent/empty."""
    try:
        value = _STAMP.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _resolve_version() -> str:
    """Resolve the version from Git, then a build stamp, then a constant fallback."""
    return _from_git() or _from_stamp() or _FALLBACK


__version__ = _resolve_version()
