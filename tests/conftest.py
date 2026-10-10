"""Shared pytest configuration and fixtures.

Two environment quirks are handled here so individual tests stay ordinary.

1. **Temporary files.** pytest's default base temp directory is not writable
   under the confined file sandbox, and neither is creating *subdirectories*
   inside the repository — only flat files in an existing writable directory are
   allowed. Tests therefore use the ``temp_db`` fixture, which hands out a unique
   file directly under the gitignored ``.cache/``.
2. **Bytecode cache.** ``__pycache__`` inside the source tree is not writable
   either, and a stale ``.pyc`` silently masks edits, so Python is pointed at a
   cache prefix under ``.cache/``.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"

# Make `import lol_agent...` and `import apps.api` work without an install.
for extra in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def pytest_configure(config: pytest.Config) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # A directory tree, not a nested temp dir: subdirectory creation inside the
    # repository is denied, so basetemp is kept to a single level that pytest can
    # still create lazily.
    config.option.basetemp = str(CACHE_ROOT)
    pycache = CACHE_ROOT / "pycache"
    pycache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache))


@pytest.fixture()
def temp_path():
    """A unique file path under .cache/, removed when the test finishes.

    Subdirectories cannot be created here, so this deliberately returns a flat
    file path rather than a temp directory.
    """
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"test-{uuid.uuid4().hex[:12]}.tmp"
    try:
        yield path
    finally:
        for attempt in range(5):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:  # Windows may hold the handle briefly
                import time

                time.sleep(0.1 * (attempt + 1))


@pytest.fixture()
def temp_db(temp_path):
    """A unique SQLite database path under .cache/."""
    return temp_path.with_suffix(".db")

