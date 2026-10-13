"""Shared pytest configuration and fixtures.

Three environment quirks are handled here so individual tests stay ordinary.

1. **API key presence.** Several code paths call ``require_api_key()`` purely to
   assert that *a* key exists — building a client, starting a streamed turn. Those
   assertions are part of the code under test, so the tests must exercise them, but
   they must not demand that a real key is configured: a fresh clone has only
   ``.env.example`` with an empty value, and CI has no key at all. A placeholder is
   injected so those paths are entered; anything that would actually *call* the API
   is gated separately by ``requires_chat`` / ``requires_embedding`` in
   ``_api_guard.py``, which skip when the account cannot be reached.
2. **Temporary files.** pytest's default base temp directory is not writable
   under the confined file sandbox, and neither is creating *subdirectories*
   inside the repository — only flat files in an existing writable directory are
   allowed. Tests therefore use the ``temp_db`` fixture, which hands out a unique
   file directly under the gitignored ``.cache/``.
3. **Bytecode cache.** ``__pycache__`` inside the source tree is not writable
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

# Deliberately not a valid credential: it only has to be a non-empty string so that
# the "is a key configured?" checks pass. Never sent anywhere, because every test that
# would reach the network is skipped by _api_guard when the API is unusable.
PLACEHOLDER_API_KEY = "test-key-not-a-real-credential"


def pytest_configure(config: pytest.Config) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # A directory tree, not a nested temp dir: subdirectory creation inside the
    # repository is denied, so basetemp is kept to a single level that pytest can
    # still create lazily.
    config.option.basetemp = str(CACHE_ROOT)
    pycache = CACHE_ROOT / "pycache"
    pycache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache))


@pytest.fixture(autouse=True)
def _api_key_present():
    """Guarantee ``DASHSCOPE_API_KEY`` is non-empty for the duration of a test.

    A real key in the developer's ``.env`` is left untouched; only a missing one is
    filled in. Without this, ``tests/test_api.py`` and
    ``tests/test_conversation.py`` failed on a fresh clone (and would fail in CI) for
    a reason unrelated to any defect — the turn aborted at the key check instead of
    reaching the code under test.
    """
    previous = os.environ.get("DASHSCOPE_API_KEY")
    if not (previous or "").strip():
        os.environ["DASHSCOPE_API_KEY"] = PLACEHOLDER_API_KEY
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DASHSCOPE_API_KEY", None)
        else:
            os.environ["DASHSCOPE_API_KEY"] = previous


def _remove_with_retry(path: Path) -> None:
    """Delete a file, tolerating Windows holding the handle briefly."""
    import time

    for attempt in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
        except OSError:
            return


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
        _remove_with_retry(path)


@pytest.fixture()
def temp_db():
    """A unique SQLite database path under .cache/, removed when the test finishes.

    This used to derive its path from ``temp_path``, which meant only the ``.tmp``
    path was cleaned up and the ``.db`` file the code created was left behind — one
    leaked file per test that used it, thousands of them over time. Each path is now
    owned by the fixture that hands it out.
    """
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"test-{uuid.uuid4().hex[:12]}.db"
    try:
        yield path
    finally:
        # SQLite in WAL mode also leaves these beside the database.
        for candidate in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            _remove_with_retry(candidate)

