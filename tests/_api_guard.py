"""Shared pytest fixtures and skip guards.

`tests/conftest.py` handles environment quirks. This module handles one external
dependency: tests that exercise real retrieval need the embedding API, and when its
quota is exhausted they fail with a confusing HTTP error rather than telling you the
check could not run.

A failing test that is really "the API quota ran out" is worse than a skipped one:
it hides genuine regressions behind noise. `requires_embedding` probes the API once
and skips with an explicit reason when it is unavailable for environmental reasons.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.config import require_api_key  # noqa: E402
from lol_agent.embeddings import build_embeddings  # noqa: E402

# Set LOL_AGENT_SKIP_API_TESTS=1 to skip every API-dependent test without probing.
SKIP_ENV = "LOL_AGENT_SKIP_API_TESTS"

_MARKERS = (
    "free quota exhausted",
    "freetieronly",
    "allocationquota",
    "insufficient balance",
    "arrearage",
)

_probe_cache: dict[str, str | None] = {}


def _probe() -> str | None:
    """Return None when embeddings work, otherwise the reason they do not.

    Only environmental causes are reported; a bad query would be a bug and must
    still fail loudly, which is why the message list is narrow.
    """
    if "reason" in _probe_cache:
        return _probe_cache["reason"]

    if os.getenv(SKIP_ENV, "").strip() not in ("", "0", "false"):
        _probe_cache["reason"] = f"{SKIP_ENV} is set"
        return _probe_cache["reason"]

    try:
        require_api_key()
    except RuntimeError as exc:
        _probe_cache["reason"] = f"DASHSCOPE_API_KEY 不可用：{exc}"
        return _probe_cache["reason"]

    try:
        build_embeddings().embed_query("探测")
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if any(marker in message for marker in _MARKERS):
            _probe_cache["reason"] = f"Embedding 额度已耗尽：{str(exc)[:120]}"
            return _probe_cache["reason"]
        # A genuine failure: let the test run and surface it.
        _probe_cache["reason"] = None
        return None

    _probe_cache["reason"] = None
    return None


def requires_embedding(func=None, *, reason: str = ""):
    """Skip a test when the embedding API is unavailable for environmental reasons."""

    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            unavailable = _probe()
            if unavailable:
                pytest.skip(f"需要 Embedding API：{unavailable}{(' — ' + reason) if reason else ''}")
            return function(*args, **kwargs)

        return wrapper

    return decorate(func) if func is not None else decorate
