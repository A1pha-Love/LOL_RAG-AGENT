"""Shared pytest fixtures and skip guards.

`tests/conftest.py` handles environment quirks. This module handles one external
dependency: tests that exercise retrieval or the Agent need the Qwen API, and when its
quota is exhausted they fail with a confusing HTTP error rather than telling you the
check could not run.

A failing test that is really "the API quota ran out" is worse than a skipped one:
it hides genuine regressions behind noise.

Embeddings and chat are billed separately and can be exhausted independently, so they
are probed separately — an account can embed while chat is refused. `requires_embedding`
covers retrieval tests; `requires_chat` covers anything that runs the Agent.
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

# Set LOL_AGENT_SKIP_API_TESTS=1 to skip every API-dependent test without probing.
SKIP_ENV = "LOL_AGENT_SKIP_API_TESTS"

# Reasons the API cannot be used that are environmental rather than defects. Kept
# deliberately narrow: a malformed request is a bug and must still fail loudly.
#
# Authentication failures belong here. A fresh clone has only `.env.example`, whose key
# is empty, so `conftest` injects a placeholder; the provider then answers 401. Without
# these markers the skip guard would not recognise that as "cannot run", and the suite
# would FAIL on a clean checkout — which is exactly what happened before they were
# added.
_MARKERS = (
    "free quota exhausted",
    "freetieronly",
    "allocationquota",
    "insufficient balance",
    "arrearage",
    # authentication / authorization failures
    "invalidapikey",
    "invalid_api_key",
    "incorrect api key",
    "authenticationerror",
    "unauthorized",
    "401",
    # an explicit statement that the credential cannot call this model
    "model not found",
    "permissiondenied",
    "access_denied",
)

_probe_cache: dict[str, str | None] = {}


def _probe(kind: str) -> str | None:
    """Return None when the endpoint works, otherwise the reason it does not.

    Only environmental causes are reported; a bad request would be a bug and must
    still fail loudly, which is why the marker list is narrow.
    """
    if kind in _probe_cache:
        return _probe_cache[kind]

    if os.getenv(SKIP_ENV, "").strip() not in ("", "0", "false"):
        _probe_cache[kind] = f"{SKIP_ENV} is set"
        return _probe_cache[kind]

    try:
        require_api_key()
    except RuntimeError as exc:
        _probe_cache[kind] = f"DASHSCOPE_API_KEY 不可用：{exc}"
        return _probe_cache[kind]

    try:
        if kind == "embedding":
            from lol_agent.embeddings import build_embeddings

            build_embeddings().embed_query("探测")
        else:
            import os as _os

            from langchain_core.messages import HumanMessage
            from langchain_openai import ChatOpenAI

            from lol_agent.config import chat_base_url

            model = ChatOpenAI(
                model=_os.getenv("QWEN_CHAT_MODEL", "qwen3.8-flash").strip(),
                api_key=require_api_key(),
                base_url=chat_base_url(),
                max_tokens=1,
            )
            model.invoke([HumanMessage(content="OK")])
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if any(marker in message for marker in _MARKERS):
            label = "Embedding" if kind == "embedding" else "Chat"
            _probe_cache[kind] = f"{label} 额度已耗尽：{str(exc)[:110]}"
            return _probe_cache[kind]
        # A genuine failure: let the test run and surface it.
        _probe_cache[kind] = None
        return None

    _probe_cache[kind] = None
    return None


def _guard(kind: str, reason: str):
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            unavailable = _probe(kind)
            if unavailable:
                pytest.skip(f"需要 {kind} API：{unavailable}{(' — ' + reason) if reason else ''}")
            return function(*args, **kwargs)

        return wrapper

    return decorate


def requires_embedding(func=None, *, reason: str = ""):
    """Skip when the embedding API is unavailable for environmental reasons."""
    decorator = _guard("embedding", reason)
    return decorator(func) if func is not None else decorator


def requires_chat(func=None, *, reason: str = ""):
    """Skip when the chat API is unavailable for environmental reasons.

    Separate from `requires_embedding` because the two quotas are independent: a
    retrieval test can run while Agent tests cannot.
    """
    decorator = _guard("chat", reason)
    return decorator(func) if func is not None else decorator
