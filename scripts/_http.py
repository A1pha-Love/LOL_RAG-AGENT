"""Shared HTTP helpers for the local-server check scripts.

Two things are handled here once instead of in every script:

* **Proxy bypass.** This machine runs a system proxy and Python reads it from the
  registry via ``getproxies()``. ``urllib`` does not honour Windows' ProxyOverride
  list, so requests to 127.0.0.1 would be routed through the proxy and fail
  intermittently with WinError 10061. An empty ``ProxyHandler`` keeps loopback
  traffic direct.
* **The skip convention.** A check that cannot run (no server) exits with code 77,
  which ``run_checks.py`` reports as SKIP rather than PASS — a check that did not run
  must never look like one that passed.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Imported by scripts that may or may not have put src/ on the path already.
_SRC = str(PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEFAULT_BASE = os.getenv("LOL_AGENT_BASE_URL", "http://127.0.0.1:8000")

# Exit code run_checks.py treats as "skipped", not "passed".
SKIP_EXIT_CODE = 77

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def redact(message: str) -> str:
    """Replace the API key if it ever reaches an error message."""
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    return message.replace(api_key, "[已隐藏]") if api_key else message


def request(method: str, url: str, payload: dict | None = None, *, timeout: int = 30):
    """Perform one request and return (status, content_type, body_text)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with OPENER.open(req, timeout=timeout) as response:
            return (
                response.status,
                response.headers.get("content-type", ""),
                response.read().decode("utf-8", "replace"),
            )
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("content-type", ""), exc.read().decode("utf-8", "replace")


def get(url: str, *, timeout: int = 30):
    return request("GET", url, None, timeout=timeout)


def post(url: str, payload: dict | None = None, *, timeout: int = 30):
    return request("POST", url, payload, timeout=timeout)


def put(url: str, payload: dict, *, timeout: int = 30):
    return request("PUT", url, payload, timeout=timeout)


def delete(url: str, *, timeout: int = 30):
    return request("DELETE", url, None, timeout=timeout)


def json_body(body: str) -> Any:
    """Parse a response body, returning None when it is not JSON."""
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def server_reachable(base: str = DEFAULT_BASE, *, timeout: float = 1.0) -> bool:
    """Whether the local API is listening, checked without an HTTP round-trip."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def skip(base: str = DEFAULT_BASE) -> int:
    """Report that the check could not run, and return the skip exit code."""
    print(f"SKIP: 服务器未运行（{base}）")
    return SKIP_EXIT_CODE


# Substrings that identify "the account cannot call the model right now" for reasons
# outside the project's control: no credential, a rejected credential, exhausted
# quota, or an unpaid account. Deliberately narrow — a malformed request is a defect
# and must still fail loudly.
#
# Credential failures belong here rather than being reported as quota problems. A
# fresh clone has no key at all, and reporting that as "quota exhausted" sent a reader
# looking in the wrong place. The two are now worded differently on purpose.
UNAVAILABLE_MARKERS = (
    # billing / quota
    "free quota exhausted",
    "freetieronly",
    "allocationquota",
    "insufficient balance",
    "arrearage",
    # credential rejected by the provider
    "invalidapikey",
    "invalid_api_key",
    "incorrect api key",
    "authenticationerror",
    "unauthorized",
    "access_denied",
)

# Kept as an alias: run_checks.py imports this name to classify a failed step's output.
QUOTA_MARKERS = UNAVAILABLE_MARKERS


def _probe_endpoint(path: str, payload: dict, *, api_key: str) -> str | None:
    """Return a reason when one endpoint is refused for a quota or credential reason."""
    from lol_agent.config import chat_base_url

    request = urllib.request.Request(
        f"{chat_base_url()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with OPENER.open(request, timeout=25) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        if any(marker in body.lower() for marker in UNAVAILABLE_MARKERS):
            return f"{path} HTTP {exc.code}: {body[:80]}"
    except Exception:  # noqa: BLE001
        # Network problems are not quota problems; let the real check report them.
        return None
    return None


def embedding_unavailable_reason() -> str | None:
    """Why embeddings cannot be used, or None when they work.

    Reports a missing credential and a refused credential as themselves, not as a
    quota problem. Conflating the three previously produced a message that sent a
    reader to check billing when the actual state was "no API key configured".
    """
    from lol_agent.config import require_api_key
    from lol_agent.knowledge import EMBEDDING_DIMENSIONS

    try:
        api_key = require_api_key()
    except RuntimeError as exc:
        return f"未配置 API 凭据（{exc}）"
    reason = _probe_endpoint(
        "/embeddings",
        {
            "model": os.getenv("QWEN_EMBEDDING_MODEL", "qwen3.7-text-embedding").strip(),
            "input": "探测",
            "dimensions": EMBEDDING_DIMENSIONS,
        },
        api_key=api_key,
    )
    return f"Embedding 端点不可用（{reason}）" if reason else None


def chat_unavailable_reason() -> str | None:
    """Why chat cannot be used, or None when it works.

    Separate from embeddings because the two are billed independently: an account can
    embed while chat is refused. A combined probe would either skip checks that could
    have run, or let chat-dependent checks fail misleadingly.
    """
    from lol_agent.config import require_api_key

    try:
        api_key = require_api_key()
    except RuntimeError as exc:
        return f"未配置 API 凭据（{exc}）"
    reason = _probe_endpoint(
        "/chat/completions",
        {
            "model": os.getenv("QWEN_CHAT_MODEL", "qwen3.8-flash").strip(),
            "messages": [{"role": "user", "content": "OK"}],
            "max_tokens": 1,
        },
        api_key=api_key,
    )
    return f"Chat 端点不可用（{reason}）" if reason else None


# Names kept for existing callers (run_checks.py, check scripts). They now report a
# missing credential and a refused credential distinctly, instead of labelling every
# unusable state as a quota problem.
embedding_quota_exhausted = embedding_unavailable_reason
chat_quota_exhausted = chat_unavailable_reason


def model_quota_exhausted() -> str | None:
    """Reason the model API is unusable in any respect, or None."""
    return embedding_unavailable_reason() or chat_unavailable_reason()


class Report:
    """Collects PASS/FAIL lines so a script can report every result at the end."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.failures: list[str] = []
        self.checked = 0

    def section(self, name: str) -> None:
        print()
        print(f"=== {name} ===")

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.checked += 1
        marker = "PASS" if ok else "FAIL"
        suffix = f"  {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not ok:
            self.failures.append(label)
        return ok

    def finish(self) -> int:
        print()
        print("=" * 72)
        if self.failures:
            print(f"RESULT: FAIL — {len(self.failures)}/{self.checked} 项未通过")
            for item in self.failures:
                print(f"  - {item}")
            return 1
        print(f"RESULT: PASS — {self.checked}/{self.checked} 项通过")
        return 0


def use_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
