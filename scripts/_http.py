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
