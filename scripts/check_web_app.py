"""End-to-end check of the built frontend served by the backend.

Verifies that the SPA is actually served (not just the API), that hashed assets
resolve, that the API routes still win over the catch-all, and that one real
streamed turn produces the full event sequence the UI depends on.

    python scripts/check_web_app.py --base http://127.0.0.1:8077
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# This machine runs a system proxy (127.0.0.1:7897) and Python reads it from the
# registry via getproxies(). urllib does NOT honour Windows' ProxyOverride list,
# so requests to 127.0.0.1 were being sent through the proxy and intermittently
# failed with WinError 10061. An empty ProxyHandler keeps loopback traffic direct.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def urlopen(url, *, timeout: int = 30, data: bytes | None = None):
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    return _OPENER.open(request, timeout=timeout)


def get(url: str, *, timeout: int = 30) -> tuple[int, str, str]:
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.status, response.headers.get("content-type", ""), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("content-type", ""), exc.read().decode("utf-8", "replace")


def post_json(url: str, payload: dict, *, timeout: int = 30) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    try:
        with urlopen(url, data=data, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-llm", action="store_true", help="do not run a real answer turn")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    failures: list[str] = []

    print("=== 静态资源 ===")
    status, ctype, body = get(f"{base}/")
    ok = status == 200 and "html" in ctype and "<div id=\"app\">" in body
    print(f"  [{'PASS' if ok else 'FAIL'}] GET /  -> {status} {ctype}")
    if not ok:
        failures.append("index.html")

    assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', body)
    print(f"  页面引用的资源: {assets}")
    for asset in assets:
        status, ctype, _ = get(f"{base}{asset}")
        good = status == 200
        print(f"  [{'PASS' if good else 'FAIL'}] GET {asset} -> {status} {ctype}")
        if not good:
            failures.append(asset)

    print()
    print("=== API 路由优先于 SPA 回退 ===")
    status, ctype, body = get(f"{base}/api/health")
    ok = status == 200 and "json" in ctype and '"status"' in body
    print(f"  [{'PASS' if ok else 'FAIL'}] GET /api/health -> {status} {ctype}")
    if not ok:
        failures.append("health shadowed by catch-all")

    status, ctype, body = get(f"{base}/some/client/route")
    ok = status == 200 and "html" in ctype
    print(f"  [{'PASS' if ok else 'FAIL'}] GET /some/client/route -> {status} {ctype}（应回退到 index.html）")
    if not ok:
        failures.append("spa fallback")

    if args.skip_llm:
        print("\n（--skip-llm：跳过实际问答）")
    else:
        print()
        print("=== 真实 SSE 一轮（前端依赖的完整事件序列）===")
        status, session_body = post_json(f"{base}/api/sessions", {})
        session_id = json.loads(session_body)["session_id"]
        print(f"  会话 {session_id}")

        # A definition question, not a statistic: it reliably makes the model
        # consult the knowledge base, which is what the retrieval panel shows.
        question = "Ban 率按什么口径计算？同一英雄同场多次禁用怎么算？"
        print(f"  问题 {question}")
        data = json.dumps({"question": question, "session_id": session_id}).encode("utf-8")
        started = time.perf_counter()
        counts: dict[str, int] = {}
        chunks_seen = 0
        tool_calls: list[str] = []
        with urlopen(f"{base}/api/sessions/ask", data=data, timeout=180) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                kind = event.get("type", "?")
                counts[kind] = counts.get(kind, 0) + 1
                if kind == "retrieval":
                    chunks_seen = len(event.get("results") or [])
                if kind == "done":
                    tool_calls = event.get("tool_calls") or []
                if kind == "error":
                    print(f"  [错误] {event.get('message')}")
                    failures.append("turn error")

        print(f"  事件统计: {counts}")
        print(f"  工具调用: {tool_calls}")
        print(f"  检索命中分块: {chunks_seen}")
        print(f"  耗时 {time.perf_counter() - started:.1f}s")

        # The answer never needs the knowledge base for a pure statistic, so the
        # retrieval event is only required when the model actually consulted it.
        for required in ("rewrite", "tools", "done"):
            if not counts.get(required):
                print(f"  !! 缺少 {required} 事件")
                failures.append(f"missing {required} event")

        if "search_project_knowledge" in tool_calls:
            if not counts.get("retrieval"):
                print("  !! 调用了知识检索但没有 retrieval 事件")
                failures.append("missing retrieval event")
            elif chunks_seen == 0:
                print("  !! retrieval 事件未携带任何分块，检索面板将是空的")
                failures.append("no chunks in retrieval event")
        else:
            print("  注：本轮模型未调用知识检索，因此不要求 retrieval 事件")

    print()
    if failures:
        print(f"RESULT: FAIL -> {failures}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
