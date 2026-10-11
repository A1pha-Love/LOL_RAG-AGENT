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


def get(url: str, *, timeout: int = 30) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.headers.get("content-type", ""), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("content-type", ""), exc.read().decode("utf-8", "replace")


def post_json(url: str, payload: dict, *, timeout: int = 30) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8077")
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

        data = json.dumps({"question": "哪个英雄被禁用次数最多？", "session_id": session_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/sessions/ask", data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        counts: dict[str, int] = {}
        chunks_seen = 0
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                kind = event.get("type", "?")
                counts[kind] = counts.get(kind, 0) + 1
                if kind == "retrieval":
                    chunks_seen = len(event.get("results") or [])
                if kind == "error":
                    print(f"  [错误] {event.get('message')}")
                    failures.append("turn error")

        print(f"  事件统计: {counts}")
        print(f"  检索命中分块: {chunks_seen}")
        print(f"  耗时 {time.perf_counter() - started:.1f}s")

        # The inspector is useless without these three.
        for required, label in (("rewrite", "检索查询"), ("tools", "工具调用"), ("retrieval", "命中分块"), ("done", "完成")):
            if not counts.get(required):
                print(f"  !! 缺少 {required} 事件（{label}）")
                failures.append(f"missing {required} event")
        if chunks_seen == 0:
            print("  !! retrieval 事件未携带任何分块，检索面板将是空的")
            failures.append("no chunks in retrieval event")

    print()
    if failures:
        print(f"RESULT: FAIL -> {failures}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
