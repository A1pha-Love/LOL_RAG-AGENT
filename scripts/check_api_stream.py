"""Live check of the streaming API: health, session CRUD, and a real SSE turn.

    python scripts/check_api_stream.py --base http://127.0.0.1:8077

Verifies HTTP framing (status, content type) and that the SSE stream arrives
incrementally rather than as one buffered blob — which is the failure mode that
makes a UI look frozen while the model is actually answering.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def request(method: str, url: str, payload: dict | None = None, *, timeout: int = 30):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            return response.status, response.headers, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read().decode("utf-8", "replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8077")
    parser.add_argument("--question", default="英雄 ID 157 被禁用了多少场？")
    parser.add_argument("--follow-up", default="那他的胜率呢？")
    parser.add_argument("--skip-llm", action="store_true", help="do not run the answer turns")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    failures: list[str] = []

    print("=== /api/health ===")
    status, _, body = request("GET", f"{base}/api/health")
    print(f"  [{status}] {body[:200]}")
    if status != 200:
        failures.append("health")

    print()
    print("=== 会话 CRUD ===")
    status, _, body = request("POST", f"{base}/api/sessions", {"title": ""})
    created = json.loads(body)["session_id"] if status == 201 else None
    print(f"  [{status}] 新建会话 -> {created}")
    if status != 201:
        failures.append("create session")

    status, _, body = request("GET", f"{base}/api/sessions")
    listed = json.loads(body)["sessions"]
    print(f"  [{status}] 会话数 {len(listed)}，最近一条 {listed[0]['session_id'] if listed else None}")
    if status != 200 or not listed:
        failures.append("list sessions")

    status, _, body = request("GET", f"{base}/api/sessions/{created}")
    print(f"  [{status}] 读取会话，turns={len(json.loads(body).get('turns', []))}")
    if status != 200:
        failures.append("get session")

    status, _, _ = request("GET", f"{base}/api/sessions/does-not-exist")
    print(f"  [{status}] 未知会话（应为 404）")
    if status != 404:
        failures.append("404 handling")

    if args.skip_llm:
        print("\n（--skip-llm：跳过实际问答）")
    else:
        for label, question, session in (
            ("第 1 轮", args.question, created),
            ("第 2 轮（追问）", args.follow_up, created),
        ):
            print()
            print(f"=== SSE {label}: {question} ===")
            payload = {"question": question, "session_id": session}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{base}/api/sessions/ask",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            events: dict[str, int] = {}
            first_delta_at: float | None = None
            answer_parts: list[str] = []
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    print(f"  [status {response.status}] content-type={response.headers.get('content-type')}")
                    if not response.headers.get("content-type", "").startswith("text/event-stream"):
                        failures.append(f"{label} content-type")
                    for raw in response:
                        line = raw.decode("utf-8", "replace").rstrip("\n")
                        if not line.startswith("data:"):
                            continue
                        event = json.loads(line[5:].strip())
                        kind = event.get("type", "?")
                        events[kind] = events.get(kind, 0) + 1
                        if kind == "delta":
                            if first_delta_at is None:
                                first_delta_at = time.perf_counter() - started
                            answer_parts.append(event.get("text", ""))
                        elif kind == "rewrite":
                            print(f"  [检索查询] {event.get('query')}")
                            if event.get("fallback"):
                                print(f"  [改写回退] {event['fallback']}")
                        elif kind == "tools":
                            print(f"  [工具] {', '.join(event.get('names', []))}")
                        elif kind == "done":
                            print(f"  [完成] {event.get('seconds')}s, 工具={event.get('tool_calls')}")
                        elif kind == "error":
                            print(f"  [错误] {event.get('message')}")
                            failures.append(f"{label} error event")
            except Exception as exc:  # noqa: BLE001
                print(f"  流式请求失败：{type(exc).__name__}: {exc}")
                failures.append(f"{label} stream")
                continue

            elapsed = time.perf_counter() - started
            answer = "".join(answer_parts)
            print(f"  事件统计: {events}")
            print(f"  首个增量到达: {first_delta_at:.2f}s | 总耗时 {elapsed:.2f}s")
            print(f"  答案片段长度: {len(answer)}")
            print(f"  答案预览: {answer[:160].replace(chr(10), ' ')}")

            if events.get("delta", 0) < 2:
                print("  !! 增量事件过少，可能被缓冲成整块返回")
                failures.append(f"{label} not incremental")
            if first_delta_at is not None and first_delta_at > elapsed * 0.8:
                print("  !! 首个增量几乎在结束时才到，流式未生效")
                failures.append(f"{label} buffered")

    status, _, body = request("GET", f"{base}/api/sessions/{created}")
    turns = json.loads(body)["turns"]
    print()
    print(f"=== 会话落盘校验 === turns={len(turns)}")
    for turn in turns:
        marker = turn.get("retrieval_query") or ""
        print(f"  [{turn['role']:9}] {turn['content'][:44]!r} 检索={marker[:44]!r}")
    if not args.skip_llm and len(turns) != 4:
        failures.append(f"persistence (expected 4 turns, got {len(turns)})")

    print()
    if failures:
        print(f"RESULT: FAIL -> {failures}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
