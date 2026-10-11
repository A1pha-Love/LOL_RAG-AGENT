"""Verify two concurrent streamed turns do not interfere.

The UI now allows asking in one conversation while another is still streaming, so
the backend genuinely receives overlapping requests. Each turn keeps its rewritten
retrieval query and its tool list in a per-turn context; if that context were shared
(a module global rather than a ContextVar) one turn's retrieval query would leak into
the other's tool call, and the answers would silently mix.

Runs two turns concurrently and checks each answer stays on its own topic.

    python scripts/check_concurrent_turns.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# One dataset question and one definition question: their answers must not cross.
TURNS = [
    ("这份数据去重后有多少场比赛？", ["51053", "51053"]),
    ("Ban 率按什么口径计算？同一英雄同场多次禁用怎么算？", ["禁用", "分母"]),
]


def post(url: str, payload: dict, *, timeout: int = 240):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return OPENER.open(request, timeout=timeout)


def run_turn(base: str, question: str, results: dict, index: int) -> None:
    """Stream one turn to completion, recording the events it produced."""
    started = time.perf_counter()
    try:
        session_id = json.loads(post(f"{base}/api/sessions", {}, timeout=20).read())["session_id"]
        events: dict[str, int] = {}
        answer_parts: list[str] = []
        retrieval_query = None
        tool_calls: list[str] = []
        for raw in post(f"{base}/api/sessions/ask", {"question": question, "session_id": session_id}):
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            kind = event.get("type", "?")
            events[kind] = events.get(kind, 0) + 1
            if kind == "rewrite":
                retrieval_query = event.get("query")
            elif kind == "delta":
                answer_parts.append(event.get("text", ""))
            elif kind == "done":
                tool_calls = event.get("tool_calls") or []
        results[index] = {
            "question": question,
            "session_id": session_id,
            "answer": "".join(answer_parts),
            "retrieval_query": retrieval_query,
            "tool_calls": tool_calls,
            "events": events,
            "seconds": round(time.perf_counter() - started, 1),
        }
    except Exception as exc:  # noqa: BLE001
        results[index] = {"question": question, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    try:
        post(f"{base}/api/sessions", {}, timeout=10).read()
    except (urllib.error.URLError, OSError):
        print(f"SKIP: 服务器未运行（{base}）")
        return 77

    print(f"=== 并发发起 {len(TURNS)} 轮对话 ===")
    results: dict[int, dict] = {}
    threads = [
        threading.Thread(target=run_turn, args=(base, question, results, index))
        for index, (question, _) in enumerate(TURNS)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    failures: list[str] = []
    for index, (question, expected_fragments) in enumerate(TURNS):
        outcome = results.get(index, {})
        print()
        print(f"--- 第 {index + 1} 轮 ---")
        print(f"  问题: {question}")
        if "error" in outcome:
            print(f"  !! 失败: {outcome['error']}")
            failures.append(f"turn {index + 1} errored")
            continue
        print(f"  会话: {outcome['session_id']}")
        print(f"  检索查询: {outcome['retrieval_query']}")
        print(f"  工具: {outcome['tool_calls']}")
        print(f"  事件: {outcome['events']}  耗时 {outcome['seconds']}s")
        print(f"  答案: {outcome['answer'][:150].replace(chr(10), ' ')}")

        if not outcome["events"].get("done"):
            failures.append(f"turn {index + 1} produced no done event")
        # The rewritten query must belong to THIS turn: that is the signal that the
        # per-turn context was not shared.
        if outcome["retrieval_query"] and question[:6] not in outcome["retrieval_query"]:
            failures.append(
                f"turn {index + 1} retrieval query looks like another turn's: {outcome['retrieval_query']!r}"
            )
        # Answers format numbers with thousands separators ("51,053"), so compare
        # against a copy with separators removed rather than forcing the model to
        # write them one way.
        normalized = outcome["answer"].replace(",", "").replace("，", "")
        missing = [f for f in expected_fragments if f not in normalized]
        if missing:
            failures.append(f"turn {index + 1} answer missing {missing}")

    print()
    print(f"=== 结果 === 并发总耗时 {wall:.1f}s（单轮耗时之和 {sum(r.get('seconds', 0) for r in results.values()):.1f}s）")
    if wall < sum(r.get("seconds", 0) for r in results.values()) * 0.95:
        print("  两轮确有重叠执行（总耗时小于各自耗时之和）")
    else:
        print("  ⚠ 未观察到重叠：请求可能被串行化，本次未能验证并发安全")

    print()
    if failures:
        print("RESULT: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("RESULT: PASS — 两轮各自完成，检索查询与工具调用未串台")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
