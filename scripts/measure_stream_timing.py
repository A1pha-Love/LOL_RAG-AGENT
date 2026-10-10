"""Measure WHEN stream chunks arrive, to see whether the UI would look frozen.

qwen3.8-flash is a reasoning model: it may emit reasoning_content before the
answer. If nothing whatsoever streams during that phase, a client sees a blank
screen for many seconds even though the model is working.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8077")
    parser.add_argument("--question", default="有多少场比赛？")
    args = parser.parse_args()

    payload = json.dumps({"question": args.question}).encode("utf-8")
    req = urllib.request.Request(
        f"{args.base.rstrip('/')}/api/sessions/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    arrivals: list[tuple[float, str, int]] = []
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            now = time.perf_counter() - started
            if not line.strip():
                continue
            if line.startswith(":"):  # SSE comment / keepalive ping
                arrivals.append((now, "ping", 0))
                continue
            if line.startswith("event:"):
                arrivals.append((now, f"event:{line[6:].strip()}", 0))
                continue
            if line.startswith("data:"):
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                arrivals.append((now, event.get("type", "?"), len(event.get("text", "") or "")))

    print(f"总耗时 {time.perf_counter() - started:.2f}s，共 {len(arrivals)} 条到达记录")
    print()
    print("前 20 条到达（时间 / 类型 / 文本长度）：")
    for when, kind, size in arrivals[:20]:
        print(f"  {when:6.2f}s  {kind:22} {size}")

    first_delta = next((t for t, k, _ in arrivals if k == "delta"), None)
    pings = sum(1 for _, k, _ in arrivals if k == "ping")
    print()
    print(f"首个 delta: {first_delta:.2f}s" if first_delta else "首个 delta: 无")
    print(f"keepalive ping 数: {pings}")

    # A gap longer than this with no traffic at all is perceived as frozen.
    gaps = [(arrivals[i + 1][0] - arrivals[i][0], arrivals[i][1]) for i in range(len(arrivals) - 1)]
    if gaps:
        worst_gap, after = max(gaps)
        print(f"最大静默间隔: {worst_gap:.2f}s（发生在 {after} 之后）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
