"""Contract test for the SSE byte format the backend actually writes.

The frontend parser was written against "\n\n" frame separators while
sse-starlette emits CRLF, so no boundary was ever found and the whole stream was
treated as one event. The frontend tests now cover every line-ending style, but
they use synthetic input — this checks the real producer, so a change in the
backend (or in sse-starlette) cannot silently break the contract again.

Skipped when the server is not reachable, so it stays safe in an offline suite.

    python scripts/verify_sse_contract.py [--base http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Python reads this machine's system proxy from the registry, and urllib does not
# honour Windows' ProxyOverride, so loopback traffic must bypass it explicitly.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(url: str, payload: dict, *, timeout: int = 180):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return OPENER.open(request, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    try:
        session_id = json.loads(_post(f"{base}/api/sessions", {}, timeout=10).read())["session_id"]
    except (urllib.error.URLError, OSError):
        print(f"SKIP: 服务器未运行（{base}）")
        # 77 is the conventional "skipped" exit code, which run_checks.py reports as
        # SKIP rather than PASS — silently passing would fake a check that never ran.
        return 77

    response = _post(f"{base}/api/sessions/ask", {"question": "一血发生率是多少？", "session_id": session_id})
    # Read the whole response: a partial read would cut a frame in half and report a
    # bogus failure (which is exactly what an earlier version of this script did).
    raw = response.read().decode("utf-8", "replace")

    failures: list[str] = []

    print("=== 分帧字节 ===")
    crlf = raw.count("\r\n")
    bare_lf = raw.count("\n") - crlf
    bare_cr = raw.count("\r") - crlf
    print(f"  CRLF: {crlf}  裸 LF: {bare_lf}  裸 CR: {bare_cr}")

    # The frontend splits on any of the three, so any is acceptable — what must
    # hold is that a blank line separates frames and every frame has one data line.
    separators = raw.count("\r\n\r\n") + raw.count("\n\n") + raw.count("\r\r")
    print(f"  空行分隔符数量: {separators}")
    if separators < 3:
        failures.append(f"too few frame separators ({separators})")

    # Every frame must contain exactly one JSON object in its data field. The
    # original bug showed two concatenated objects, which this detects.
    frames = raw.replace("\r\n", "\n").split("\n\n")
    payloads: list[dict] = []
    for index, frame in enumerate(frames):
        data_lines = [line[5:].lstrip(" ") for line in frame.split("\n") if line.startswith("data:")]
        if not data_lines:
            continue
        joined = "\n".join(data_lines)
        try:
            payloads.append(json.loads(joined))
        except json.JSONDecodeError as exc:
            failures.append(f"frame {index} is not a single JSON object: {exc.msg}")
            print(f"  !! 帧 {index} 解析失败: {joined[:120]}")

    kinds = [payload.get("type") for payload in payloads]
    summary = {kind: kinds.count(kind) for kind in dict.fromkeys(kinds)}
    print(f"  事件统计: {summary}")
    if not payloads:
        failures.append("no parsable frames")
    for required in ("status", "rewrite"):
        if required not in kinds:
            failures.append(f"missing {required} event")

    # A frame must never contain a second top-level object.
    for payload in payloads:
        if not isinstance(payload, dict) or "type" not in payload:
            failures.append(f"unexpected payload shape: {str(payload)[:80]}")

    # Comment (keepalive) lines must be present on a long turn, and must not be
    # mistaken for data.
    print(f"  keepalive 注释行: {sum(1 for line in raw.splitlines() if line.startswith(':'))}")

    print()
    if failures:
        print("RESULT: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("RESULT: PASS — 每个数据帧都是单个 JSON 对象")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
