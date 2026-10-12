"""Smoke-test every HTTP endpoint, fast and without any model call.

Why this exists: SSE framing and concurrent turns were covered by checks, but the
session endpoints — including the pin and touch routes that ordering depends on — were
only ever verified by hand. A regression there would break the sidebar silently.

Deliberately does not call /api/sessions/ask: that spends model quota. The turn path
is covered by verify_sse_contract.py and check_concurrent_turns.py.

    python scripts/verify_api_endpoints.py [--base URL]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _http import (  # noqa: E402
    DEFAULT_BASE,
    Report,
    delete,
    get,
    json_body,
    post,
    put,
    server_reachable,
    skip,
    use_utf8_streams,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args()
    base = args.base.rstrip("/")

    if not server_reachable(base):
        return skip(base)

    report = Report("api endpoints")
    created: list[str] = []

    def new_session() -> str | None:
        status, _, body = post(f"{base}/api/sessions", {})
        payload = json_body(body)
        if status != 201 or not payload or "session_id" not in payload:
            return None
        created.append(payload["session_id"])
        return payload["session_id"]

    try:
        # ---------------------------------------------------------------- health
        report.section("健康检查")
        status, ctype, body = get(f"{base}/api/health")
        payload = json_body(body) or {}
        report.check(status == 200 and "json" in ctype, "GET /api/health 返回 JSON", f"HTTP {status}")
        for key in ("status", "index_exists", "index_stale", "index_reason", "session_db"):
            report.check(key in payload, f"health 含字段 {key}")
        report.check(
            isinstance(payload.get("index_exists"), bool) and isinstance(payload.get("index_stale"), bool),
            "索引状态是布尔值而非字符串",
        )

        # ---------------------------------------------------------------- create
        report.section("会话创建")
        status, _, body = post(f"{base}/api/sessions", {})
        payload = json_body(body) or {}
        report.check(status == 201, "POST /api/sessions 返回 201", f"HTTP {status}")
        first = payload.get("session_id")
        report.check(bool(first), "返回了 session_id")
        if first:
            created.append(first)

        # An explicit title must be honoured.
        status, _, body = post(f"{base}/api/sessions", {"title": "自定义标题"})
        titled = (json_body(body) or {}).get("session_id")
        report.check(status == 201 and bool(titled), "可用自定义标题创建会话")
        if titled:
            created.append(titled)

        # ---------------------------------------------------------------- list
        report.section("会话列表")
        status, _, body = get(f"{base}/api/sessions")
        listed = json_body(body) or {}
        report.check(status == 200 and isinstance(listed.get("sessions"), list), "GET /api/sessions 返回列表")
        rows = listed.get("sessions") or []
        report.check(len(rows) >= len(created), "新建的会话都出现在列表里")
        if rows:
            required = {"session_id", "title", "created_at", "updated_at", "turn_count", "pinned", "last_active_at"}
            missing = required - set(rows[0])
            report.check(not missing, "列表项含全部字段", f"缺少 {sorted(missing)}" if missing else "")
        # A limit must be clamped rather than exploding on an absurd value.
        status, _, _ = get(f"{base}/api/sessions?limit=99999")
        report.check(status == 200, "limit 超限被钳制而不是报错", f"HTTP {status}")

        # ---------------------------------------------------------------- get one
        report.section("读取单个会话")
        if first:
            status, _, body = get(f"{base}/api/sessions/{first}")
            detail = json_body(body) or {}
            report.check(status == 200, "GET /api/sessions/{id} 返回 200", f"HTTP {status}")
            report.check(isinstance(detail.get("turns"), list), "详情含 turns 列表")
        status, _, _ = get(f"{base}/api/sessions/definitely-not-a-session")
        report.check(status == 404, "未知会话返回 404", f"HTTP {status}")

        # ---------------------------------------------------------------- pin
        report.section("置顶")
        if first:
            status, _, body = put(f"{base}/api/sessions/{first}/pin", {"pinned": True})
            pinned = json_body(body) or {}
            report.check(status == 200 and pinned.get("pinned") is True, "置顶返回 pinned=true", f"HTTP {status}")

            status, _, body = get(f"{base}/api/sessions")
            rows = (json_body(body) or {}).get("sessions") or []
            report.check(rows[0]["session_id"] == first, "置顶会话排在列表首位", f"首位={rows[0]['session_id'][:6]}")
            report.check(rows[0]["pinned"] == 1, "置顶标记已持久化")

            status, _, body = put(f"{base}/api/sessions/{first}/pin", {"pinned": False})
            report.check(status == 200 and (json_body(body) or {}).get("pinned") is False, "取消置顶返回 pinned=false")

        status, _, _ = put(f"{base}/api/sessions/definitely-not-a-session/pin", {"pinned": True})
        report.check(status == 404, "未知会话置顶返回 404", f"HTTP {status}")
        status, _, _ = put(f"{base}/api/sessions/{first or 'x'}/pin", {})
        report.check(status == 422, "缺少 pinned 字段返回 422", f"HTTP {status}")

        # ---------------------------------------------------------------- touch
        report.section("打开会话（touch）")
        second = new_session()
        if first and second:
            # Pin `first` so the check can prove touch never overtakes a pinned row.
            put(f"{base}/api/sessions/{first}/pin", {"pinned": True})
            status, _, _ = post(f"{base}/api/sessions/{second}/touch", None)
            report.check(status == 200, "POST /api/sessions/{id}/touch 返回 200", f"HTTP {status}")
            status, _, body = get(f"{base}/api/sessions")
            rows = (json_body(body) or {}).get("sessions") or []
            order = [row["session_id"] for row in rows]
            report.check(order and order[0] == first, "打开会话不会越过置顶会话")
            report.check(
                order.index(second) < len(order) and (second in order),
                "被打开的会话在列表中可见",
            )
            # And among the unpinned it should be ahead of the older ones.
            unpinned = [row["session_id"] for row in rows if row["pinned"] == 0]
            report.check(unpinned and unpinned[0] == second, "被打开的会话排在非置顶组首位")
            put(f"{base}/api/sessions/{first}/pin", {"pinned": False})

        status, _, _ = post(f"{base}/api/sessions/definitely-not-a-session/touch", None)
        report.check(status == 404, "未知会话 touch 返回 404", f"HTTP {status}")

        # ---------------------------------------------------------------- delete
        report.section("删除")
        if second:
            status, _, _ = delete(f"{base}/api/sessions/{second}")
            report.check(status == 204, "DELETE 返回 204", f"HTTP {status}")
            created.remove(second)
            status, _, _ = get(f"{base}/api/sessions/{second}")
            report.check(status == 404, "删除后读取返回 404", f"HTTP {status}")
        status, _, _ = delete(f"{base}/api/sessions/definitely-not-a-session")
        report.check(status == 404, "删除未知会话返回 404", f"HTTP {status}")

        # ---------------------------------------------------------------- ask guards
        report.section("问答接口的参数校验（不调用模型）")
        status, _, _ = post(f"{base}/api/sessions/ask", {"question": ""})
        report.check(status == 422, "空问题返回 422", f"HTTP {status}")
        status, ctype, body = post(
            f"{base}/api/sessions/ask", {"question": "x", "session_id": "definitely-not-a-session"}
        )
        report.check("text/event-stream" in ctype, "未知会话仍以 SSE 返回错误", ctype)
        report.check('"type": "error"' in body, "SSE 流里含 error 事件")

        return report.finish()
    finally:
        # Leave the store as it was found.
        for session_id in created:
            delete(f"{base}/api/sessions/{session_id}")


if __name__ == "__main__":
    use_utf8_streams()
    raise SystemExit(main())
