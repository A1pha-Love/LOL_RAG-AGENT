"""Check that a definition-type question returns a populated retrieval panel."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
QUESTION = sys.argv[2] if len(sys.argv) > 2 else "Ban 率按什么口径计算？同一英雄同场多次禁用怎么算？"

# Python picks up this machine's system proxy from the registry, and urllib does
# not honour Windows' ProxyOverride list, so requests to 127.0.0.1 would
# otherwise be routed through the proxy and intermittently fail. Keep them direct.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload: dict, *, timeout: int = 180):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return OPENER.open(request, timeout=timeout)


session_id = json.loads(post(BASE + "/api/sessions", {}, timeout=20).read())["session_id"]
print(f"session: {session_id}")
print(f"question: {QUESTION}")
print()

chunks: list[dict] = []
counts: dict[str, int] = {}
for raw in post(BASE + "/api/sessions/ask", {"question": QUESTION, "session_id": session_id}):
    line = raw.decode("utf-8", "replace").rstrip("\n")
    if not line.startswith("data:"):
        continue
    event = json.loads(line[5:].strip())
    kind = event.get("type", "?")
    counts[kind] = counts.get(kind, 0) + 1
    if kind == "rewrite":
        print(f"  检索查询: {event.get('query')}")
    elif kind == "retrieval":
        chunks = event.get("results") or []

print(f"  事件统计: {counts}")
print()
print(f"命中分块数: {len(chunks)}")
for chunk in chunks:
    score = chunk.get("rerank_score")
    if score is None:
        score = chunk.get("similarity") or 0
    print(f"  {score:.3f}  [{chunk.get('category')}] {chunk.get('heading_path')}")
    if chunk.get("preview"):
        print(f"          {chunk['preview'][:90]}")
