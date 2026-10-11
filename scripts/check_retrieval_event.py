"""Check that a definition-type question returns a populated retrieval panel."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
QUESTION = sys.argv[2] if len(sys.argv) > 2 else "Ban 率按什么口径计算？同一英雄同场多次禁用怎么算？"

create = urllib.request.Request(
    BASE + "/api/sessions", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
)
session_id = json.loads(urllib.request.urlopen(create, timeout=20).read())["session_id"]
print(f"session: {session_id}")
print(f"question: {QUESTION}")
print()

body = json.dumps({"question": QUESTION, "session_id": session_id}).encode("utf-8")
request = urllib.request.Request(
    BASE + "/api/sessions/ask", data=body, headers={"Content-Type": "application/json"}, method="POST"
)

chunks: list[dict] = []
counts: dict[str, int] = {}
for raw in urllib.request.urlopen(request, timeout=180):
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
