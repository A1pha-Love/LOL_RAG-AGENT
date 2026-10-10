"""Compare the workspace-scoped endpoint against the generic DashScope endpoint.

The project currently builds URLs as https://{workspace}.{region}.maas.aliyuncs.com,
which ties configuration to one business space. DashScope also serves a generic
host that authenticates purely by API key. If the generic host works, the
workspace id stops being a required setting.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
WORKSPACE = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
REGION = os.getenv("DASHSCOPE_REGION", "cn-beijing").strip()
EMBED_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "qwen3.7-text-embedding").strip()
CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen3.8-flash").strip()
RERANK_MODEL = os.getenv("QWEN_RERANK_MODEL", "qwen3-rerank").strip()

TARGETS = [
    ("workspace-scoped", f"https://{WORKSPACE}.{REGION}.maas.aliyuncs.com/compatible-mode/v1"),
    ("generic", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ("generic-intl", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
]

RERANK_TARGETS = [
    ("workspace-scoped", f"https://{WORKSPACE}.{REGION}.maas.aliyuncs.com/compatible-api/v1/reranks"),
    ("generic", "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"),
    ("generic-compat-mode", "https://dashscope.aliyuncs.com/compatible-mode/v1/reranks"),
]


def post(url: str, payload: dict) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode("utf-8", "replace")[:200]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:200]
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def summarize(status: int, body: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body[:120].replace("\n", " ")
    if "error" in parsed:
        error = parsed["error"]
        return f"{error.get('code') or error.get('type')}: {error.get('message')}"
    if parsed.get("data"):
        first = parsed["data"][0]
        if first.get("embedding"):
            return f"OK embeddings dim={len(first['embedding'])}"
        return "OK"
    if parsed.get("choices"):
        return f"OK chat model={parsed.get('model')}"
    if parsed.get("results"):
        return f"OK rerank results={len(parsed['results'])}"
    return json.dumps(parsed, ensure_ascii=False)[:120]


print("=" * 92)
print("CHAT + EMBEDDINGS")
print("=" * 92)
for label, base in TARGETS:
    for kind, path, payload in (
        ("chat", "/chat/completions", {"model": CHAT_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}),
        ("embed", "/embeddings", {"model": EMBED_MODEL, "input": ["测试"], "dimensions": 1024}),
    ):
        status, body = post(base + path, payload)
        print(f"  [{status:>3}] {label:16} {kind:5}  {summarize(status, body)}")
    print()

print("=" * 92)
print("RERANK")
print("=" * 92)
for label, url in RERANK_TARGETS:
    status, body = post(url, {"model": RERANK_MODEL, "query": "Ban 率", "documents": ["Ban 率的分母", "闪现"]})
    print(f"  [{status:>3}] {label:22} {summarize(status, body)}")
