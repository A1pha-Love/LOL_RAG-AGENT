"""Verify that a new DASHSCOPE_API_KEY works with every endpoint this project uses.

Checks, in order:
  1. whether the key embeds a workspace id (sk-ws-...) and whether it matches .env;
  2. chat + embedding on the OpenAI-compatible path (/compatible-mode/v1);
  3. rerank on its own path (/compatible-api/v1/reranks).

Nothing here is destructive; it only makes a few small API calls.
"""

from __future__ import annotations

import json
import os
import re
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
CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen3.8-flash").strip()
EMBED_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "qwen3.7-text-embedding").strip()
RERANK_MODEL = os.getenv("QWEN_RERANK_MODEL", "qwen3-rerank").strip()

BASE = f"https://{WORKSPACE}.{REGION}.maas.aliyuncs.com"
MODE_BASE = f"{BASE}/compatible-mode/v1"
API_BASE = f"{BASE}/compatible-api/v1"

# The application resolves hosts through lol_agent.config; a hand-built URL here
# would not prove the real code path works, so both are reported.
from lol_agent.config import chat_base_url, rerank_url  # noqa: E402

APP_CHAT_BASE = chat_base_url()
APP_RERANK_URL = rerank_url()


def mask(value: str, keep: int = 6) -> str:
    if not value:
        return "(empty)"
    return f"{value[:keep]}...{value[-4:]} (len={len(value)})" if len(value) > keep + 4 else "(short)"


def decode_workspace_from_key(key: str) -> str | None:
    """Aliyun workspace-scoped keys look like sk-ws-<base64 of {workspaceId, ...}>."""
    match = re.match(r"^sk-ws-([A-Za-z0-9_\-=]+)$", key)
    if not match:
        return None
    token = match.group(1)
    token += "=" * (-len(token) % 4)
    import base64

    try:
        raw = base64.urlsafe_b64decode(token).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        found = re.search(r"(ws-[a-z0-9]+)", raw)
        return found.group(1) if found else raw[:60]
    for field in ("workspaceId", "workspace_id", "WorkspaceId", "id"):
        if isinstance(parsed, dict) and parsed.get(field):
            return str(parsed[field])
    return json.dumps(parsed, ensure_ascii=False)[:80]


def call(url: str, payload: dict, *, label: str) -> tuple[bool, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:260].replace("\n", " ")
        return False, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"

    # OpenAI-compatible shape
    if isinstance(body, dict) and body.get("data"):
        first = body["data"][0]
        if isinstance(first, dict) and first.get("embedding"):
            return True, f"OK embeddings, dim={len(first['embedding'])}"
        if isinstance(first, dict) and first.get("message"):
            return True, "OK chat completion"
    if isinstance(body, dict) and isinstance(body.get("results"), list):
        scores = [round(item.get("relevance_score", 0), 4) for item in body["results"][:3]]
        return True, f"OK rerank, {len(body['results'])} results, top scores={scores}"
    if isinstance(body, dict) and body.get("output"):
        first = body["output"][0] if isinstance(body["output"], list) else body["output"]
        return True, f"OK, output={json.dumps(first, ensure_ascii=False)[:120]}"
    return True, f"OK, unexpected shape: {json.dumps(body, ensure_ascii=False)[:150]}"


print("=" * 92)
print("CONFIG IN USE")
print("=" * 92)
print(f"  DASHSCOPE_API_KEY      : {mask(API_KEY)}")
print(f"  DASHSCOPE_WORKSPACE_ID : {WORKSPACE or '(empty)'}")
print(f"  DASHSCOPE_REGION       : {REGION}")
print(f"  chat / embed / rerank  : {CHAT_MODEL} / {EMBED_MODEL} / {RERANK_MODEL}")

embedded = decode_workspace_from_key(API_KEY)
print()
print(f"  workspace embedded in key : {embedded if embedded else '(not an sk-ws- key, or could not decode)'}")
if embedded and WORKSPACE and embedded != WORKSPACE:
    print("  !! MISMATCH: the key belongs to a different workspace than DASHSCOPE_WORKSPACE_ID.")
elif embedded and embedded == WORKSPACE:
    print("  -> matches DASHSCOPE_WORKSPACE_ID. Only the key needed replacing.")

print()
print("=" * 92)
print("ENDPOINT CHECKS")
print("=" * 92)
print("  (URLs below come from lol_agent.config — the same resolver the app uses)")
checks = [
    (f"{APP_CHAT_BASE}/embeddings", {"model": EMBED_MODEL, "input": ["测试"], "dimensions": 1024}, "embeddings"),
    (f"{APP_CHAT_BASE}/chat/completions", {"model": CHAT_MODEL, "messages": [{"role": "user", "content": "回复 OK"}]}, "chat"),
    (APP_RERANK_URL, {"model": RERANK_MODEL, "query": "Ban 率", "documents": ["Ban 率的分母是对局数", "闪现"]}, "rerank"),
]
failures = []
for url, payload, label in checks:
    ok, detail = call(url, payload, label=label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:11} {url}")
    print(f"         {detail}")
    if not ok:
        failures.append(label)

print()
print("=" * 92)
if failures:
    print(f"RESULT: {len(failures)} endpoint(s) failed -> {', '.join(failures)}")
    print("Reading the error code:")
    print("  AccessDenied.Unpurchased  -> key is fine, but that MODEL is not enabled/paid for on this account.")
    print("  access_denied (Workspace endpoint access denied)")
    print("                            -> the key does not belong to DASHSCOPE_WORKSPACE_ID.")
    print("                               Fix: remove DASHSCOPE_WORKSPACE_ID (falls back to the generic host)")
    print("                               or set it to the workspace this key belongs to.")
    print("  InvalidApiKey             -> the key string itself is wrong.")
else:
    print("RESULT: all endpoints OK — the key is fully usable with the current configuration.")
print("=" * 92)
raise SystemExit(1 if failures else 0)
