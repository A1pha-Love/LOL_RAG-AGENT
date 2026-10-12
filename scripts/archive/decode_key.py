"""Decode the workspace id embedded in an sk-ws- DashScope key."""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
key = os.getenv("DASHSCOPE_API_KEY", "").strip()
configured = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()

print(f"configured workspace : {configured}")
print(f"key prefix           : {key[:10]}...  (len={len(key)})")
print()

body = key[6:] if key.startswith("sk-ws-") else None
print(f"after 'sk-ws-' prefix : repr={body[:24]!r}... len={len(body) if body else 0}")
if body is None:
    raise SystemExit("key does not use the sk-ws- form; cannot decode")

for label, candidate in (
    ("raw", body),
    ("padding+raw", body + "=" * (-len(body) % 4)),
):
    for decoder_name, decoder in (("urlsafe", base64.urlsafe_b64decode), ("standard", base64.b64decode)):
        try:
            raw = decoder(candidate)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:14} {decoder_name:8} -> {type(exc).__name__}")
            continue
        text = raw.decode("utf-8", "replace")
        printable = "".join(ch if ch.isprintable() else "." for ch in text)
        print(f"  {label:14} {decoder_name:8} -> {printable[:200]}")

        found = re.findall(r"ws-[a-z0-9]{6,}", text)
        if found:
            print(f"        >>> workspace id candidate: {found[0]}")
        try:
            parsed = json.loads(text)
            print(f"        >>> json fields: {list(parsed) if isinstance(parsed, dict) else type(parsed).__name__}")
            if isinstance(parsed, dict):
                print(f"        >>> json = {json.dumps(parsed, ensure_ascii=False)[:300]}")
        except json.JSONDecodeError:
            pass
