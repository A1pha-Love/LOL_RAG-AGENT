"""Clarify whether npm can actually be used: registry reachability and disk writes.

PowerShell on this machine fails TLS handshakes (see earlier DDragon/Kaggle
probes), so registry reachability must be checked from Python, whose TLS stack
works. Likewise, writes into the project tree are limited by the DSH file
sandbox, so what matters is whether the *sandbox* allows the project directory,
not whether PowerShell can write to the user's global npm prefix.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("=== npm registry reachability (from Python) ===")
for url in (
    "https://registry.npmjs.org/vite/latest",
    "https://registry.npmjs.org/vue/latest",
    "https://registry.npmmirror.com/vite/latest",
):
    request = urllib.request.Request(url, headers={"User-Agent": "node-probe"})
    try:
        # Read the whole body: truncating mid-JSON would fake a decode failure
        # and misreport a reachable registry as unreachable.
        with urllib.request.urlopen(request, timeout=25, context=ssl.create_default_context()) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
            print(f"  OK   HTTP {response.status}  {url}  version={payload.get('version', '?')}")
    except urllib.error.HTTPError as exc:
        print(f"  HTTP {exc.code}  {url}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {type(exc).__name__}: {exc}  {url}")

print()
print("=== project-tree writability (what npm install actually needs) ===")
targets = ["frontend", "node_modules", ".cache"]
for name in targets:
    path = PROJECT_ROOT / name
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        print(f"  WRITABLE    {name}/")
    except Exception as exc:  # noqa: BLE001
        print(f"  NOT-WRITABLE {name}/ -> {type(exc).__name__}: {exc}")

print()
print("=== can we create a nested directory tree (needed by vite scaffolds)? ===")
deep = PROJECT_ROOT / "frontend" / "src" / "components"
try:
    deep.mkdir(parents=True, exist_ok=True)
    (deep / ".probe").write_text("x", encoding="utf-8")
    (deep / ".probe").unlink()
    print(f"  OK   created and wrote {deep.relative_to(PROJECT_ROOT)}")
except Exception as exc:  # noqa: BLE001
    print(f"  FAIL {type(exc).__name__}: {exc}")

print()
print("=== existing node_modules in the project? ===")
nm = PROJECT_ROOT / "node_modules"
print("  exists:", nm.is_dir())
