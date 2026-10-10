"""Start the local HTTP backend.

    python scripts/serve.py                 # 127.0.0.1:8000
    python scripts/serve.py --port 8077 --reload

Binds to loopback only: this server has no authentication, so exposing it on a
network interface would let anyone read the session store and spend the API key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="仅建议使用回环地址")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="代码变更后自动重启（开发用）")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"警告：{args.host} 不是回环地址。本服务没有身份认证，"
            "对外暴露会让任何人都能读取会话记录并消耗你的 API 配额。",
            file=sys.stderr,
        )

    import uvicorn

    print(f"API: http://{args.host}:{args.port}   文档: http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "apps.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(PROJECT_ROOT / "apps"), str(PROJECT_ROOT / "src")] if args.reload else None,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
