"""Build or refresh the local RAG index from the kbase/ knowledge base.

Chunking runs locally; embeddings are requested from the configured Qwen
Embedding endpoint in batches, so this command does call the API and consumes
quota. Use --dry-run to inspect chunking with no API call.

Usage:
    python scripts/index_knowledge.py                 # build / refresh the index
    python scripts/index_knowledge.py --dry-run       # chunking only, no API
    python scripts/index_knowledge.py --check         # report staleness, no API
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.knowledge import (  # noqa: E402
    INDEX_PATH,
    build_knowledge_index,
    check_index_staleness,
)


def _redact(message: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    return message.replace(api_key, "[已隐藏]") if api_key else message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="chunk only; never call the Embedding API")
    parser.add_argument("--check", action="store_true", help="report whether the index is stale, then exit")
    args = parser.parse_args()

    if args.check:
        report = check_index_staleness()
        print(f"索引文件: {INDEX_PATH.relative_to(PROJECT_ROOT).as_posix()}")
        print(f"存在: {report['index_exists']}  过期: {report['stale']}  原因: {report.get('reason')}")
        for key in ("new_documents", "changed_documents", "missing_documents"):
            if report[key]:
                print(f"  {key}:")
                for item in report[key]:
                    print(f"    - {item}")
        if report.get("meta"):
            print("索引元数据:")
            for key, value in sorted(report["meta"].items()):
                print(f"    {key} = {value}")
        return 0 if (report["index_exists"] and not report["stale"]) else 1

    try:
        report = build_knowledge_index(dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"知识索引生成失败（{type(exc).__name__}）：{_redact(str(exc))}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("（本次未调用 Embedding API；不带 --dry-run 才会写入索引。）")
        return 0

    print(
        f"完成：{report['chunks']} 个分块 / {report['documents']} 份文档"
        f" -> {report['index_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
