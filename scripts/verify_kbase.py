"""Verify the knowledge base: front matter, chunking, size distribution.

Uses the real chunker (not a simulation), so the reported numbers are exactly
what the retrieval layer stores.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.knowledge import (  # noqa: E402
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    build_knowledge_index,
    check_index_staleness,
    discover_documents,
    load_knowledge_chunks,
)


def main() -> int:
    problems: list[str] = []
    print("=" * 96)
    print(f"documents on disk: {len(discover_documents())}")
    print("=" * 96)
    build_knowledge_index(dry_run=True)

    chunks = load_knowledge_chunks()
    sizes = [len(c.text) for c in chunks]
    oversized = sum(1 for s in sizes if s > MAX_CHUNK_CHARS)
    print()
    print(f"超过上限 {MAX_CHUNK_CHARS} 的分块（应为 0）: {oversized}")
    if oversized:
        problems.append(f"{oversized} chunks exceed MAX_CHUNK_CHARS")

    print()
    print("分块数（按分类）:")
    for category, count in collections.Counter(c.category for c in chunks).most_common():
        print(f"  {category:10} {count:>5}")

    print()
    print("分块最多的 6 份文档:")
    for doc_id, count in collections.Counter(c.doc_id for c in chunks).most_common(6):
        print(f"  {doc_id:44} {count:>4}")

    champion_chunks = [c for c in chunks if c.source.startswith("kbase/domain_champion")]
    # Real invariant: the heading path must not name two different champions,
    # since a chunk holds exactly one champion's content. Level count alone is
    # fine — sections such as "基础属性" legitimately extend the path.
    champion_names = [
        line[3:].strip()
        for path in sorted((PROJECT_ROOT / "kbase").glob("domain_champion_mechanics_*.md"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    ]
    names_set = set(champion_names)
    wrongly_mixed = []
    for chunk in champion_chunks:
        parts = [part.strip() for part in chunk.heading_path.split(">") if part.strip()]
        hits = [part for part in parts if part in names_set]
        # 0 names = a document-level chunk (intro, batch preamble); 1 = correct.
        if len(hits) > 1:
            wrongly_mixed.append((chunk.heading_path, hits))
    empty_path = [c for c in chunks if not c.heading_path]
    print()
    print(f"英雄分块 {len(champion_chunks)}（覆盖 {len(names_set)} 个英雄名）")
    print(f"标题路径含多于一个英雄名的分块（应为 0）: {len(wrongly_mixed)}")
    for path, hits in wrongly_mixed[:5]:
        print(f"    path={path!r} hits={hits}")
    print(f"无标题路径的分块（应为 0）: {len(empty_path)}")
    if wrongly_mixed:
        problems.append(f"{len(wrongly_mixed)} champion chunks name more than one champion")
    if empty_path:
        problems.append(f"{len(empty_path)} chunks have no heading path")

    print()
    print("=== 一个 project 分块 ===")
    for chunk in chunks:
        if chunk.category == "project" and "Ban" in chunk.heading_path:
            print("path:", chunk.heading_path)
            print(chunk.text[:280].replace("\n", " | "))
            break

    print()
    print("=== 一个 offtopic 分块 ===")
    for chunk in chunks:
        if chunk.category == "offtopic":
            print("path:", chunk.heading_path, "| source:", chunk.source)
            print(chunk.text[:180].replace("\n", " | "))
            break

    print()
    print("=== 索引过期检测 ===")
    staleness = check_index_staleness()
    print("  index exists:", staleness["index_exists"])
    print("  stale       :", staleness["stale"])
    print("  reason      :", staleness.get("reason"))

    print()
    print("=" * 96)
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for problem in problems:
            print("  -", problem)
    else:
        print("PROBLEMS: none")
    print("=" * 96)
    return 1 if problems else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
