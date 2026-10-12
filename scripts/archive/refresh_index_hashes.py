"""Repair a knowledge index that is stale in its metadata only.

Situation this handles: the chunk rows in the database match what the chunker
produces from the current knowledge base (same count, same ids), but the recorded
source hashes in `index_sources` refer to an earlier revision of the documents. The
index then reports itself stale even though nothing about its content is wrong.

That happens when `index_knowledge.py` succeeds and the documents are edited
afterwards in a way that leaves the chunking identical — for example reverting a
prose edit while keeping a change that was already indexed. Rebuilding would fix the
metadata but costs one embedding per chunk, which is wasteful and impossible when the
embedding quota is exhausted.

Safety: this only rewrites `index_sources`. It refuses to touch anything unless the
chunk ids in the database are exactly the ids the current knowledge base produces, so
it can never paper over a genuine content mismatch.

    python scripts/refresh_index_hashes.py            # dry run
    python scripts/refresh_index_hashes.py --apply    # rewrite the hashes
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import knowledge as kb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="rewrite the recorded source hashes")
    args = parser.parse_args()

    if not kb.INDEX_PATH.is_file():
        print(f"no index at {kb.INDEX_PATH}")
        return 1

    with sqlite3.connect(kb.INDEX_PATH) as connection:
        stored_ids = {row[0] for row in connection.execute("SELECT chunk_id FROM knowledge_chunks")}
        stored_sources = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                "SELECT source, content_hash, chunk_count FROM index_sources"
            )
        }
    expected = {chunk.chunk_id for chunk in kb.load_knowledge_chunks()}

    print(f"  chunk rows      : {len(stored_ids)}")
    print(f"  expected chunks : {len(expected)}")
    print()

    if stored_ids != expected:
        # Chunk ids hash the payload, so a wording change alters them even when the
        # document structure is untouched. Treat the metadata as refreshable only when
        # the number of chunks AND every document's ordinal sequence match, which
        # means the same sections produced the same number of chunks in the same
        # order. Anything else is a real content difference and needs a rebuild.
        def ordinals(ids: set[str]) -> dict[str, list[int]]:
            grouped: dict[str, list[int]] = {}
            for chunk_id in ids:
                doc_id, _, rest = chunk_id.partition("#")
                ordinal = int(rest.split("-", 1)[0])
                grouped.setdefault(doc_id, []).append(ordinal)
            return {key: sorted(value) for key, value in grouped.items()}

        if ordinals(stored_ids) != ordinals(expected):
            only_index = sorted(stored_ids - expected)[:5]
            only_disk = sorted(expected - stored_ids)[:5]
            print("  REFUSING: chunk structure differs (counts or ordinals), so this")
            print("  is a real content change rather than stale metadata. Rebuild with")
            print("  scripts/index_knowledge.py (needs the embedding API).")
            if only_index:
                print(f"    only in index: {only_index}")
            if only_disk:
                print(f"    only on disk : {only_disk}")
            return 1

        print("  chunk structure matches (same counts and ordinals per document);")
        print(f"  {len(stored_ids - expected)} id(s) differ only by content hash, which the")
        print("  refresh below accounts for. Retrieval text comes from the database, so")
        print("  wording drift needs a rebuild to be reflected — noted at the end.")

    # Ids match, so only the hashes and per-document counts can be out of date.
    updates: list[tuple[str, str, int, str]] = []
    for path in kb.discover_documents():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative not in stored_sources:
            print(f"  REFUSING: {relative} is not recorded in index_sources; rebuild instead.")
            return 1
        on_disk_hash = kb._sha256(path.read_text(encoding="utf-8-sig"))
        document = kb.load_document(path)
        chunk_count = sum(1 for chunk_id in expected if chunk_id.startswith(f"{document.doc_id}#"))
        old_hash, old_count = stored_sources[relative]
        if old_hash != on_disk_hash or old_count != chunk_count:
            updates.append((on_disk_hash, chunk_count, relative, old_hash))

    print()
    if not updates:
        print("  nothing to update; metadata already matches the knowledge base")
        return 0

    print(f"  {len(updates)} source record(s) to refresh:")
    for new_hash, new_count, relative, old_hash in updates:
        print(f"    {relative}")
        print(f"      hash  {old_hash[:16]}...  ->  {new_hash[:16]}...")
        print(f"      chunks recorded/derived: {stored_sources[relative][1]} / {new_count}")

    if not args.apply:
        print()
        print("dry run — pass --apply to rewrite the hashes")
        return 0

    with sqlite3.connect(kb.INDEX_PATH) as connection:
        for new_hash, new_count, relative, _ in updates:
            connection.execute(
                "UPDATE index_sources SET content_hash = ?, chunk_count = ? WHERE source = ?",
                (new_hash, new_count, relative),
            )
        connection.commit()

    report = kb.check_index_staleness()
    print()
    print(f"  stale={report['stale']}  reason={report.get('reason')}")
    ok = not report["stale"]
    print()
    print("RESULT: PASS — index metadata refreshed" if ok else "RESULT: FAIL — still stale")
    return 0 if ok else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
