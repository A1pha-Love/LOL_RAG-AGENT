"""Verify retrieval assertions offline, reusing vectors already stored in the index.

Needs no Embedding API call: the index holds every chunk vector, and evaluation
questions were embedded earlier. Reranking, when enabled, does call the rerank
endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lol_agent.knowledge import INDEX_PATH, MIN_RERANK_SCORE  # noqa: E402
from lol_agent.reranker import RerankUnavailable, rerank  # noqa: E402
import _eval_retrieval as _retrieval  # noqa: E402

# Question vectors are cached from previous runs so this check stays offline.
# Question vectors are cached so this check can be re-run without paying for
# embeddings again. Kept at the workspace root because subdirectories such as
# reports/ are not writable when the file sandbox is confined.
CACHE_PATH = PROJECT_ROOT / ".cache" / "query_vectors.json"


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    return dot / (ln * rn) if ln and rn else 0.0


def embed_queries(queries: list[str]) -> dict[str, list[float]]:
    """Return cached query vectors, fetching only those not already cached."""
    cache: dict[str, list[float]] = {}
    if CACHE_PATH.is_file():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    missing = [query for query in queries if query not in cache]
    if missing:
        from lol_agent.knowledge import embed_texts

        vectors = embed_texts(missing)
        cache.update(dict(zip(missing, vectors, strict=True)))
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"embedded {len(missing)} new question(s); cache -> {CACHE_PATH.name}")
    else:
        print(f"all {len(queries)} question vectors served from cache (no API call)")
    return cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true", help="also run the rerank stage (calls the API)")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in (PROJECT_ROOT / "evals" / "rag_cases.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    cases = [case for case in cases if case.get("expected_retrieval")]

    with sqlite3.connect(INDEX_PATH) as connection:
        rows = connection.execute(
            "SELECT chunk_id, source, heading_path, content, embedding FROM knowledge_chunks"
        ).fetchall()
    index = [
        {"chunk_id": cid, "source": src, "heading_path": hp, "content": content, "vector": json.loads(vec)}
        for cid, src, hp, content, vec in rows
    ]
    print(f"index chunks: {len(index)} | cases with expectations: {len(cases)} | rerank: {args.rerank}")

    vectors = embed_queries([case["question"] for case in cases])

    passed = 0
    for case in cases:
        query_vector = vectors[case["question"]]
        scored = sorted(
            (
                {**item, "embedding_score": cosine(query_vector, item["vector"])}
                for item in index
            ),
            key=lambda item: -item["embedding_score"],
        )
        pool = scored[:20]
        top = pool
        note = "embedding order"
        if args.rerank:
            try:
                ordering = rerank(case["question"], [item["content"] for item in pool], top_n=len(pool))
                pool = [{**pool[i], "rerank_score": round(score, 6)} for i, score in ordering]
                pool.sort(key=lambda item: -item["rerank_score"])
                top = [item for item in pool if item["rerank_score"] >= MIN_RERANK_SCORE]
                note = f"reranked, {len(top)} above cutoff"
            except RerankUnavailable as exc:
                note = f"rerank unavailable ({exc})"
        top = top[: args.top_k]

        misses = [
            entry
            for entry in case["expected_retrieval"]
            if not _retrieval.is_satisfied(entry, top)
        ]
        passed += int(not misses)
        print(f"[{'PASS' if not misses else 'FAIL'}] {case['id']}  ({note})")
        for entry in case["expected_retrieval"]:
            label = _retrieval.describe_expectation(entry)
            position = _retrieval.best_rank(entry, pool)
            mark = "ok" if not any(entry == miss for miss in misses) else "MISS"
            print(f"    {mark:4} rank={position}  {label}")

    print()
    print(f"RETRIEVAL ASSERTIONS: {passed}/{len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
