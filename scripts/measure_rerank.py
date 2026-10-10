"""Measure Recall@k for embedding-only vs reranked retrieval.

Uses one batched embedding call for all queries, then reranks candidates per
query. Writes a JSON report so numbers can be cited without re-running.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.knowledge import INDEX_PATH, embed_texts  # noqa: E402
from lol_agent.reranker import RerankUnavailable, rerank, rerank_model  # noqa: E402

CANDIDATE_POOL = 20
REPORT_PATH = PROJECT_ROOT / "reports" / "retrieval_rerank_report.json"


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def load_cases() -> list[dict]:
    path = PROJECT_ROOT / "evals" / "retrieval_calibration.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_index() -> list[tuple[str, str, str, str, str, list[float]]]:
    with sqlite3.connect(INDEX_PATH) as connection:
        rows = connection.execute(
            "SELECT chunk_id, source, heading_path, category, content, embedding FROM knowledge_chunks"
        ).fetchall()
    return [
        (chunk_id, source, heading_path, category, content, json.loads(vector_json))
        for chunk_id, source, heading_path, category, content, vector_json in rows
    ]


def matches(chunk: dict, expected: list[dict]) -> bool:
    return all(
        any(
            candidate["source"] == entry["source"] and entry["heading"] in candidate["heading_path"]
            for candidate in chunk
        )
        for entry in expected
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true", help="skip the rerank pass")
    args = parser.parse_args()

    cases = load_cases()
    index = load_index()
    print(f"index chunks: {len(index)} | cases: {len(cases)} | rerank model: {rerank_model()}")
    vectors = embed_texts([case["query"] for case in cases])

    results = []
    rerank_failures = 0
    for case, query_vector in zip(cases, vectors, strict=True):
        scored = sorted(
            (
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "heading_path": heading_path,
                    "category": category,
                    "content": content,
                    "embedding_score": round(cosine(query_vector, vector), 6),
                }
                for chunk_id, source, heading_path, category, content, vector in index
            ),
            key=lambda item: (-item["embedding_score"], item["chunk_id"]),
        )
        pool = scored[:CANDIDATE_POOL]
        expected = case.get("expected_retrieval") or []

        record = {
            "id": case["id"],
            "class": case["class"],
            "expected": expected,
            "embedding_top3": [item["heading_path"] for item in pool[:3]],
            "embedding_top1_score": round(pool[0]["embedding_score"], 6) if pool else None,
        }
        if expected:
            record["embedding_hit@3"] = matches(pool[:3], expected)
            record["embedding_hit@1"] = matches(pool[:1], expected)

        if not args.no_rerank:
            try:
                ranked = rerank(case["query"], [item["content"] for item in pool], top_n=CANDIDATE_POOL)
            except RerankUnavailable as exc:
                rerank_failures += 1
                record["rerank_error"] = str(exc)[:200]
                ranked = None
            if ranked is not None:
                reranked = [
                    {**pool[index], "relevance_score": round(relevance, 6)}
                    for index, relevance in ranked
                ]
                record["reranked_top3"] = [item["heading_path"] for item in reranked[:3]]
                record["rerank_scores"] = [item["relevance_score"] for item in reranked[:3]]
                record["rerank_top1_score"] = reranked[0]["relevance_score"] if reranked else None
                if expected:
                    record["reranked_hit@3"] = matches(reranked[:3], expected)
                    record["reranked_hit@1"] = matches(reranked[:1], expected)
                    # Where did the expected chunk move to?
                    record["expected_rank_after"] = [
                        next(
                            (
                                position
                                for position, item in enumerate(reranked, start=1)
                                if item["source"] == entry["source"] and entry["heading"] in item["heading_path"]
                            ),
                            None,
                        )
                        for entry in expected
                    ]

        results.append(record)
        flag = ""
        if expected:
            before = record.get("embedding_hit@3")
            after = record.get("reranked_hit@3")
            if before is False and after is True:
                flag = "  <-- rerank FIXED"
            elif before is True and after is False:
                flag = "  <-- rerank BROKE"
        print(f"  {case['id']:32} {case['class']:13}{flag}")

    scored_cases = [r for r in results if r.get("expected")]
    summary = {
        "cases_total": len(cases),
        "cases_scored": len(scored_cases),
        "rerank_model": rerank_model(),
        "rerank_failures": rerank_failures,
        "embedding_recall@1": sum(r.get("embedding_hit@1", False) for r in scored_cases),
        "embedding_recall@3": sum(r.get("embedding_hit@3", False) for r in scored_cases),
        "reranked_recall@1": sum(r.get("reranked_hit@1", False) for r in scored_cases),
        "reranked_recall@3": sum(r.get("reranked_hit@3", False) for r in scored_cases),
        "candidate_pool": CANDIDATE_POOL,
    }
    for key in ("embedding_top1_score", "rerank_top1_score"):
        print()
        print(f"--- {key} by class ---")
        summary[f"{key}_by_class"] = {}
        for label in ("relevant", "offtopic", "unanswerable"):
            values = [r[key] for r in results if r["class"] == label and r.get(key) is not None]
            if not values:
                continue
            summary[f"{key}_by_class"][label] = {
                "min": round(min(values), 4), "max": round(max(values), 4),
                "n": len(values),
            }
            print(f"  {label:14} n={len(values):2}  {min(values):.4f} – {max(values):.4f}")
    # Restore natural report order: header, then summary, then breakdowns.
    ordered = {
        key: summary[key]
        for key in (
            "cases_total", "cases_scored", "rerank_model", "rerank_failures",
            "candidate_pool", "embedding_recall@1", "embedding_recall@3",
            "reranked_recall@1", "reranked_recall@3",
            "embedding_top1_score_by_class", "rerank_top1_score_by_class",
        )
        if key in summary
    }
    summary.clear()
    summary.update(ordered)
    for key, value in summary.items():
        print(f"  {key:22} {value}" if not isinstance(value, dict) else f"  {key:22} {json.dumps(value, ensure_ascii=False)}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nreport -> {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
