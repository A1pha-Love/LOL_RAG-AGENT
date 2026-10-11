"""Compare relevant and unrelated query scores to calibrate a no-match cutoff.

Reads the index built by scripts/index_knowledge.py. The stored chunk text
already carries its heading path prefix, so scoring matches what search does.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.knowledge import (  # noqa: E402
    INDEX_PATH,
    MIN_RETRIEVAL_SCORE,
    _bm25_scores,
    check_index_staleness,
    embed_texts,
    score_chunks,
)


def _load_cases() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "evals" / "retrieval_calibration.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _redact(message: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    return message.replace(api_key, "[已隐藏]") if api_key else message


def main() -> int:
    cases = _load_cases()
    if not INDEX_PATH.is_file():
        print("Knowledge index is missing; run scripts/index_knowledge.py first.", file=sys.stderr)
        return 1

    staleness = check_index_staleness()
    if staleness["stale"]:
        print(f"WARNING: 索引已过期（{staleness.get('reason')}），校准结果不可信。", file=sys.stderr)
        for key in ("new_documents", "changed_documents", "missing_documents"):
            for item in staleness[key]:
                print(f"  {key}: {item}", file=sys.stderr)

    try:
        # embed_texts, not embed_documents: DashScope rejects batches larger than 20
        # and this case list has grown past that. Calling the raw client directly
        # here is what made the 24th case fail with InvalidParameter.
        vectors = embed_texts([case["query"] for case in cases])
    except Exception as exc:  # noqa: BLE001
        print(f"Calibration embedding failed ({type(exc).__name__}): {_redact(str(exc))}", file=sys.stderr)
        return 1

    with sqlite3.connect(INDEX_PATH) as connection:
        rows = connection.execute(
            "SELECT chunk_id, source, heading_path, category, content, embedding FROM knowledge_chunks"
        ).fetchall()
    if not rows:
        print("Knowledge index is empty; run scripts/index_knowledge.py first.", file=sys.stderr)
        return 1

    meta = staleness.get("meta") or {}
    print(f"index: {len(rows)} chunks | chunker={meta.get('chunker_version')} | model={meta.get('embedding_model')}")
    print(f"current cutoff MIN_RETRIEVAL_SCORE = {MIN_RETRIEVAL_SCORE}")
    print()

    # Three classes: in-scope (must retrieve), offtopic-but-present (retrieval is
    # correct, just out of the product's scope), and unanswerable (nothing in the
    # corpus supports them — only these should score below a no-answer cutoff).
    class_scores: dict[str, list[float]] = {"relevant": [], "offtopic": [], "unanswerable": []}
    for case, vector in zip(cases, vectors, strict=True):
        with sqlite3.connect(INDEX_PATH) as connection:
            bm25 = _bm25_scores(connection, case["query"])
        candidates = [
            {
                "chunk_id": chunk_id,
                "source": source,
                "heading_path": heading_path,
                "category": category,
                "content": content,
                "vector": json.loads(vector_json),
                "query_vector": vector,
            }
            for chunk_id, source, heading_path, category, content, vector_json in rows
        ]
        ranked_chunks = score_chunks(case["query"], candidates, bm25)
        ranked = sorted(
            ((item["similarity"], item["source"], item["heading_path"], item["category"]) for item in ranked_chunks),
            key=lambda entry: (-entry[0], entry[1]),
        )
        class_scores[case["class"]].append(ranked[0][0])

        top3 = ranked[:3]
        retrieved = {(source, heading) for _, source, heading, _ in top3}
        required = {(item["source"], item["heading"]) for item in case.get("expected_retrieval", [])}
        # A chunk heading is the full path ("文档 > 节 > 小节"), so an expected
        # heading matches when it appears anywhere in that path.
        relevant_found = (
            all(
                any(source == got_source and expected in got_heading for got_source, got_heading in retrieved)
                for source, expected in required
            )
            if required
            else None
        )
        retrieval_label = str(relevant_found).lower() if relevant_found is not None else "n/a"
        print(f"{case['id']} [{case['class']}] expected_retrieval@3={retrieval_label}")
        for score, source, heading_path, category in top3:
            print(f"  {score:.4f}  [{category}] {source} :: {heading_path}")

    print()
    for label, key in (("Relevant (in scope)", "relevant"), ("Offtopic (present in corpus)", "offtopic")):
        scores = class_scores[key]
        print(f"{label} max-score range: {min(scores):.4f}–{max(scores):.4f}")
    unanswerable = class_scores["unanswerable"]
    print(f"Unanswerable max-score range: {min(unanswerable):.4f}–{max(unanswerable):.4f}")
    floor = min(class_scores["relevant"])
    ceiling = max(unanswerable)
    if floor > ceiling:
        print(f"Observed gap; provisional no-answer cutoff midpoint: {(floor + ceiling) / 2:.4f}")
    else:
        print("In-scope and unanswerable score ranges overlap; do not apply a global cutoff from this sample.")
    print(
        f"A cutoff is only a provisional choice from these {len(cases)} queries; "
        "review failures before enabling or adjusting it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
