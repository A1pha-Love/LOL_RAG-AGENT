"""Compare scoring strategies: weighted sum vs reciprocal rank fusion."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.knowledge import (  # noqa: E402
    INDEX_PATH,
    _bm25_scores,
    _lexical_similarity,
    _cosine_similarity,
    embed_texts,
)

cases = [
    json.loads(line)
    for line in (PROJECT_ROOT / "evals" / "retrieval_calibration.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
vectors = embed_texts([case["query"] for case in cases])

with sqlite3.connect(INDEX_PATH) as connection:
    rows = connection.execute(
        "SELECT chunk_id, source, heading_path, category, content, embedding FROM knowledge_chunks"
    ).fetchall()

INDEX = [
    (chunk_id, source, heading_path, category, content, json.loads(vector_json))
    for chunk_id, source, heading_path, category, content, vector_json in rows
]

RRF_K = 60


def rank_lists(query: str, query_vector: list[float]):
    bm25 = _bm25_scores(sqlite3.connect(INDEX_PATH), query)
    vector_scores = []
    lexical_scores = []
    bm25_scores = []
    for chunk_id, _source, heading_path, _category, content, vector in INDEX:
        vector_scores.append((chunk_id, _cosine_similarity(query_vector, vector)))
        lexical_scores.append((chunk_id, _lexical_similarity(query, content)))
        bm25_scores.append((chunk_id, bm25.get(chunk_id, 0.0)))
    ordering = {}
    for name, scores in (("vector", vector_scores), ("lexical", lexical_scores), ("bm25", bm25_scores)):
        ordering[name] = {
            chunk_id: rank
            for rank, (chunk_id, _) in enumerate(sorted(scores, key=lambda item: -item[1]), start=1)
        }
    return ordering


def evaluate(strategy, weights):
    hits = 0
    total = 0
    per_case = []
    for case, vector in zip(cases, vectors, strict=True):
        ordering = rank_lists(case["query"], vector)
        fused = []
        for chunk_id, source, heading_path, category, content, _ in INDEX:
            score = sum(
                weight / (RRF_K + ordering[name][chunk_id])
                for name, weight in weights.items()
            )
            fused.append((score, chunk_id, source, heading_path))
        fused.sort(key=lambda item: (-item[0], item[1]))
        expected = case.get("expected_retrieval") or []
        if not expected:
            continue
        total += 1
        top3 = fused[:3]
        hit = all(
            any(source == entry["source"] and entry["heading"] in heading_path for _, _, source, heading_path in top3)
            for entry in expected
        )
        hits += int(hit)
        per_case.append((case["id"], hit))
    return hits, total, per_case


STRATEGIES = {
    "vector only        ": {"vector": 1.0},
    "vector+lexical     ": {"vector": 0.75, "lexical": 0.25},
    "RRF v+l+b (1,1,1)  ": {"vector": 1.0, "lexical": 1.0, "bm25": 1.0},
    "RRF v+l+b (2,1,2)  ": {"vector": 2.0, "lexical": 1.0, "bm25": 2.0},
    "RRF v+b (1,1)      ": {"vector": 1.0, "bm25": 1.0},
    "RRF b only         ": {"bm25": 1.0},
}

for name, weights in STRATEGIES.items():
    hits, total, per_case = evaluate(name, weights)
    misses = [case_id for case_id, hit in per_case if not hit]
    print(f"{name}  recall@3 = {hits}/{total}   missed: {', '.join(misses) if misses else '-'}")
