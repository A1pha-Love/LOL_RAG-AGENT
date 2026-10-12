"""Validate candidate retrieval cases before writing them into the eval set.

A case whose expected (source, heading) never reaches the top K would be a
permanently failing assertion. This checks each candidate against the real
retrieval path first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.knowledge import search_knowledge  # noqa: E402

CANDIDATES = [
    {
        "id": "color_side_scope_from_fields",
        "question": "为什么不按蓝方红方拆分比较胜率？team1 和 team2 到底代表什么？",
        "expected_retrieval": [
            {"source": "kbase/project_field_reference.md", "heading": "字段语义与颜色方约定"}
        ],
    },
    {
        "id": "spell_pick_rate_denominator",
        "question": "召唤师技能出场率的分母是比赛数还是队伍出场次数？为什么不能把槽位数当分母？",
        "expected_retrieval": [
            {"source": "kbase/project_metrics_summoner_spell.md", "heading": "技能出场率（队伍口径）"}
        ],
    },
    {
        "id": "evaluation_method_scope",
        "question": "这个项目用什么方法评测 RAG 检索与 Agent 回答？判分原则是什么？",
        "expected_retrieval": [
            {"source": "kbase/project_evaluation_method.md", "heading": "两类评测集"}
        ],
    },
    {
        "id": "patch_timeline_new_champions",
        "question": "7.11.1 到 7.18.1 之间英雄池有什么变化？期间新加入了哪些英雄？",
        "expected_retrieval": [
            {"source": "kbase/domain_version_timeline.md", "heading": "期间新加入的英雄"}
        ],
    },
    {
        "id": "dataset_provenance_and_license",
        "question": "这份对局数据来自哪里，使用许可和版本边界是什么？",
        "expected_retrieval": [
            {"source": "kbase/meta_dataset_provenance.md", "heading": "对局数据"}
        ],
    },
]

TOP_K = 8


def matches(hit: dict, expected: dict) -> bool:
    return (
        str(expected["source"]) in str(hit.get("source") or "")
        and str(expected["heading"]) in str(hit.get("heading_path") or "")
    )


def main() -> int:
    failures: list[str] = []
    for case in CANDIDATES:
        hits = search_knowledge(case["question"], top_k=TOP_K)
        print(f"--- {case['id']} ---")
        print(f"    {case['question']}")
        for expected in case["expected_retrieval"]:
            rank = next(
                (i for i, hit in enumerate(hits, start=1) if matches(hit, expected)),
                None,
            )
            status = "PASS" if rank else "FAIL"
            if not rank:
                failures.append(f"{case['id']}: {expected['source']} :: {expected['heading']}")
            print(f"    [{status}] rank={rank}  {expected['source']} :: {expected['heading']}")
        print("    top3:")
        for hit in hits[:3]:
            print(f"      {hit['rerank_score']:.3f}  {hit['source']} :: {hit['heading_path']}")
        print()

    print(json.dumps({"failures": failures}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
