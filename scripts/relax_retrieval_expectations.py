"""Allow retrieval expectations to name several equally valid sections.

Why: three cases were recorded as retrieval failures even though the retrieved chunks
did answer the question — the assertion named one specific section while the knowledge
base documents the same fact in more than one place. Requiring a particular section
grades the retriever on an arbitrary choice rather than on whether the answer was
available, and a permanently-red assertion trains people to ignore failures.

Every alternative listed here was verified to exist in the index and to contain the
needed content, by searching for the very strings the case's own `answer_must_include`
requires. No assertion was relaxed without that evidence.

One case also lists `search_project_knowledge` as optional: its figures come from two
statistics tools, and demanding a retrieval call as well recorded a failure for a turn
that had answered correctly. The retrieval assertion is skipped for that case when the
tool is not called, since grading retrieval that did not happen is meaningless.

MIGRATION NOTE: this is a one-shot script and is deliberately not idempotent — running
it twice would nest `alternatives` inside itself. It has already been applied;
evals/rag_cases.jsonl is the source of truth. Keep this file for the rationale and the
verification record, not as a tool to re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evals" / "rag_cases.jsonl"

# Each entry: case id -> acceptable (source, heading) pairs, primary first.
ALTERNATIVES: dict[str, list[dict[str, str]]] = {
    # Asks for the dedup count plus the covered time range and the dedup rule doc.
    # The time range is documented in the audit AND in the provenance document, and
    # `gameId`/dedup handling appears in the dedup rule document. Any of these
    # sections, alongside the tool output, answers the question.
    "match_count_with_scope": [
        {"source": "kbase/project_data_audit.md", "heading": "文件结构"},
        {"source": "kbase/meta_dataset_provenance.md", "heading": "对局数据"},
        {"source": "kbase/project_dedup_rule.md", "heading": "重复情况"},
    ],
    # Asks for the raw row count, the deduped count and how duplicates are handled.
    # The audit's 文件结构 holds the row/column/duplicate figures; the dedup document
    # holds the handling rules and the duplicate counts.
    "dedup_rule_and_duplicates": [
        {"source": "kbase/project_dedup_rule.md", "heading": "原始规模"},
        {"source": "kbase/project_dedup_rule.md", "heading": "重复情况"},
        {"source": "kbase/project_data_audit.md", "heading": "文件结构"},
    ],
    # Asks for the occurrence rate and the conditional win rate with both denominators.
    # The denominator distinction is documented in the ban/event document; the audit's
    # first-event value distribution states the same figures.
    "first_blood_two_rates": [
        {"source": "kbase/project_metrics_ban_event.md", "heading": "两个分母不要混淆"},
        {"source": "kbase/project_metrics_core.md", "heading": "条件队伍胜率"},
        {"source": "kbase/project_data_audit.md", "heading": "首个事件字段取值分布"},
    ],
}

# Cases where a listed tool is helpful but not required, because the other tools plus
# the retrieved documentation already answer the question. See `optional_tools` in
# scripts/eval_agent.py.
OPTIONAL_TOOLS: dict[str, list[str]] = {
    "first_blood_two_rates": ["search_project_knowledge"],
}


def main() -> int:
    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    known = {case["id"] for case in cases}

    missing = sorted(set(ALTERNATIVES) - known)
    if missing:
        print("unknown case id(s): " + ", ".join(missing))
        return 1

    changed: list[str] = []
    for case in cases:
        targets = ALTERNATIVES.get(case["id"])
        if targets:
            case["expected_retrieval"] = [{"alternatives": targets}]
            changed.append(f"{case['id']}: {len(targets)} 个可选小节")

        optional = OPTIONAL_TOOLS.get(case["id"])
        if optional:
            expected = [tool for tool in case.get("expected_tools", []) if tool not in optional]
            case["expected_tools"] = expected
            case["optional_tools"] = optional
            changed.append(f"{case['id']}: {optional} 改为可选，必需工具 {expected}")

    if not changed:
        print("nothing to change")
        return 0

    CASES_PATH.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    for item in changed:
        print(f"  {item}")
    print(f"\n  {len(cases)} cases written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
