"""Measure retrieval-query rewriting on real follow-up questions.

The manual review criterion is whether the rewritten query still contains the
subject the follow-up question omitted. That is checked programmatically via a
small keyword table, so the result is a number rather than an impression.

    python scripts/measure_rewrite.py            # offline-only checks, no API
    python scripts/measure_rewrite.py --live     # also run the rewrite model call

The live pass costs one chat completion per follow-up question (cheap, and it
never rebuilds embeddings).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.conversation import Turn, format_history, rewrite_query  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    """A follow-up question and the subject its rewrite must still mention."""

    name: str
    history: list[tuple[str, str]]
    question: str
    must_mention_any: tuple[str, ...]


SCENARIOS = [
    Scenario(
        name="champion_followup",
        history=[
            ("user", "英雄 ID 157 被禁用了多少场？"),
            ("assistant", "英雄 ID 157 被禁用 27,146 场，Ban 率 53.17%。"),
        ],
        question="那他的胜率呢？",
        must_mention_any=("157", "亚索", "疾风剑豪"),
    ),
    Scenario(
        name="event_followup",
        history=[
            ("user", "一血发生率是多少？"),
            ("assistant", "一血发生率为 98.91%（50,499 / 51,053）。"),
        ],
        question="那个条件胜率呢？",
        must_mention_any=("一血", "firstBlood"),
    ),
    Scenario(
        name="dedup_followup",
        history=[
            ("user", "这份数据有多少场比赛？"),
            ("assistant", "去重后有 51,053 场。"),
        ],
        question="那原始行数是多少？重复记录怎么处理？",
        must_mention_any=("行数", "重复", "51,490"),
    ),
]


def offline_checks() -> int:
    problems = []
    print("=== 离线可验证的部分 ===")
    for scenario in SCENARIOS:
        turns = [Turn(role=role, content=text, created_at="2026-01-01T00:00:00+00:00") for role, text in scenario.history]
        transcript = format_history(turns)
        ok = all(text[:20] in transcript for _, text in scenario.history)
        print(f"  [{'PASS' if ok else 'FAIL'}] {scenario.name}: 历史{len(turns)}轮已进入改写提示词")
        if not ok:
            problems.append(scenario.name)

    # With no history there must be no rewrite attempt at all.
    query, fallback = rewrite_query("这份数据有多少场比赛？", [])
    no_history_ok = query == "这份数据有多少场比赛？" and fallback is None
    print(f"  [{'PASS' if no_history_ok else 'FAIL'}] 首轮不改写、不调用模型")
    if not no_history_ok:
        problems.append("no_history_shortcut")
    return len(problems)


def live_checks() -> tuple[int, list[dict]]:
    print()
    print("=== 实际改写（调用对话模型）===")
    kept, records = 0, []
    for scenario in SCENARIOS:
        turns = [Turn(role=role, content=text, created_at="2026-01-01T00:00:00+00:00") for role, text in scenario.history]
        started = time.perf_counter()
        rewritten, fallback = rewrite_query(scenario.question, turns)
        elapsed = time.perf_counter() - started

        changed = rewritten.strip() != scenario.question.strip()
        retained = any(token in rewritten for token in scenario.must_mention_any)
        kept += int(retained)
        marker = "PASS" if retained else "MISS"

        print(f"  [{marker}] {scenario.name}  ({elapsed:.1f}s)")
        print(f"        原问题 : {scenario.question}")
        print(f"        改写后 : {rewritten}")
        print(f"        是否改写: {changed} | 保留了主语({ '/'.join(scenario.must_mention_any) }): {retained}")
        if fallback:
            print(f"        !! 回退原因: {fallback}")

        records.append(
            {
                "name": scenario.name,
                "question": scenario.question,
                "rewritten": rewritten,
                "changed": changed,
                "retained_subject": retained,
                "fallback": fallback,
                "seconds": round(elapsed, 2),
            }
        )
    return kept, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="also run the rewrite model call")
    args = parser.parse_args()

    problems = offline_checks()
    report: dict = {"offline_problems": problems}

    if args.live:
        kept, records = live_checks()
        report["live"] = records
        report["live_kept_subject"] = f"{kept}/{len(SCENARIOS)}"
        print()
        print(f"改写后仍保留主语的用例：{kept}/{len(SCENARIOS)}")

    report_path = PROJECT_ROOT / "reports" / "rewrite_measurement.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport -> {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    except PermissionError:
        print(f"\n（无法写入 {report_path}：沙箱限制，报告未落盘）")

    if not args.live:
        print("（未运行模型改写；加 --live 会实际调用对话模型）")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
