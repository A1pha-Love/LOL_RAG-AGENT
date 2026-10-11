"""Measure rerank score distributions per calibration class.

Threshold calibration needs to know whether relevant, offtopic and unanswerable
queries actually separate. This runs every calibration case through the real
retrieval path and reports the best score per class, so a cutoff can be chosen from
data instead of intuition.

    python scripts/calibrate_threshold.py            # report only
    python scripts/calibrate_threshold.py --sweep    # also sweep candidate cutoffs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.knowledge import (  # noqa: E402
    MIN_RERANK_SCORE,
    RERANK_CANDIDATE_POOL,
    embed_texts,
    search_knowledge,
)

CALIBRATION_PATH = PROJECT_ROOT / "evals" / "retrieval_calibration.jsonl"


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in CALIBRATION_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def heading_matches(hit: dict, expected: list[dict]) -> bool:
    """Whether a hit satisfies any expected (source, heading) pair."""
    source = str(hit.get("source") or "")
    heading_path = str(hit.get("heading_path") or "")
    for item in expected:
        want_source = str(item.get("source") or "")
        want_heading = str(item.get("heading") or "")
        if want_source and want_source not in source:
            continue
        if want_heading and want_heading not in heading_path:
            continue
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    cases = load_cases()
    print(f"标定用例: {len(cases)}    当前 MIN_RERANK_SCORE = {MIN_RERANK_SCORE}")
    print(f"候选池: {RERANK_CANDIDATE_POOL}    检索 top_k = {args.top_k}")
    print()

    # Warm the embedding cache once; every query needs its vector anyway.
    embed_texts([case["query"] for case in cases])

    results: list[dict] = []
    for case in cases:
        # include_below_threshold so the raw score is visible even when the cutoff
        # would have dropped it — that is what makes the distributions comparable.
        # include_noise because measuring these distractors is the entire point of
        # calibration; normal retrieval excludes them.
        hits = search_knowledge(
            case["query"],
            top_k=args.top_k,
            include_below_threshold=True,
            include_noise=True,
        )
        scores = [
            hit.get("rerank_score") if hit.get("rerank_score") is not None else hit.get("similarity")
            for hit in hits
        ]
        top = max((s for s in scores if s is not None), default=None)
        expected = case.get("expected_retrieval") or []
        hit_rank = None
        for index, hit in enumerate(hits, start=1):
            if expected and heading_matches(hit, expected):
                hit_rank = index
                break
        results.append(
            {
                "id": case["id"],
                "class": case.get("class"),
                "top_score": top,
                "hit_rank": hit_rank,
                "expected": bool(expected),
                "scores": scores,
            }
        )

    print("=== 逐条结果 ===")
    print(f"  {'类别':<13}{'用例':<28}{'最高分':>8}  {'期望来源排名':>12}")
    for item in results:
        top = f"{item['top_score']:.4f}" if item["top_score"] is not None else "n/a"
        rank = item["hit_rank"] if item["hit_rank"] else ("—" if item["expected"] else "（无期望）")
        print(f"  {item['class']:<13}{item['id']:<28}{top:>8}  {str(rank):>12}")

    print()
    print("=== 各分类的最高分分布 ===")
    by_class: dict[str, list[float]] = {}
    for item in results:
        if item["top_score"] is not None:
            by_class.setdefault(str(item["class"]), []).append(item["top_score"])
    for name, values in sorted(by_class.items()):
        values.sort()
        print(
            f"  {name:<13} n={len(values):<3} min={values[0]:.4f} "
            f"中位={values[len(values) // 2]:.4f} max={values[-1]:.4f}"
        )

    relevant = by_class.get("relevant", [])
    others = by_class.get("offtopic", []) + by_class.get("unanswerable", [])
    if relevant and others:
        print()
        print("=== 可分性 ===")
        print(f"  相关最高分最小值: {min(relevant):.4f}")
        print(f"  无关/不可答最高分最大值: {max(others):.4f}")
        gap = min(relevant) - max(others)
        if gap > 0:
            print(f"  完全可分，间隔 {gap:.4f}：任何 [{max(others):.4f}, {min(relevant):.4f}] 内的阈值都行")
        else:
            print(f"  ⚠ 重叠 {-gap:.4f}：不存在能同时避免漏召与误召的单一阈值")

    # Recall@k for the expected-source cases.
    with_source = [item for item in results if item["expected"]]
    if with_source:
        found = [item for item in with_source if item["hit_rank"]]
        print()
        print(f"=== 期望来源召回（top {args.top_k}，不限阈值）===")
        print(f"  {len(found)}/{len(with_source)}")
        for item in with_source:
            if not item["hit_rank"]:
                print(f"    未命中: {item['id']}（最高分 {item['top_score']:.4f}）")

    if args.sweep:
        print()
        print("=== 阈值扫描（按 top_k 内是否仍返回结果判定）===")
        print(f"  {'阈值':>7}{'相关有结果':>12}{'无关有结果':>12}{'误召':>8}")
        for step in range(30, 100, 5):
            cutoff = step / 100
            relevant_shown = sum(1 for item in results if item["class"] == "relevant" and item["top_score"] and item["top_score"] >= cutoff)
            other_shown = sum(1 for item in results if item["class"] in {"offtopic", "unanswerable"} and item["top_score"] and item["top_score"] >= cutoff)
            total_relevant = len(by_class.get("relevant", []))
            total_other = len(by_class.get("offtopic", [])) + len(by_class.get("unanswerable", []))
            print(
                f"  {cutoff:>7.2f}{relevant_shown:>7}/{total_relevant:<4}{other_shown:>7}/{total_other:<4}"
                f"{other_shown:>7}"
            )

    report_path = PROJECT_ROOT / "reports" / "threshold_calibration.json"
    payload = {
        "min_rerank_score_current": MIN_RERANK_SCORE,
        "top_k": args.top_k,
        "by_class": {name: sorted(values) for name, values in by_class.items()},
        "cases": results,
    }
    try:
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport -> {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    except PermissionError:
        print(f"\n（无法写入 {report_path}：沙箱限制）")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
