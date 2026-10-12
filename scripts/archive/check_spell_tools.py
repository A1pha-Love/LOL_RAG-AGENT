"""Inspect the new summoner-spell tools and sanity-check their arithmetic.

    python scripts/check_spell_tools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import metrics, tools  # noqa: E402

print("=== 总体分布（按出场队伍数排序）===")
result = tools.get_spell_stats(sort_by="picked_teams")
print(f"  总技能数: {result['total_spells']}   返回: {len(result['results'])}")
print()
print(f"  {'ID':>3} {'名称':<8} {'队伍数':>8} {'出场率%':>8} {'胜场':>7} {'被选胜率%':>10}")
for row in result["results"]:
    print(
        f"  {row['spell_id']:>3} {str(row['spell_name']):<8} {row['picked_teams']:>8,} "
        f"{row['pick_rate']['value']:>8.2f} {row['wins']:>7,} {row['win_rate_when_picked']['value']:>10.2f}"
    )

print()
print("=== 分母核对 ===")
rows = metrics.load_matches(PROJECT_ROOT / "ShuJuYuan" / "games.csv")
print(f"  去重后比赛数: {len(rows):,}")
print(f"  队伍出现次数（应为 2×比赛数）: {len(rows) * 2:,}")
top = result["results"][0]
print(f"  出现最多的技能 {top['spell_name']}（{top['spell_id']}）:")
print(f"    队伍数: {top['picked_teams']:,}")
print(f"    出场率: {top['picked_teams']:,} / {len(rows) * 2:,} = {top['pick_rate']['value']:.4f}%")
print(f"    被选时胜率: {top['wins']:,} / {top['picked_teams']:,} = {top['win_rate_when_picked']['value']:.4f}%")

# 独立重算：所有技能的队伍数合计应等于 20 列去重后按队伍统计的总和
print()
print("=== 独立复核：技能队伍数合计 ===")
recomputed = 0
for row in rows:
    for team in (1, 2):
        recomputed += len(metrics._spells_for_team(row, team))
total_picked = sum(item["picked_teams"] for item in result["results"])
print(f"  逐队去重计数合计: {recomputed:,}")
print(f"  各技能队伍数合计: {total_picked:,}")
print(f"  一致: {recomputed == total_picked}")

print()
print("=== 按 ID 与中文名查询应得到同一结果 ===")
by_id = tools.get_spell_stats("4")
by_name = tools.get_spell_stats("闪现")
print(f"  ID '4'   -> {json.dumps(by_id['results'], ensure_ascii=False)[:120]}")
print(f"  名称 闪现 -> {json.dumps(by_name['results'], ensure_ascii=False)[:120]}")
print(f"  结果一致: {by_id['results'] == by_name['results']}")

print()
print("=== 无法识别的技能应给出明确提示，而非空结果 ===")
unknown = tools.get_spell_stats("不存在的技能")
print(f"  results: {unknown['results']}   note: {unknown.get('note')}")

print()
print("=== 有/无技能对比 ===")
split = tools.get_spell_win_split("闪现")
print(f"  {split['spell_name']}（{split['spell_id']}）")
print(f"    有: {split['with_spell']['wins']:,} / {split['with_spell']['teams']:,} = {split['with_spell']['win_rate']['value']:.4f}%")
print(f"    无: {split['without_spell']['wins']:,} / {split['without_spell']['teams']:,} = {split['without_spell']['win_rate']['value']:.4f}%")
print(f"    两侧队伍数合计: {split['with_spell']['teams'] + split['without_spell']['teams']:,}（应等于 2×比赛数）")
print(f"    caveat: {split.get('caveat')}")

print()
print("=== 排序键 ===")
for key in ("picked_teams", "pick_rate", "win_rate_when_picked"):
    rows_sorted = tools.get_spell_stats(sort_by=key)["results"]
    # `picked_teams` is a plain int; the rate fields are numerator/denominator dicts.
    values = [
        row[key] if isinstance(row[key], int) else row[key]["value"]
        for row in rows_sorted
    ]
    print(
        f"  {key:22} 降序: {values == sorted(values, reverse=True)}   "
        f"首位: {rows_sorted[0]['spell_name']} ({values[0]:.4f})"
    )
