"""Verify every figure stated in kbase/conflict_item_catalog_drift.md.

The document was written from computed values, but prose is transcribed by hand and
numbers in prose rot silently. This recomputes each claim from the Data Dragon
snapshots and compares it against the document text, so a wrong or stale figure fails
loudly instead of misleading a reader (and the Agent, since this file is indexed).

    python scripts/verify_item_drift.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DDRAGON = PROJECT_ROOT / "ShuJuYuan" / "ddragon"
DOC = PROJECT_ROOT / "kbase" / "conflict_item_catalog_drift.md"
SR_MAP = "11"
VERSIONS = ["7.11.1", "7.12.1", "7.13.1", "7.14.1", "7.15.1", "7.16.1", "7.17.1", "7.18.1"]


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checked = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        self.checked += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
        if not ok:
            self.failures.append(label)


def load(version: str) -> dict[str, dict]:
    payload = json.loads((DDRAGON / version / "zh_CN" / "item.json").read_text(encoding="utf-8"))
    # Identical filter to domain_item_catalog_7.17.1.md so the counts are comparable.
    return {
        item_id: entry
        for item_id, entry in (payload.get("data") or {}).items()
        if (entry.get("maps") or {}).get(SR_MAP)
    }


def main() -> int:
    report = Report()
    text = DOC.read_text(encoding="utf-8")
    catalogues = {version: load(version) for version in VERSIONS}

    print("=== 文档与快照的一致性 ===")

    # Every patch name must appear in the document's source list.
    missing_versions = [v for v in VERSIONS if v not in text]
    report.check(not missing_versions, "8 个补丁版本都在文档中列出", f"缺 {missing_versions}")

    # The per-patch table has three columns: maps['11'] total, priced, unpriced.
    for version, items in catalogues.items():
        priced = sum(1 for entry in items.values() if (entry.get("gold") or {}).get("total", 0) > 0)
        unpriced = len(items) - priced
        pattern = (
            rf"\|\s*{re.escape(version)}\s*\|\s*\*{{0,2}}(\d+)\*{{0,2}}\s*\|"
            rf"\s*\*{{0,2}}(\d+)\*{{0,2}}\s*\|\s*(\d+)\s*\|"
        )
        match = re.search(pattern, text)
        if not match:
            report.check(False, f"{version} 的三个计数出现在文档表格中")
            continue
        stated_total, stated_priced, stated_unpriced = (int(match.group(i)) for i in (1, 2, 3))
        for label, stated, computed in (
            ("maps[11] 条目数", stated_total, len(items)),
            ("有价条目数", stated_priced, priced),
            ("无价条目数", stated_unpriced, unpriced),
        ):
            report.check(
                stated == computed,
                f"{version} {label} = {computed}",
                f"文档写 {stated}" if stated != computed else "",
            )

    first, last = catalogues[VERSIONS[0]], catalogues[VERSIONS[-1]]
    added = sorted(set(last) - set(first), key=int)
    removed = sorted(set(first) - set(last), key=int)

    report.check(len(added) == 10, "期间新增 10 件", f"计算得 {len(added)}")
    report.check(not removed, "期间移除 0 件", f"计算得 {len(removed)}")
    report.check("新增 10 件、移除 0 件" in text, "文档写明新增/移除数")

    # Each added item must be listed with its computed total price.
    print()
    print("=== 新增装备的 ID 与总价 ===")
    for item_id in added:
        entry = last[item_id]
        total = (entry.get("gold") or {}).get("total")
        # The row must carry both the id and the price.
        row = re.search(rf"\|\s*`{item_id}`\s*\|\s*([^|]+)\|\s*(\d+)\s*\|", text)
        if not row:
            report.check(False, f"`{item_id}` 在新增表中列出（含总价）")
            continue
        stated_name = row.group(1).strip()
        stated_total = int(row.group(2))
        report.check(
            stated_total == total and stated_name == entry["name"],
            f"`{item_id}` {entry['name']} 总价 {total}",
            f"文档写 {stated_name} {stated_total}"
            if (stated_total != total or stated_name != entry["name"])
            else "",
        )

    # Price changes: count them and confirm each documented row.
    print()
    print("=== 价格变更 ===")
    price_changes: list[tuple[str, str, str, int, int]] = []
    stat_changes: list[tuple[str, str, str]] = []
    renames: list[tuple[str, str, str]] = []
    for previous, current in zip(VERSIONS, VERSIONS[1:], strict=False):
        before, after = catalogues[previous], catalogues[current]
        for item_id in sorted(set(before) & set(after), key=int):
            b, a = before[item_id], after[item_id]
            if b.get("name") != a.get("name"):
                renames.append((item_id, b["name"], a["name"]))
                continue
            gb = (b.get("gold") or {}).get("total")
            ga = (a.get("gold") or {}).get("total")
            if gb != ga:
                price_changes.append((previous, current, item_id, gb, ga))
            elif (b.get("stats") or {}) != (a.get("stats") or {}):
                stat_changes.append((previous, current, item_id))

    report.check(
        len(price_changes) == 16,
        f"价格变更 16 次",
        f"计算得 {len(price_changes)}",
    )
    report.check("价格变更 16 次" in text, "文档写明价格变更次数", "")

    for previous, current, item_id, before_price, after_price in price_changes:
        name = catalogues[current][item_id]["name"]
        row = re.search(
            rf"\|\s*{re.escape(previous)}\s*→\s*{re.escape(current)}\s*\|\s*`{item_id}`\s*\|"
            rf"\s*([^|]+?)\s*\|\s*(\d+)\s*→\s*(\d+)\s*\|",
            text,
        )
        if not row:
            report.check(False, f"价格行 {item_id} ({previous}→{current}) 存在")
            continue
        report.check(
            row.group(1).strip() == name
            and int(row.group(2)) == before_price
            and int(row.group(3)) == after_price,
            f"价格 {item_id} {name}: {before_price} → {after_price}",
            f"文档写 {row.group(1).strip()} {row.group(2)}→{row.group(3)}"
            if (row.group(1).strip() != name
                or int(row.group(2)) != before_price
                or int(row.group(3)) != after_price)
            else "",
        )

    print()
    print("=== 属性变更与改名 ===")
    report.check(len(stat_changes) == 12, "属性变更 12 次", f"计算得 {len(stat_changes)}")
    report.check(
        "属性变更（价格不变）12 次" in text,
        "文档写明属性变更 12 次",
    )
    # The three same-named enchant entries are why the count is 12 and not 10; the
    # document must say so, otherwise a reader keying on names would disagree.
    report.check(
        "会被合并成 1 条" in text,
        "文档说明按名称统计会漏计同名附魔",
    )
    for item_id, before_name, after_name in renames:
        report.check(
            f"`{item_id}`" in text and before_name in text and after_name in text,
            f"改名 {item_id}: {before_name} → {after_name}",
        )

    # The duplicate-name point is what justifies keying on item id, so verify it.
    print()
    print("=== 同名不同 ID 的附魔系列 ===")
    by_name: dict[str, list[str]] = {}
    for item_id, entry in catalogues["7.17.1"].items():
        by_name.setdefault(entry["name"], []).append(item_id)
    enchant_ids = sorted(by_name.get("附魔：熔渣巨人", []), key=int)
    report.check(
        enchant_ids == ["1401", "1409", "1413", "3672"],
        f"附魔：熔渣巨人有 4 个 ID {enchant_ids}",
    )
    for item_id in enchant_ids:
        report.check(f"`{item_id}`" in text, f"文档列出附魔 ID `{item_id}`")

    # 7.17.1 must agree with the dedicated catalogue document. That document uses a
    # narrower filter (maps['11'] AND gold.total > 0), so the count it shows is 193,
    # not 228 — verified against the document's own per-category headings.
    print()
    print("=== 与 domain_item_catalog_7.17.1.md 的口径一致性 ===")
    catalogue_path = PROJECT_ROOT / "kbase" / "domain_item_catalog_7.17.1.md"
    catalogue_doc = catalogue_path.read_text(encoding="utf-8")

    priced_7171 = sorted(
        item_id
        for item_id, entry in catalogues["7.17.1"].items()
        if (entry.get("gold") or {}).get("total", 0) > 0
    )
    report.check(len(priced_7171) == 193, "7.17.1 按 maps[11] + 有价 得 193 条")

    # Count the document's own entries: the categories must sum to that number.
    category_counts = [int(m) for m in re.findall(r"^## 分类：.+?（(\d+) 件）", catalogue_doc, re.M)]
    report.check(bool(category_counts), "目录文档有分类计数标题")
    report.check(
        sum(category_counts) == len(priced_7171),
        f"目录文档分类计数合计 = {len(priced_7171)}（与有价口径一致）",
        f"合计 {sum(category_counts)}",
    )
    entries = len(re.findall(r"^### ", catalogue_doc, re.M))
    report.check(
        entries == len(priced_7171),
        f"目录文档装备条目数 = {len(priced_7171)}",
        f"实际 {entries}",
    )
    report.check(
        "原始 262 条，其中 34 条不属于召唤师峡谷" in catalogue_doc,
        "目录文档的来源说明仍然正确（262 中 34 条非峡谷）",
    )
    # The drift document must state the catalogue's own scope, so the two are not
    # mistaken for disagreeing.
    report.check(
        re.search(r"\*\*193\*\*", text) is not None,
        "漂移文档写明目录口径为 193",
    )
    report.check(
        re.search(r"\*\*228\*\*", text) is not None,
        "漂移文档写明自身使用 228 口径",
    )

    # The 'purchasable' trap: it is not a proxy for "has a price". Both the overlap
    # and the exception count are stated in the document, so verify them.
    sr_7171 = catalogues["7.17.1"]
    zero_price = {i for i, e in sr_7171.items() if (e.get("gold") or {}).get("total", 0) <= 0}
    not_purchasable = {
        i for i, e in sr_7171.items() if not (e.get("gold") or {}).get("purchasable")
    }
    report.check(len(zero_price) == 35, "7.17.1 无价 35 条", f"计算得 {len(zero_price)}")
    report.check(
        len(not_purchasable) == 21, "7.17.1 purchasable=false 21 条", f"计算得 {len(not_purchasable)}"
    )
    report.check(
        len(zero_price & not_purchasable) == 18,
        "两者交集 18 条",
        f"计算得 {len(zero_price & not_purchasable)}",
    )
    report.check(
        len(zero_price - not_purchasable) == 17,
        "有 17 条 total=0 但 purchasable=true",
        f"计算得 {len(zero_price - not_purchasable)}",
    )
    report.check("只有 18 条重叠" in text, "文档写明两个集合只重叠 18 条")
    report.check("17 条 `total = 0` 却标着" in text, "文档写明 purchasable 不可当有价用")

    print()
    print("=" * 72)
    if report.failures:
        print(f"RESULT: FAIL — {len(report.failures)}/{report.checked} 项未通过")
        for item in report.failures:
            print(f"  - {item}")
        return 1
    print(f"RESULT: PASS — {report.checked}/{report.checked} 项与快照一致")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
