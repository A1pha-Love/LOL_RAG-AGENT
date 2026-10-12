"""Independently verify the CSV against the dataset's own metadata attachments.

`kbase/project_data_audit.md` lists this as the one outstanding verification item:
field definitions were confirmed from the dataset's official description, but never
cross-checked against the metadata files shipped with it. This script performs that
cross-check, so the claim can be retired or corrected with evidence.

Attachments (Riot Data Dragon format: {"type", "version", "data"}):

    ShuJuYuan/champion_info.json         champion, 7.17.2
    ShuJuYuan/champion_info_2.json       champion, 7.18.1  (contains the -1 sentinel)
    ShuJuYuan/summoner_spell_info.json   summoner, 7.17.2

    python scripts/verify_dataset_metadata.py            # report
    python scripts/verify_dataset_metadata.py --strict   # also fail on warnings
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import knowledge as kb  # noqa: E402
from lol_agent import metrics  # noqa: E402

DATA_DIR = PROJECT_ROOT / "ShuJuYuan"
CSV_PATH = DATA_DIR / "games.csv"
ATTACHMENTS = {
    "champion_7_17_2": DATA_DIR / "champion_info.json",
    "champion_7_18_1": DATA_DIR / "champion_info_2.json",
    "summoner_7_17_2": DATA_DIR / "summoner_spell_info.json",
}


class Findings:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
        if not ok:
            self.failures.append(label)
        return ok

    def warn(self, label: str) -> None:
        print(f"  [WARN] {label}")
        self.warnings.append(label)


def load_attachment(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("type", "version", "data"):
        if key not in payload:
            raise ValueError(f"{path.name}: missing {key!r}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args()

    findings = Findings()

    print("=" * 78)
    print("1. 元数据附件自身的完整性")
    print("=" * 78)
    attachments: dict[str, dict] = {}
    for name, path in ATTACHMENTS.items():
        if not path.is_file():
            findings.check(False, f"{name} 存在", str(path))
            continue
        payload = load_attachment(path)
        attachments[name] = payload
        entries = payload["data"]
        findings.check(
            bool(entries),
            f"{name}: type={payload['type']} version={payload['version']} 条目={len(entries)}",
        )
        # Every entry must be a dict carrying an id; otherwise it cannot be compared.
        malformed = [k for k, v in entries.items() if not isinstance(v, dict) or "id" not in v]
        findings.check(not malformed, f"{name}: 每个条目都有 id", f"异常 {malformed[:3]}" if malformed else "")

    champion_attachment = attachments.get("champion_7_17_2")
    champion_latest = attachments.get("champion_7_18_1")
    spell_attachment = attachments.get("summoner_7_17_2")
    if not (champion_attachment and spell_attachment):
        print("\n缺少必要的附件，无法继续。")
        return 1

    # ------------------------------------------------------------------ CSV shape
    print()
    print("=" * 78)
    print("2. CSV 形状与官方说明的一致性")
    print("=" * 78)
    rows = metrics.load_matches(CSV_PATH)
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        raw_count = sum(1 for _ in reader)

    findings.check(len(columns) == 61, "字段数为 61", f"实际 {len(columns)}")
    findings.check(raw_count == 51_490, "原始行数为 51,490", f"实际 {raw_count:,}")
    findings.check(len(rows) == 51_053, "去重后为 51,053 场", f"实际 {len(rows):,}")

    # The data dictionary claims these encodings. Verify against actual values.
    for column, allowed in (
        ("winner", {"1", "2"}),
        ("firstBlood", {"0", "1", "2"}),
        ("firstTower", {"0", "1", "2"}),
    ):
        seen = {str(row.get(column, "")).strip() for row in rows}
        findings.check(
            seen <= allowed,
            f"{column} 取值 ⊆ {sorted(allowed)}",
            f"实际 {sorted(seen)}",
        )

    findings.check(
        {str(row.get("seasonId", "")).strip() for row in rows} == {"9"},
        "seasonId 恒为 9（据此切片无意义）",
    )

    # ------------------------------------------------------- champion id coverage
    print()
    print("=" * 78)
    print("3. 英雄 ID：CSV vs 元数据附件")
    print("=" * 78)
    csv_ids: set[str] = set()
    ban_sentinel_rows = 0
    for row in rows:
        for team in (1, 2):
            csv_ids |= metrics._champions_for_team(row, team)
            for slot in range(1, 6):
                if str(row.get(f"t{team}_ban{slot}", "")).strip() == "-1":
                    ban_sentinel_rows += 1

    attachment_ids = {str(entry["id"]) for entry in champion_attachment["data"].values()}
    findings.check(
        csv_ids <= attachment_ids,
        f"CSV 全部英雄 ID 都能在附件中找到（{len(csv_ids)} 个）",
        f"附件未覆盖 {sorted(csv_ids - attachment_ids)[:5]}" if csv_ids - attachment_ids else "",
    )
    findings.check(
        ban_sentinel_rows > 0,
        "CSV 中确实存在 -1 禁用位哨兵",
        f"{ban_sentinel_rows:,} 处",
    )

    # The audit claims -1 is a placeholder named None in the dataset's own champion
    # info. champion_info.json (7.17.2) has no such entry, but the 7.18.1 attachment
    # does — so the claim is true of the companion file, not of every version.
    sentinel_in_7_17_2 = any(int(entry["id"]) < 0 for entry in champion_attachment["data"].values())
    sentinel_in_7_18_1 = any(int(entry["id"]) < 0 for entry in (champion_latest or {"data": {}})["data"].values())
    if sentinel_in_7_17_2:
        findings.check(True, "7.17.2 附件含负 id 哨兵")
    else:
        findings.warn("7.17.2 附件不含 -1 哨兵条目（7.18.1 附件才含）— 数据字典的表述需限定版本")
    if champion_latest is not None:
        findings.check(sentinel_in_7_18_1, "7.18.1 附件含负 id 哨兵（对应 CSV 的 -1）")

    # ---------------------------------------------------------- name mapping check
    print()
    print("=" * 78)
    print("4. 英雄名称映射：项目锁定快照 vs 数据集附件")
    print("=" * 78)
    mapping_path = PROJECT_ROOT / "docs" / "champion_mapping_7_17_1.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    project_names = {str(champion["id"]): champion for champion in mapping["champions"]}
    # The file records its origin in `source` ("Riot Data Dragon 7.17.1") and the
    # exact URL in `retrieved_from`, not in a dedicated version field.
    project_version = str(mapping.get("source", "")).split()[-1]
    findings.check(
        project_version == "7.17.1",
        "项目映射标注的版本为 7.17.1",
        f"source={mapping.get('source')!r}",
    )

    # English key is the stable comparator across versions; Chinese names can drift.
    mismatched_keys: list[tuple[str, str, str]] = []
    for champion_id, entry in champion_attachment["data"].items():
        champion_id = str(entry["id"])
        if champion_id not in project_names:
            continue
        project_key = project_names[champion_id].get("english_id")
        if project_key and project_key != entry["key"]:
            mismatched_keys.append((champion_id, project_key, entry["key"]))
    findings.check(
        not mismatched_keys,
        f"英文标识符与附件完全一致（{len(project_names)} 个）",
        f"不一致 {mismatched_keys[:3]}" if mismatched_keys else "",
    )

    print()
    print("  项目快照版本与附件版本差异：")
    versions = {name: payload["version"] for name, payload in attachments.items()}
    for name, version in versions.items():
        print(f"    {name:22} {version}")
    print(f"    {'docs/champion_mapping_7_17_1.json':22} {project_version}")
    if versions.get("champion_7_17_2") != project_version:
        # Measured with scripts/compare_snapshot_versions.py: the champion sets and
        # every English-key -> numeric-id mapping are identical across the two
        # adjacent patches, so this is a version label rather than a data difference.
        findings.warn(
            f"项目锁定 {project_version}，附件为 {versions.get('champion_7_17_2')}；"
            "已实测两者英雄集合与 ID 映射完全一致，故不影响名称映射"
        )

    # Does the mapping file itself agree with the attachment on name/title?
    differing_labels: list[tuple[str, str, str]] = []
    for champion_id, entry in champion_attachment["data"].items():
        champion_id = str(entry["id"])
        project = project_names.get(champion_id)
        if not project:
            continue
        # The attachment is English, the project snapshot is zh_CN, so only the
        # presence of an English identifier is comparable — not the labels.
        if not project.get("english_id"):
            differing_labels.append((champion_id, "english_id 缺失", ""))
    findings.check(not differing_labels, "项目映射为每个英雄都提供英文标识符")

    # ------------------------------------------------------------- spell coverage
    print()
    print("=" * 78)
    print("5. 召唤师技能 ID：CSV vs 元数据附件")
    print("=" * 78)
    csv_spells: set[str] = set()
    for row in rows:
        for team in (1, 2):
            csv_spells |= metrics._spells_for_team(row, team)

    spell_attachment_ids = {str(entry["id"]) for entry in spell_attachment["data"].values()}
    findings.check(
        csv_spells <= spell_attachment_ids,
        f"CSV 全部技能 ID 都能在附件中找到（{len(csv_spells)} 个）",
        f"附件未覆盖 {sorted(csv_spells - spell_attachment_ids)}" if csv_spells - spell_attachment_ids else "",
    )
    findings.check(
        len(csv_spells) == 9,
        "CSV 实际使用 9 个技能 ID",
        f"{sorted(csv_spells, key=int)}",
    )
    # The attachment holds more spells than the dataset uses; that is expected and
    # must not be mistaken for a coverage gap.
    findings.check(
        len(spell_attachment_ids) > len(csv_spells),
        "附件包含数据集中未出现的技能（属正常，非缺失）",
        f"附件 {len(spell_attachment_ids)} 个 / 使用 {len(csv_spells)} 个",
    )

    # ------------------------------------------------------------- index coverage
    print()
    print("=" * 78)
    print("6. 知识库索引与文档一致")
    print("=" * 78)
    if kb.INDEX_PATH.is_file():
        staleness = kb.check_index_staleness()
        findings.check(not staleness["stale"], "索引与知识库一致", str(staleness.get("reason")))
    else:
        findings.warn("索引不存在，跳过一致性检查")

    # ------------------------------------------------------------------- summary
    print()
    print("=" * 78)
    if findings.failures:
        print(f"RESULT: FAIL — {len(findings.failures)} 项失败, {len(findings.warnings)} 项警告")
        for item in findings.failures:
            print(f"  - {item}")
        return 1
    if findings.warnings and args.strict:
        print(f"RESULT: FAIL (--strict) — {len(findings.warnings)} 项警告")
        for item in findings.warnings:
            print(f"  - {item}")
        return 1
    print(f"RESULT: PASS — 全部核对通过（{len(findings.warnings)} 项警告）")
    for item in findings.warnings:
        print(f"  ! {item}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
