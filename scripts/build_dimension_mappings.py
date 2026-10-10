"""Build ID mappings from the fetched dimension data and audit CSV coverage.

Inputs (all produced by `fetch_dimension_data.py`):
  ShuJuYuan/champion_info.json, champion_info_2.json, summoner_spell_info.json
  ShuJuYuan/ddragon/<patch>/zh_CN/{champion,championFull,item,summoner}.json

Outputs:
  docs/mappings/summoner_spells.json   CSV spell ID -> zh_CN name (union of patches)
  docs/mappings/champions_by_version.json   champion ID -> name/title per patch
  reports/dimension_coverage.md + .json     coverage audit of every CSV ID vocabulary

Read-only with respect to the raw CSV; nothing here writes to ShuJuYuan/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import tools  # noqa: E402

RAW_ROOT = PROJECT_ROOT / "ShuJuYuan"
DDRAGON_ROOT = RAW_ROOT / "ddragon"
# Outputs live in docs/ directly (prefix `mapping_`) rather than a new
# docs/mappings/ subdirectory: creating directories inside the workspace
# subdirectories is restricted in this environment, while file writes are not.
MAPPINGS_DIR = PROJECT_ROOT / "docs"
REPORT_MD = PROJECT_ROOT / "reports" / "dimension_coverage.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "dimension_coverage.json"

SPELL_SUBKEYS = ("summoner_spell_info", "spells", "summonerSpells")
CHAMPION_SUBKEYS = ("champion_info", "champions", "data")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _normalize_pairs(payload: Any) -> dict[str, str]:
    """Best-effort extraction of {id: name} from the several shapes these files use."""
    out: dict[str, str] = {}

    def consider(key: Any, value: Any) -> None:
        if isinstance(key, (str, int)) and str(key).strip().lstrip("-").isdigit():
            if isinstance(value, str):
                out[str(key).strip()] = value.strip()
            elif isinstance(value, dict):
                for name_key in ("name", "title", "champion_name", "spell_name"):
                    if isinstance(value.get(name_key), str):
                        out[str(key).strip()] = value[name_key].strip()
                        return

    if isinstance(payload, dict):
        for sub in SPELL_SUBKEYS + CHAMPION_SUBKEYS:
            if isinstance(payload.get(sub), (dict, list)):
                consider_nested = payload[sub]
                if isinstance(consider_nested, dict):
                    for key, value in consider_nested.items():
                        consider(key, value)
                else:
                    for item in consider_nested:
                        if isinstance(item, dict):
                            for key_field in ("id", "key", "championId", "spellId"):
                                if key_field in item:
                                    consider(item[key_field], item)
                                    break
        for key, value in payload.items():
            consider(key, value)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                for key_field in ("id", "key", "championId", "spellId", "champion_id"):
                    if key_field in item:
                        consider(item[key_field], item)
                        break
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                consider(item[0], item[1])
    return out


def _ddragon_champions(version: str) -> dict[str, dict[str, Any]]:
    data = _load(DDRAGON_ROOT / version / "zh_CN" / "champion.json")["data"]
    return {str(entry["key"]): entry for entry in data.values()}


def _ddragon_spells(version: str) -> dict[str, str]:
    data = _load(DDRAGON_ROOT / version / "zh_CN" / "summoner.json")["data"]
    return {str(entry["key"]): entry["name"].strip() for entry in data.values()}


def main() -> int:
    print("=" * 96)
    print("STEP 1 — inspect the Kaggle companion files (dataset's own vocabulary)")
    print("=" * 96)
    kaggle_shapes: dict[str, Any] = {}
    for filename in ("champion_info.json", "champion_info_2.json", "summoner_spell_info.json"):
        path = RAW_ROOT / filename
        if not path.is_file():
            print(f"  MISSING {filename}")
            continue
        payload = _load(path)
        pairs = _normalize_pairs(payload)
        kaggle_shapes[filename] = pairs
        sample = list(pairs.items())[:3]
        print(f"  {filename}")
        print(f"     top-level keys : {list(payload)[:8] if isinstance(payload, dict) else f'list[{len(payload)}]'}")
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            first_key = next(iter(payload["data"]))
            print(f"     data['{first_key}'] = {json.dumps(payload['data'][first_key], ensure_ascii=False)[:220]}")
        print(f"     extracted pairs: {len(pairs)}  sample={sample}")
    print()

    print("=" * 96)
    print("STEP 2 — Data Dragon patches on disk")
    print("=" * 96)
    versions = sorted(p.name for p in DDRAGON_ROOT.iterdir() if p.is_dir())
    print(f"  patches: {', '.join(versions)}")
    per_version_champions = {v: _ddragon_champions(v) for v in versions}
    per_version_spells = {v: _ddragon_spells(v) for v in versions}
    for v in versions:
        print(f"  {v}: champions={len(per_version_champions[v])} spells={len(per_version_spells[v])}")
    print()

    print("=" * 96)
    print("STEP 3 — CSV ID vocabularies")
    print("=" * 96)
    rows = tools._rows()
    pick_ids: Counter[str] = Counter()
    ban_ids: Counter[str] = Counter()
    spell_ids: Counter[str] = Counter()
    for row in rows:
        for team in (1, 2):
            for slot in range(1, 6):
                pick = str(row.get(f"t{team}_champ{slot}id", "")).strip()
                if pick:
                    pick_ids[pick] += 1
                ban = str(row.get(f"t{team}_ban{slot}", "")).strip()
                if ban:
                    ban_ids[ban] += 1
                for sub in ("sum1", "sum2"):
                    spell = str(row.get(f"t{team}_champ{slot}_{sub}", "")).strip()
                    if spell:
                        spell_ids[spell] += 1
    print(f"  distinct pick IDs   : {len(pick_ids)}")
    print(f"  distinct ban IDs    : {len(ban_ids)}")
    print(f"  distinct spell IDs  : {len(spell_ids)} -> {sorted(spell_ids, key=int)}")

    champion_values = set(pick_ids) | set(ban_ids)
    print(f"  distinct champion values (pick∪ban): {len(champion_values)}")
    print()

    print("=" * 96)
    print("STEP 4 — cross-check: dataset CSVs vs Data Dragon")
    print("=" * 96)
    primary = "7.17.1"
    dd_primary = per_version_champions[primary]
    kaggle_champs = kaggle_shapes.get("champion_info.json", {}) | kaggle_shapes.get("champion_info_2.json", {})

    unmapped_ddragon = sorted(champion_values - set(dd_primary), key=lambda x: int(x))
    print(f"  CSV champion values missing from DDragon {primary}: {unmapped_ddragon or 'none'}")
    if kaggle_champs:
        unmapped_kaggle = sorted(champion_values - set(kaggle_champs), key=lambda x: int(x))
        print(f"  CSV champion values missing from Kaggle companion : {unmapped_kaggle or 'none'}")
        overlap = set(dd_primary) & set(kaggle_champs)
        conflicts = {
            cid: (dd_primary[cid]["name"], kaggle_champs[cid])
            for cid in sorted(overlap, key=int)
            if dd_primary[cid]["name"] != kaggle_champs[cid]
        }
        print(f"  DDragon vs Kaggle name conflicts on {len(overlap)} shared IDs: {len(conflicts)}")
        for cid, (a, b) in list(conflicts.items())[:8]:
            print(f"     id={cid:>4}  ddragon={a!r}  kaggle={b!r}")
    else:
        conflicts = {}

    spell_union: dict[str, str] = {}
    spell_sources: dict[str, list[str]] = {}
    for v in versions:
        for sid, name in per_version_spells[v].items():
            spell_union.setdefault(sid, name)
            spell_sources.setdefault(sid, []).append(v)
    for filename, pairs in kaggle_shapes.items():
        if "spell" in filename:
            for sid, name in pairs.items():
                spell_union.setdefault(sid, name)
                spell_sources.setdefault(sid, []).append(f"kaggle:{filename}")

    missing_spells = sorted(set(spell_ids) - set(spell_union), key=int)
    print(f"  CSV spell IDs missing from all spell sources: {missing_spells or 'none'}")
    print()

    print("=" * 96)
    print("STEP 5 — version drift (same champion, different label across patches)")
    print("=" * 96)
    drift: dict[str, dict[str, Any]] = {}
    for cid in sorted(set(per_version_champions[versions[0]]) & set(per_version_champions[versions[-1]]), key=int):
        names = {v: per_version_champions[v].get(cid, {}).get("name") for v in versions}
        titles = {v: per_version_champions[v].get(cid, {}).get("title") for v in versions}
        if len(set(names.values())) > 1 or len(set(titles.values())) > 1:
            drift[cid] = {"names": names, "titles": titles}
    print(f"  champions whose name/title changed across {versions[0]}..{versions[-1]}: {len(drift)}")
    for cid, info in drift.items():
        changed_names = {v: n for v, n in info["names"].items() if n != info["names"][versions[0]]}
        print(f"     id={cid:>4}  {versions[0]}={info['names'][versions[0]]!r} -> {changed_names}")

    added = sorted(set(per_version_champions[versions[-1]]) - set(per_version_champions[versions[0]]), key=int)
    print(f"  champions added between {versions[0]} and {versions[-1]}: {added}")
    print()

    print("=" * 96)
    print("STEP 6 — write derived mappings")
    print("=" * 96)
    if not MAPPINGS_DIR.is_dir():
        raise SystemExit(f"output directory missing: {MAPPINGS_DIR}")
    retrieved_at = json.loads((DDRAGON_ROOT / "_manifest.json").read_text(encoding="utf-8"))["updated_at"]

    spell_doc = {
        "generated_by": "scripts/build_dimension_mappings.py",
        "generated_at_retrieved_from": retrieved_at,
        "purpose": "Map the CSV summoner-spell ID columns (t{1,2}_champ{1..5}_sum{1,2}) to zh_CN names.",
        "license": "Riot Data Dragon static data; dataset companion files are CC0.",
        "sources": ["ShuJuYuan/ddragon/<patch>/zh_CN/summoner.json", "ShuJuYuan/summoner_spell_info.json"],
        "id_to_name": {sid: spell_union[sid] for sid in sorted(spell_union, key=int)},
        "csv_ids_used": {sid: spell_ids[sid] for sid in sorted(spell_ids, key=int)},
        "csv_coverage": f"{len(set(spell_ids) & set(spell_union))}/{len(spell_ids)}",
        "unmapped_csv_ids": missing_spells,
    }
    (MAPPINGS_DIR / "mapping_summoner_spells.json").write_text(
        json.dumps(spell_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  wrote docs/mapping_summoner_spells.json  ({len(spell_union)} IDs, CSV coverage {spell_doc['csv_coverage']})")

    champion_doc = {
        "generated_by": "scripts/build_dimension_mappings.py",
        "versions": versions,
        "primary_version": primary,
        "note": "Per-patch static snapshots. Never evidence of the patch a single match was played on.",
        "champions": {
            cid: {
                "english_id": per_version_champions[primary].get(cid, {}).get("id"),
                "name_by_version": {v: per_version_champions[v].get(cid, {}).get("name") for v in versions},
                "title_by_version": {v: per_version_champions[v].get(cid, {}).get("title") for v in versions},
            }
            for cid in sorted(set().union(*[set(c) for c in per_version_champions.values()]), key=int)
        },
        "label_drift_ids": sorted(drift, key=int),
        "added_ids": added,
        "csv_coverage_primary": f"{len(champion_values & set(dd_primary))}/{len(champion_values)}",
    }
    (MAPPINGS_DIR / "mapping_champions_by_version.json").write_text(
        json.dumps(champion_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  wrote docs/mapping_champions_by_version.json  ({len(champion_doc['champions'])} IDs across {len(versions)} patches)")

    manifest = json.loads((DDRAGON_ROOT / "_manifest.json").read_text(encoding="utf-8"))
    audit = {
        "generated_by": "scripts/build_dimension_mappings.py",
        "ddragon_patches": versions,
        "primary_patch": primary,
        "csv_distinct_pick_ids": len(pick_ids),
        "csv_distinct_ban_ids": len(ban_ids),
        "csv_distinct_champion_values": len(champion_values),
        "csv_distinct_spell_ids": len(spell_ids),
        "champion_coverage_ddragon_primary": champion_doc["csv_coverage_primary"],
        "unmapped_csv_champion_values": unmapped_ddragon,
        "spell_coverage": spell_doc["csv_coverage"],
        "unmapped_csv_spell_ids": missing_spells,
        "ddragon_vs_kaggle_name_conflicts": conflicts,
        "label_drift_ids": sorted(drift, key=int),
        "champions_added_during_range": added,
        "manifest_entry_count": manifest.get("entry_count"),
        "manifest_updated_at": manifest.get("updated_at"),
        "raw_file_hashes": {
            path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
            for path in sorted(DDRAGON_ROOT.rglob("*.json"))
        },
    }
    REPORT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 维度数据覆盖审计",
        "",
        f"- 生成方式：`scripts/build_dimension_mappings.py`（读 `ShuJuYuan/` 下的原始快照，不写回）",
        f"- Data Dragon 快照：{', '.join(versions)}；主版本 `{primary}`",
        f"- 原始文件清单：`ShuJuYuan/ddragon/_manifest.json`（{manifest.get('entry_count')} 个条目，含 sha256）",
        "",
        "## 英雄 ID 覆盖",
        "",
        f"- CSV 中不同的英雄取值（pick ∪ ban）：**{len(champion_values)}**",
        f"- 可被 Data Dragon {primary} 映射：**{champion_doc['csv_coverage_primary']}**",
        f"- 未映射：`{unmapped_ddragon or '无'}`（`-1` 是空禁用位哨兵值，不是英雄）",
        "",
        "## 召唤师技能 ID 覆盖（CSV 中此前完全未被使用的 20 列）",
        "",
        f"- CSV 中不同的技能 ID：**{len(spell_ids)}**",
        f"- 可映射：**{spell_doc['csv_coverage']}**",
        f"- 未映射：`{missing_spells or '无'}`",
        "",
        "| ID | 名称 | CSV 中出现次数 |",
        "|---|---|---:|",
    ]
    lines += [
        f"| `{sid}` | {spell_union[sid]} | {spell_ids[sid]:,} |"
        for sid in sorted(spell_ids, key=int)
        if sid in spell_union
    ]
    lines += [
        "",
        "## 跨版本标签漂移（可作 RAG 冲突样本的天然材料）",
        "",
        f"- {versions[0]} 与 {versions[-1]} 之间名称或称号发生变化的英雄：**{len(drift)}**",
        "",
        "| 英雄 ID | " + " | ".join(versions) + " |",
        "|---|" + "---|" * len(versions),
    ]
    for cid, info in drift.items():
        cells = " | ".join(str(info["names"].get(v)) for v in versions)
        lines.append(f"| `{cid}` | {cells} |")
    lines += [
        "",
        f"- 期间新加入的英雄 ID：`{added or '无'}`",
        "",
        "## 版本边界声明",
        "",
        "- 上述快照用于**名称映射、机制说明与版本对照**，不构成任何单场比赛的补丁证据；",
        "  CSV 不含逐场 patch 字段，任何按补丁切片的分析都必须另行标注不确定性。",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote reports/dimension_coverage.md + .json")
    print()
    print("=" * 96)
    print("DONE")
    print(f"  champion coverage : {champion_doc['csv_coverage_primary']}")
    print(f"  spell coverage    : {spell_doc['csv_coverage']}")
    print(f"  label drift ids   : {sorted(drift, key=int)}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
