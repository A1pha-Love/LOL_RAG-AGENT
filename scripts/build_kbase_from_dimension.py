"""Generate knowledge-base Markdown documents from the fetched dimension data.

Turns Data Dragon static snapshots into section-aware Markdown with YAML front
matter, so the retrieval layer can chunk them by heading and cite version,
source and license. Nothing here touches the raw CSV.

Output tree (kept at the workspace root: nested subdirectories are restricted
in this environment, root-level ones are not):

    kbase/00_scheme.md
    kbase/domain_champion_mechanics_7.17.1.md
    kbase/domain_item_catalog_7.17.1.md
    kbase/domain_summoner_spells.md
    kbase/domain_version_timeline.md
    kbase/conflict_champion_labels.md
    kbase/conflict_urgot_rename.md
    kbase/conflict_champion_additions.md
    kbase/meta_dataset_provenance.md

Run after `fetch_dimension_data.py` and `build_dimension_mappings.py`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import tools  # noqa: E402

RAW_ROOT = PROJECT_ROOT / "ShuJuYuan"
DDRAGON_ROOT = RAW_ROOT / "ddragon"
MANIFEST_PATH = DDRAGON_ROOT / "_manifest.json"
KBASE = PROJECT_ROOT / "kbase"
MAPPING_SPELLS = PROJECT_ROOT / "docs" / "mapping_summoner_spells.json"
MAPPING_CHAMPIONS = PROJECT_ROOT / "docs" / "mapping_champions_by_version.json"

PRIMARY = "7.17.1"
CHAMPION_CHUNK_LIMIT = 6  # champions per document; keeps files review-able

TAG_RE = re.compile(r"<[^>]+>")
TEMPLATE_RE = re.compile(r"\{\{\s*[^}]*?\s*\}\}")


def _clean(text: str) -> str:
    text = TAG_RE.sub("", text or "")
    text = TEMPLATE_RE.sub("…", text)
    return re.sub(r"\s+", " ", text).strip()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _front_matter(*, doc_id: str, title: str, category: str, source_type: str,
                  version: str, source_url: str, license_: str, retrieved_at: str,
                  extra: dict[str, Any] | None = None) -> str:
    def quote(value: Any) -> str:
        text = str(value)
        return f'"{text}"' if any(ch in text for ch in ':#"') else text

    lines = [
        "---",
        f"doc_id: {doc_id}",
        f"title: {quote(title)}",
        f"category: {category}",
        f"source_type: {source_type}",
        f"version: {quote(version)}",
        f"source_url: {quote(source_url)}",
        f"license: {quote(license_)}",
        f"retrieved_at: {retrieved_at}",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"{key}: {quote(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _write(path: Path, text: str, stats: list[tuple[str, int, int]]) -> None:
    path.write_text(text, encoding="utf-8")
    stats.append((path.relative_to(PROJECT_ROOT).as_posix(), len(text), text.count("\n## ")))


def main() -> int:
    if not KBASE.is_dir():
        raise SystemExit(f"kbase directory missing: {KBASE}")
    manifest = _load(MANIFEST_PATH)
    retrieved_at = manifest["updated_at"][:10]
    versions = sorted(p.name for p in DDRAGON_ROOT.iterdir() if p.is_dir())

    champions = _load(DDRAGON_ROOT / PRIMARY / "zh_CN" / "championFull.json")["data"]
    by_key = {str(entry["key"]): entry for entry in champions.values()}
    spells = _load(DDRAGON_ROOT / PRIMARY / "zh_CN" / "summoner.json")["data"]
    spell_by_key = {str(entry["key"]): entry for entry in spells.values()}
    items = _load(DDRAGON_ROOT / PRIMARY / "zh_CN" / "item.json")
    item_data = items["data"]

    spell_map = _load(MAPPING_SPELLS)
    champion_map = _load(MAPPING_CHAMPIONS)

    stats: list[tuple[str, int, int]] = []
    src_base = f"https://ddragon.leagueoflegends.com/cdn/{PRIMARY}/data/zh_CN"

    # ------------------------------------------------------------------ scheme
    scheme = _front_matter(
        doc_id="kbase_scheme", title="知识库结构与分类约定", category="meta",
        source_type="self", version="1.0", source_url="internal",
        license_="project-internal", retrieved_at=retrieved_at,
    ) + f"""# 知识库结构与分类约定

本知识库把项目文档与英雄联盟官方静态数据统一为带元数据的 Markdown，供检索层按标题分块。

## 分类 category

- `project`：项目自有文档（指标字典、英雄映射说明、数据审计、验收记录）。
- `domain`：领域知识，来自 Riot Data Dragon 官方静态数据。
- `conflict`：刻意保留的版本冲突样本，用于检验引用是否准确。
- `meta`：关于知识库自身的说明（本文件、数据来源与许可）。
- `offtopic`：与项目无关的干扰文档，仅用于阈值校准。

## 元数据字段

- `doc_id`：稳定唯一标识，改名不改值。
- `title`：人类可读标题，会进入检索上下文的软提示。
- `category`：见上。
- `source_type`：`official` / `wiki` / `self` / `derived`。
- `version`：数据所属补丁版本，或 `n/a`。**这是判断答案适用版本的关键字段。**
- `source_url`：原始出处。
- `license`：许可与署名要求。
- `retrieved_at`：抓取日期。

## 版本边界（重要）

本知识库包含 {versions[0]} 至 {versions[-1]} 共 {len(versions)} 个补丁的静态快照。

这些快照用于**名称映射、机制说明与版本对照**，不构成任何单场比赛的补丁证据。
原始 CSV 不含逐场 patch 字段，因此任何"按补丁切片"的结论都必须显式标注不确定性。

## 与原始数据的关系

- 原始对局数据：`ShuJuYuan/games.csv`（51,053 场去重对局，不在本知识库内，不做向量化）。
- 维度数据快照：`ShuJuYuan/ddragon/<补丁>/zh_CN/`，抓取记录见 `ShuJuYuan/ddragon/_manifest.json`（含 sha256）。
- 派生映射：`docs/mapping_summoner_spells.json`、`docs/mapping_champions_by_version.json`。
"""
    _write(KBASE / "00_scheme.md", scheme, stats)

    # --------------------------------------------------- champion mechanics
    # Heading shape is deliberate: H2 = one champion (the retrieval entity),
    # H3 = sections inside it, no H4. That shape lets the chunker keep one
    # champion coherent instead of emitting a fragment per ability.
    ordered = sorted(by_key.items(), key=lambda kv: int(kv[0]))
    for page, start in enumerate(range(0, len(ordered), CHAMPION_CHUNK_LIMIT), start=1):
        batch = ordered[start:start + CHAMPION_CHUNK_LIMIT]
        ids = f"{batch[0][0]}-{batch[-1][0]}"
        page_title = f"英雄机制与技能（{PRIMARY}）第 {page} 批：ID {ids}"
        parts = [
            _front_matter(
                doc_id=f"domain_champion_mechanics_{PRIMARY}_{page:02d}",
                title=page_title,
                category="domain", source_type="official", version=PRIMARY,
                source_url=f"{src_base}/championFull.json",
                license_="Riot Games 官方静态数据（Data Dragon），非商业用途",
                retrieved_at=retrieved_at,
            ),
            f"# {page_title}\n\n",
            "本文件来自 Riot Data Dragon 官方静态数据，用于解释英雄定位、基础属性与技能效果。"
            "原始 CSV 的英雄字段只存数值 ID，名称与技能均来自本映射，不代表逐场比赛的补丁版本。\n",
        ]
        for cid, champ in batch:
            name = champ["name"]
            parts.append(f"\n## {name}（{champ['title']}）\n\n")
            parts.append(
                f"- 英雄 ID：`{cid}`\n"
                f"- 英文标识符：`{champ['id']}`\n"
                f"- 定位标签：{'、'.join(champ.get('tags') or []) or '无'}\n"
                f"- 资源类型：{champ.get('partype') or '无'}\n"
            )
            history = champion_map["champions"].get(cid, {}).get("name_by_version", {})
            distinct = sorted({v for v in history.values() if v})
            if len(distinct) > 1:
                parts.append(f"- 版本名称变化：{' / '.join(distinct)}（见冲突样本文档）\n")
            stats_block = champ.get("stats") or {}
            parts.append(
                "\n### 基础属性\n\n"
                f"| 属性 | 数值 |\n|---|---:|\n"
                f"| 生命值 | {stats_block.get('hp')} |\n"
                f"| 生命回复/5秒 | {stats_block.get('hpregen')} |\n"
                f"| 法力值 | {stats_block.get('mp')} |\n"
                f"| 法力回复/5秒 | {stats_block.get('mpregen')} |\n"
                f"| 攻击力 | {stats_block.get('attackdamage')} |\n"
                f"| 攻击速度偏移 | {stats_block.get('attackspeedoffset')} |\n"
                f"| 护甲 | {stats_block.get('armor')} |\n"
                f"| 魔法抗性 | {stats_block.get('spellblock')} |\n"
                f"| 移动速度 | {stats_block.get('movespeed')} |\n"
                f"| 攻击距离 | {stats_block.get('attackrange')} |\n"
            )
            passive = champ.get("passive") or {}
            parts.append(
                f"\n### 被动：{passive.get('name', '无')}\n\n"
                f"{_clean(passive.get('description', ''))}\n"
            )
            parts.append("\n### 技能\n")
            for index, spell in enumerate(champ.get("spells") or [], start=1):
                parts.append(
                    f"\n**{index}. {spell.get('name')}**\n\n"
                    f"{_clean(spell.get('description', ''))}\n\n"
                    f"- 冷却：{spell.get('cooldownBurn')} 秒\n"
                    f"- 消耗：{spell.get('costBurn')}（{_clean(spell.get('costType', '')) or '无'}）\n"
                    f"- 射程：{spell.get('rangeBurn')}\n"
                )
            if champ.get("allytips"):
                parts.append("\n### 使用提示\n\n" + "\n".join(f"- {_clean(t)}" for t in champ["allytips"]) + "\n")
            if champ.get("enemytips"):
                parts.append("\n### 对抗提示\n\n" + "\n".join(f"- {_clean(t)}" for t in champ["enemytips"]) + "\n")
        _write(KBASE / f"domain_champion_mechanics_{PRIMARY}_{page:02d}.md", "".join(parts), stats)

    # ------------------------------------------------------------ item catalog
    # item.json has no per-item group field; the first tag is the practical
    # classifier. `maps['11']` marks availability on Summoner's Rift.
    tag_titles = {
        "Boots": "鞋子", "Boots": "鞋子", "Consumable": "消耗品", "Trinket": "饰品",
        "GoldPer": "工资装", "Lane": "对线", "Jungle": "打野", "Health": "生命",
        "HealthRegen": "生命回复", "Armor": "护甲", "SpellBlock": "魔抗",
        "Damage": "攻击", "AttackSpeed": "攻速", "CriticalStrike": "暴击",
        "LifeSteal": "生命偷取", "SpellDamage": "法术", "Mana": "法力",
        "ManaRegen": "法力回复", "CooldownReduction": "冷却缩减", "MagicPenetration": "法术穿透",
        "ArmorPenetration": "护甲穿透", "SpellVamp": "法术吸血", "NonbootsMovement": "移动速度",
        "Slow": "减速", "Tenacity": "韧性", "Aura": "光环", "Active": "主动",
        "Stealth": "潜行", "OnHit": "命中效果", "NonItem": "非装备",
    }
    by_group: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    skipped_not_sr = 0
    for item_id, item in item_data.items():
        if not (item.get("maps") or {}).get("11"):
            skipped_not_sr += 1
            continue
        if (item.get("gold") or {}).get("total", 0) <= 0:
            continue
        tags = item.get("tags") or ["其他"]
        group = tags[0] if tags[0] != "NonItem" else "其他"
        by_group[group].append((item_id, item))

    parts = [
        _front_matter(
            doc_id=f"domain_item_catalog_{PRIMARY}",
            title=f"装备目录（{PRIMARY}）",
            category="domain", source_type="official", version=PRIMARY,
            source_url=f"{src_base}/item.json",
            license_="Riot Games 官方静态数据（Data Dragon），非商业用途",
            retrieved_at=retrieved_at,
        ),
        f"# 装备目录（{PRIMARY}）\n",
        f"本文件收录 Data Dragon {PRIMARY} 中在召唤师峡谷可用的装备条目"
        f"（原始 {len(item_data)} 条，其中 {skipped_not_sr} 条不属于召唤师峡谷），"
        "用于解释装备属性、分类与合成关系。装备数据不在原始对局 CSV 内，"
        "因此不能据此回答某场对局的具体出装。\n",
    ]
    for group in sorted(by_group, key=lambda g: -len(by_group[g])):
        entries = by_group[group]
        parts.append(f"\n## 分类：{tag_titles.get(group, group)}（{len(entries)} 件）\n")
        for item_id, item in sorted(entries, key=lambda kv: (kv[1].get("gold") or {}).get("total", 0), reverse=True):
            gold = (item.get("gold") or {}).get("total")
            parts.append(
                f"\n### {item.get('name')}\n\n"
                f"- 物品 ID：`{item_id}`\n"
                f"- 总价：{gold} 金币\n"
                f"- 标签：{'、'.join(item.get('tags') or []) or '无'}\n"
                f"- 说明：{_clean(item.get('plaintext') or item.get('description', ''))}\n"
            )
    _write(KBASE / f"domain_item_catalog_{PRIMARY}.md", "".join(parts), stats)

    # --------------------------------------------------------- summoner spells
    used = spell_map["csv_ids_used"]
    parts = [
        _front_matter(
            doc_id="domain_summoner_spells",
            title=f"召唤师技能目录（{PRIMARY}）与数据中的实际使用",
            category="domain", source_type="official", version=PRIMARY,
            source_url=f"{src_base}/summoner.json",
            license_="Riot Games 官方静态数据（Data Dragon），非商业用途",
            retrieved_at=retrieved_at,
        ),
        f"# 召唤师技能目录（{PRIMARY}）与数据中的实际使用\n",
        "原始对局 CSV 用 `t{1,2}_champ{1..5}_sum1/sum2` 保存召唤师技能的**数值 ID**，不含名称。"
        "本文件给出官方中文名称，并标注该技能在本数据集 51,053 场对局中的实际出现次数。\n",
        "\n## 本数据集中实际出现的技能\n\n",
        "| ID | 名称 | 出现次数 |\n|---|---|---:|\n",
    ]
    for spell_id, count in sorted(used.items(), key=lambda kv: -kv[1]):
        name = spell_map["id_to_name"].get(spell_id, "未映射")
        parts.append(f"| `{spell_id}` | {name} | {count:,} |\n")
    parts.append(f"\n覆盖情况：{spell_map['csv_coverage']} 个 CSV 技能 ID 均可映射。\n")
    parts.append("\n## 全部技能条目\n")
    for spell_id, entry in sorted(spell_by_key.items(), key=lambda kv: int(kv[0])):
        parts.append(
            f"\n### {entry.get('name')}\n\n"
            f"- 技能 ID：`{spell_id}`\n"
            f"- 英文标识符：`{entry.get('id')}`\n"
            f"- 解锁等级：{entry.get('summonerLevel')}\n"
            f"- 冷却：{entry.get('cooldownBurn')} 秒\n"
            f"- 射程：{entry.get('rangeBurn')}\n"
            f"- 说明：{_clean(entry.get('description', ''))}\n"
        )
    _write(KBASE / "domain_summoner_spells.md", "".join(parts), stats)

    # --------------------------------------------------------- version timeline
    per_version_counts = {}
    for version in versions:
        data = _load(DDRAGON_ROOT / version / "zh_CN" / "champion.json")["data"]
        per_version_counts[version] = len(data)
    added = champion_map["added_ids"]
    added_names = {cid: champion_map["champions"][cid]["name_by_version"] for cid in added}
    parts = [
        _front_matter(
            doc_id="domain_version_timeline",
            title=f"补丁时间线：{versions[0]} 至 {versions[-1]} 的英雄池变化",
            category="domain", source_type="official", version=f"{versions[0]}~{versions[-1]}",
            source_url="https://ddragon.leagueoflegends.com/api/versions.json",
            license_="Riot Games 官方静态数据（Data Dragon），非商业用途",
            retrieved_at=retrieved_at,
        ),
        f"# 补丁时间线：{versions[0]} 至 {versions[-1]} 的英雄池变化\n",
        "原始数据集的 51,053 场对局覆盖 2017-06-08 至 2017-09-06（UTC），"
        f"对应下列 {len(versions)} 个补丁快照。**这是版本对照，不是逐场补丁证据。**\n",
        "\n## 各补丁英雄数量\n\n",
        "| 补丁 | 英雄数量 |\n|---|---:|\n",
    ]
    for version in versions:
        parts.append(f"| {version} | {per_version_counts[version]} |\n")
    parts.append("\n## 期间新加入的英雄\n")
    if added:
        for cid in added:
            names = added_names[cid]
            first = next((v for v in versions if names.get(v)), "未知")
            parts.append(f"\n### 英雄 ID {cid}：{names.get(first)}\n\n- 首次出现在快照：{first}\n- 英文标识符：{champion_map['champions'][cid]['english_id']}\n")
    else:
        parts.append("\n无。\n")
    parts.append("\n## 同时期其他变化\n")
    for cid in champion_map["label_drift_ids"]:
        entry = champion_map["champions"][cid]
        parts.append(
            f"\n### 英雄 ID {cid}：{entry['english_id']}\n\n"
            f"- 各版本名称：{json.dumps(entry['name_by_version'], ensure_ascii=False)}\n"
            f"- 各版本称号：{json.dumps(entry['title_by_version'], ensure_ascii=False)}\n"
        )
    _write(KBASE / "domain_version_timeline.md", "".join(parts), stats)

    # -------------------------------------------------------------- conflicts
    drift_ids = champion_map["label_drift_ids"]
    parts = [
        _front_matter(
            doc_id="conflict_champion_labels",
            title="版本冲突样本：同一英雄在不同补丁的名称与称号",
            category="conflict", source_type="derived", version=f"{versions[0]}~{versions[-1]}",
            source_url=f"{src_base}/champion.json",
            license_="Riot Games 官方静态数据（Data Dragon）；本文为派生对照",
            retrieved_at=retrieved_at,
        ),
        "# 版本冲突样本：同一英雄在不同补丁的名称与称号\n",
        "本文件刻意保留冲突信息，用于检验检索与引用是否准确：**同一英雄 ID 在不同补丁下名称可能不同**。"
        "任何引用本文的答案都必须同时说明所用补丁，否则视为口径不明。\n",
        f"\n## 冲突总览\n\n在 {versions[0]} 至 {versions[-1]} 之间，名称或称号发生过变化的英雄共 {len(drift_ids)} 个："
        + ("`" + "`、`".join(drift_ids) + "`" if drift_ids else "无") + "。\n",
    ]
    for cid in drift_ids:
        entry = champion_map["champions"][cid]
        parts.append(f"\n## 英雄 ID {cid}：{entry['english_id']}\n\n| 补丁 | 名称 | 称号 |\n|---|---|---|\n")
        for version in versions:
            parts.append(f"| {version} | {entry['name_by_version'].get(version)} | {entry['title_by_version'].get(version)} |\n")
        parts.append(
            "\n**引用要求**：回答涉及该英雄名称时，必须注明所依据的补丁版本；"
            "不得把某一补丁的名称当作全时间段通用名称。\n"
        )
    _write(KBASE / "conflict_champion_labels.md", "".join(parts), stats)

    urgot = champion_map["champions"].get("6")
    if urgot:
        renamed_at = next(
            (v for v in versions if urgot["name_by_version"].get(v) == "无畏战车"), "未知"
        )
        previous_at = next(
            (v for v in reversed(versions) if urgot["name_by_version"].get(v) == "首领之傲"), "未知"
        )
        parts = [
            _front_matter(
                doc_id="conflict_urgot_rename",
                title="版本冲突样本：英雄 ID 6（厄加特）的更名",
                category="conflict", source_type="derived", version=f"{previous_at}~{renamed_at}",
                source_url=f"{src_base}/champion.json",
                license_="Riot Games 官方静态数据（Data Dragon）；本文为派生对照",
                retrieved_at=retrieved_at,
            ),
            "# 版本冲突样本：英雄 ID 6（厄加特）的更名\n",
            "## 事实\n\n"
            f"- 中文名称在 {previous_at} 为「首领之傲」，自 {renamed_at} 起变为「无畏战车」。\n"
            f"- 中文称号「{urgot['title_by_version'].get(renamed_at)}」与英文标识符 `{urgot['english_id']}` 未变。\n"
            "- 名称变化伴随该英雄的重做（技能组同时改变），因此**同一 ID 在不同补丁下的技能描述也不同**。\n",
            "\n## 为什么这是重要的冲突样本\n\n"
            "项目文档 `docs/champion_mapping.md` 记录了这一变化，但未标注发生的确切补丁。"
            f"依据 Data Dragon 快照，可以确认变化发生在 {previous_at} 之后、{renamed_at} 之前。\n",
            "\n## 引用要求\n\n"
            "1. 检索到本文时，答案必须说明所依据的补丁版本。\n"
            "2. 不得据名称变化推断某场对局使用了哪个名称——CSV 无逐场 patch 字段。\n"
            "3. 讨论该英雄技能时，必须明确技能组来自哪个补丁。\n",
        ]
        _write(KBASE / "conflict_urgot_rename.md", "".join(parts), stats)

    parts = [
        _front_matter(
            doc_id="conflict_champion_additions",
            title="版本冲突样本：数据期间新加入的英雄",
            category="conflict", source_type="derived", version=f"{versions[0]}~{versions[-1]}",
            source_url=f"{src_base}/champion.json",
            license_="Riot Games 官方静态数据（Data Dragon）；本文为派生对照",
            retrieved_at=retrieved_at,
        ),
        "# 版本冲突样本：数据期间新加入的英雄\n",
        "原始数据集覆盖 2017-06-08 至 2017-09-06。在此期间有新英雄上线，"
        "因此**同一英雄的出场率会因上线时间而被稀释**，跨补丁比较英雄热度时必须注意这一点。\n",
        "\n## 新加入的英雄\n",
    ]
    for cid in added:
        entry = champion_map["champions"][cid]
        first = next((v for v in versions if entry["name_by_version"].get(v)), "未知")
        parts.append(
            f"\n### {entry['name_by_version'].get(first)}（英雄 ID {cid}）\n\n"
            f"- 英文标识符：`{entry['english_id']}`\n"
            f"- 首次出现在快照：{first}\n"
            f"- 含义：该英雄在数据期后段才可用，其全期出场率会系统性偏低。\n"
        )
    parts.append(
        "\n## 引用要求\n\n"
        "回答涉及这些英雄的出场率或胜率时，必须说明其可用时间短于数据全期，"
        "不得直接与其他全期英雄比较热度。\n"
    )
    _write(KBASE / "conflict_champion_additions.md", "".join(parts), stats)

    # ------------------------------------------------------------ meta provenance
    kaggle_entries = {k: v for k, v in manifest["entries"].items() if k.startswith("kaggle/")}
    parts = [
        _front_matter(
            doc_id="meta_dataset_provenance",
            title="数据来源、许可与版本边界",
            category="meta", source_type="self", version=f"{versions[0]}~{versions[-1]}",
            source_url="https://www.kaggle.com/datasets/datasnaek/league-of-legends",
            license_="原始数据集 CC0: Public Domain；Data Dragon 为 Riot 官方静态数据",
            retrieved_at=retrieved_at,
        ),
        "# 数据来源、许可与版本边界\n",
        "## 对局数据\n\n"
        "- 来源：Kaggle 数据集 `datasnaek/league-of-legends`（(LoL) League of Legends Ranked Games）。\n"
        "- 作者：Mitchell J；许可：**CC0: Public Domain**。\n"
        "- 本地文件 `ShuJuYuan/games.csv` 与该数据集发布的 games.csv 字节数一致（9,287,739 字节）。\n"
        "- 内容：2017-06-08 至 2017-09-06（UTC）的 EUW 排位对局，原始表 51,490 行、61 列，"
        "按 `gameId` 去重后 51,053 场。\n",
        "\n## 随数据集发布的配套文件\n\n"
        "| 文件 | 字节 | 用途 |\n|---|---:|---|\n"
        + "".join(
            f"| `{name.split('/')[-1]}` | {entry['bytes']:,} | 英雄与召唤师技能 ID 对照 |\n"
            for name, entry in sorted(kaggle_entries.items())
        ),
        "\n这些配套文件使用**英文**名称，与 Data Dragon 的中文名称互补；"
        "其中记录了一个 id 为 -1 的占位条目，对应 CSV 中表示空禁用位的 `-1` 哨兵值。\n",
        "\n## 维度数据\n\n"
        f"- 来源：Riot Data Dragon 官方静态数据，快照补丁 {'、'.join(versions)}。\n"
        "- 抓取记录：`ShuJuYuan/ddragon/_manifest.json`，含每个文件的 URL、字节数与 sha256。\n"
        "- 用途：英雄名称映射、技能与机制说明、装备与召唤师技能目录、版本对照。\n",
        "\n## 边界声明\n\n"
        "1. Data Dragon 快照**不构成任何单场比赛的补丁证据**；CSV 无逐场 patch 字段。\n"
        "2. 本数据集不可再生：Riot 官方比赛接口仅保留约两年的对局，2017 年数据无法重新抓取。\n"
        "3. 数据不含玩家标识、分路、逐分钟过程或装备选择，因此不能回答玩家级、分路级与出装问题。\n",
    ]
    _write(KBASE / "meta_dataset_provenance.md", "".join(parts), stats)

    print("=" * 96)
    print("GENERATED KNOWLEDGE-BASE DOCUMENTS")
    print("=" * 96)
    total_chars = 0
    total_sections = 0
    for name, chars, sections in sorted(stats):
        total_chars += chars
        total_sections += sections
        print(f"  {name:52} {chars:>9,} chars  {sections:>4} sections")
    print("-" * 96)
    print(f"  {'TOTAL':52} {total_chars:>9,} chars  {total_sections:>4} sections  ({len(stats)} documents)")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
