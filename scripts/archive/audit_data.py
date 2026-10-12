from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Derived from this file's location, not hardcoded: an absolute path would break
# on any other machine and would leak a personal path into the generated report.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "ShuJuYuan" / "games.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "data_audit.md"
EVENT_COLUMNS = (
    "winner", "firstBlood", "firstTower", "firstInhibitor", "firstBaron",
    "firstDragon", "firstRiftHerald",
)
TEAM_NUMERIC_COLUMNS = (
    "t1_towerKills", "t1_inhibitorKills", "t1_baronKills", "t1_dragonKills",
    "t1_riftHeraldKills", "t2_towerKills", "t2_inhibitorKills", "t2_baronKills",
    "t2_dragonKills", "t2_riftHeraldKills",
)


def audit(csv_path: Path) -> str:
    counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = {name: Counter() for name in EVENT_COLUMNS}
    seen_ids: set[str] = set()
    duplicate_ids = 0
    malformed_rows = 0
    duration_min: int | None = None
    duration_max: int | None = None
    creation_min: int | None = None
    creation_max: int | None = None
    invalid_numeric: Counter[str] = Counter()
    headers: list[str] = []
    expected_columns = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = list(reader.fieldnames or [])
        expected_columns = len(headers)
        for row in reader:
            counts["rows"] += 1
            if None in row:
                malformed_rows += 1
            for name in headers:
                value = row.get(name)
                if value is None or value.strip() == "":
                    missing[name] += 1
            game_id = (row.get("gameId") or "").strip()
            if game_id:
                if game_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(game_id)
            for name in EVENT_COLUMNS:
                value = (row.get(name) or "").strip()
                if value:
                    value_counts[name][value] += 1
            for name in TEAM_NUMERIC_COLUMNS:
                value = (row.get(name) or "").strip()
                if value:
                    try:
                        int(value)
                    except ValueError:
                        invalid_numeric[name] += 1
            try:
                duration = int(row["gameDuration"])
                duration_min = duration if duration_min is None else min(duration_min, duration)
                duration_max = duration if duration_max is None else max(duration_max, duration)
            except (KeyError, TypeError, ValueError):
                invalid_numeric["gameDuration"] += 1
            try:
                creation = int(row["creationTime"])
                creation_min = creation if creation_min is None else min(creation_min, creation)
                creation_max = creation if creation_max is None else max(creation_max, creation)
            except (KeyError, TypeError, ValueError):
                invalid_numeric["creationTime"] += 1

    def table(counter: Counter[str]) -> str:
        if not counter:
            return "|（无）|0|\n|---|---:|"
        return "| 值 | 数量 |\n|---|---:|\n" + "".join(
            f"| `{key}` | {count} |\n" for key, count in sorted(counter.items())
        )

    time_range = "无法解析"
    if creation_min is not None and creation_max is not None:
        time_range = (
            f"{datetime.fromtimestamp(creation_min / 1000, tz=timezone.utc).isoformat()} 至 "
            f"{datetime.fromtimestamp(creation_max / 1000, tz=timezone.utc).isoformat()}（按毫秒时间戳试算，需核实字段定义）"
        )

    # Report a repository-relative path: an absolute one would leak the
    # machine-specific location this audit happened to run from.
    try:
        display_path = csv_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(csv_path)

    lines = [
        "# 数据源初步审计报告",
        "",
        f"- 文件：`{display_path}`",
        f"- 文件大小：{csv_path.stat().st_size:,} bytes",
        f"- 审计时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- 原始文件以只读方式打开；本脚本不写回或转换源 CSV。",
        "",
        "## 文件结构",
        "",
        f"- 数据行数：{counts['rows']:,}",
        f"- 字段数：{expected_columns}",
        f"- 列名：{', '.join(f'`{name}`' for name in headers)}",
        f"- 字段数不一致的行：{malformed_rows:,}（DictReader 发现额外字段的行；缺字段另从空值统计观察）",
        f"- 重复非空 `gameId`：{duplicate_ids:,}",
        f"- `creationTime` 范围：{time_range}",
        f"- `gameDuration` 范围：{duration_min} 至 {duration_max} 秒" if duration_min is not None else "- `gameDuration` 范围：无法解析",
        "",
        "## 关键字段取值",
        "",
    ]
    for name, counter in value_counts.items():
        lines.extend([f"### `{name}`", "", table(counter), ""])
    lines.extend(["## 缺失字段", "", "| 字段 | 空值行数 |", "|---|---:|"])
    lines.extend(f"| `{name}` | {missing[name]:,} |" for name in headers if missing[name])
    if not any(missing.values()):
        lines.append("|（无）|0|")
    lines.extend(["", "## 非法数值", "", "| 字段 | 无法解析数量 |", "|---|---:|"])
    lines.extend(f"| `{name}` | {count:,} |" for name, count in sorted(invalid_numeric.items()))
    if not invalid_numeric:
        lines.append("|（无）|0|")
    lines.extend([
        "",
        "## 初步解释与待核实项",
        "",
        "- `winner` 以及各个 `first*` 字段的语义和值域必须结合数据集说明核验；本报告只列出观察到的取值，不推断编码含义。",
        "- `creationTime` 暂按毫秒 Unix 时间戳试算；正式分析前需确认单位、时区和数据覆盖赛季。",
        "- 当前文件只有对局级字段；不能据此分析玩家级、分路级、逐分钟经济或具体装备过程。",
        "- 胜率、Ban 率、登场率等指标的定义和分母需在后续数据字典中明确。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="只读检查 League of Legends games.csv 并生成 Markdown 报告")
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if not args.csv_path.is_file():
        raise SystemExit(f"找不到 CSV 文件：{args.csv_path}")
    report = audit(args.csv_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"审计完成：{args.output}")
    print(f"CSV：{args.csv_path}")


if __name__ == "__main__":
    main()
