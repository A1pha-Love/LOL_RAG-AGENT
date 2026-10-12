"""Agent-facing, deterministic tools over the local ranked-match CSV."""

# ============================================================================
# 中文说明
#
# 本模块是项目的"数据层 + 工具层"：把只读的 games.csv 读进内存（带缓存），把中文名
# 或数字 ID 形式的参数解析成列里真正存的值，再调用 metrics.py 的纯函数算出数字，
# 最后补上英雄/召唤师技能的中文标签，作为工具的返回值交给模型。
#
# 在整体架构中的位置：
#     CSV -> tools.py（读取 / 缓存 / 参数解析 / 补标签）
#         -> metrics.py（口径定义与计算）
#         -> LangGraph 的 ToolNode -> 模型把数字组织成中文回答
# 模型只决定"调用哪个工具、以及怎么表述"，不参与数值计算（见 kbase/project_overview.md）。
#
# 为什么工具的参数非法时要抛异常，而不是"尽力返回一份空结果"：
#   工具节点是 ToolNode(AGENT_TOOLS, handle_tool_errors=True)（见 agent.py），这个
#   开关会把工具内部抛出的异常转成一条 tool 消息回灌给模型。实测中模型偶尔会传
#   limit=0 之类的非法参数；若不开该开关，异常会直接冒出 `invoke`，整轮问答直接失败；
#   开了之后模型能看到 "limit must be a positive integer" 并自行改正参数重试。
#   因此工具的校验保持严格——错误是被**呈现**给模型，而不是被悄悄吞掉。
#
# 唯一例外是"查不到的实体"（例如未识别的召唤师技能）：那不是参数格式错误，而是这个
# 名字不在映射表里，此时返回一条带 note 的说明性结果比抛异常更有用（模型可以据此
# 换一个精确名字再试），所以可以看到下面两条路径风格不同。
#
# 关于 docstring（本文件的统一约定）：
#   下面这些 `get_*` 函数的 docstring 会被 LangChain 当作**工具描述**发给模型，属于
#   提示词的一部分；改动它会改变每轮请求的 token 数与模型看到的信息。因此本文件里
#   所有面向模型的工具 docstring 一律保持英文原样不动，中文解释统一以 docstring
#   下方的 `#` 注释块给出。私有函数 `_*` 不暴露给模型，其中文说明或直接写成中文
#   docstring（新加的），或在保留原英文 docstring 的前提下追加中文段落。
#
# 缓存策略：
#   `_load_cached` 与 `_load_dataset_overview` 都用了 lru_cache，并把文件的
#   mtime_ns 与 size 一起放进缓存键。这样替换数据文件后无需重启进程就会重新读取，
#   也避免"忘了清缓存"导致返回过期数字。映射表（英雄/技能）是静态快照，直接
#   用 maxsize=1 缓存整表。
# ============================================================================

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import csv
import json
from pathlib import Path
from typing import Any

from . import metrics

# 三个固定的资源路径。PROJECT_ROOT 由本文件位置反推：src/lol_agent/tools.py 的
# parents 依次是 lol_agent、src、项目根，所以 parents[2] 才是仓库根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "ShuJuYuan" / "games.csv"
# 英雄名与技能名都是"静态快照映射"：CSV 里只有数字 ID，名称只用于查询与展示，
# 不代表逐场比赛的补丁版本（见 kbase/project_overview.md 的回答边界）。
CHAMPION_MAPPING_PATH = PROJECT_ROOT / "docs" / "champion_mapping_7_17_1.json"
SPELL_MAPPING_PATH = PROJECT_ROOT / "docs" / "mapping_summoner_spells.json"

# 工具名 -> 一句话说明。它是给人看的工具清单，不参与运行时逻辑：
# 模型看到的工具描述来自各函数自己的 docstring，而不是这张表。
# 注意：仓库内除本定义外没有其它代码引用它，若有文档生成脚本依赖，需要另行确认。
TOOL_DESCRIPTIONS = {
    "get_match_count": "Count distinct matches after deduplicating identical gameId rows.",
    "get_dataset_overview": "Return deterministic CSV row/column counts, unique game count, and duplicate row/group counts.",
    "get_champion_stats": "Get champion pick appearances and win rates by numeric champion ID or exact Chinese/English champion name; optionally filter one champion or return a sorted top N.",
    "get_ban_stats": "Get champion ban incidence by match; optionally filter by numeric champion ID or exact Chinese/English champion name, or return a sorted top N.",
    "get_first_event_win_rate": "Get the win rate among matches where a team achieved the requested first event.",
    "get_first_event_occurrence_rate": "Get the occurrence rate of a first event, including matches where it did not occur.",
    "get_average_game_duration": "Get the average positive, parseable game duration in seconds.",
    "get_spell_stats": "Get summoner-spell pick rates and win rates by spell ID or exact Chinese name; optionally filter one spell or return a sorted top N.",
    "get_spell_win_split": "Compare team win rates with versus without a specific summoner spell, over the same matches.",
    "get_team_objective_averages": "Get the mean end-of-match objective counts per team (towers, inhibitors, barons, dragons, rift heralds); optionally restrict to one objective.",
}


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime_ns: int, size: int) -> tuple[dict[str, str], ...]:
    """真正读 CSV 并缓存的函数；路径与文件元数据都是缓存键的一部分。

    参数:
        path: CSV 的绝对路径字符串。
        mtime_ns: 文件修改时间（纳秒），仅用于让缓存失效，函数体不使用它。
        size: 文件字节数，同样只用于让缓存失效。

    返回:
        去重后的行元组（用 tuple 是为了可哈希、不会被调用方就地改动）。

    为什么把 mtime/size 放进签名:
        lru_cache 只按参数缓存。把"文件是否变过"编码成参数后，替换数据文件会自然
        触发重新读取，无需手动清缓存或重启进程。
    """
    # File metadata is part of the cache key, so replacing the CSV refreshes the data.
    return tuple(metrics.load_matches(path))


def _rows(csv_path: str | Path | None = None) -> tuple[dict[str, str], ...]:
    """取出去重后的比赛行（默认取仓库内固定的数据集）。

    参数:
        csv_path: 可选的 CSV 路径；None 表示使用 DEFAULT_CSV_PATH。

    返回:
        去重后的行元组。

    异常:
        FileNotFoundError: 路径不存在或不是普通文件。测试与脚本依赖这个显式错误，
            而不是等到解析空文件时才发现问题。
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    # expanduser 支持 ~ 写法；resolve 把相对路径变绝对路径，保证同一个文件的不同
    # 写法在缓存里命中同一条记录（否则 "games.csv" 与绝对路径会各缓存一份）。
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Match CSV not found: {path}")
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=1)
def _champion_maps() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load the pinned 7.17.1 champion labels and exact-match name aliases.

    载入英雄映射表，返回两张表：

    返回:
        (by_id, by_name)
        by_id: 英雄 ID 字符串 -> 该英雄的完整记录（name / title / english_id / aliases）。
        by_name: 归一化后的名称 -> 英雄 ID 字符串；用于把中文名或英文名解析成 ID。

    说明:
        * 归一化只做 strip + casefold（大小写折叠），因此英文名大小写不敏感，
          但不做模糊匹配、不支持错别字——工具的参数是模型生成的，精确匹配能让
          "查不到"变成一个明确信号，而不是猜错一个英雄。
        * 该映射是 7.17.1 的中文快照，**不代表逐场比赛的补丁版本**，只用于查询与展示。
        * 同一个名字被映射到两个不同 ID 时直接抛错：那说明映射文件自身有歧义，
          静默择一会让统计指向错误的英雄。
    """
    data = json.loads(CHAMPION_MAPPING_PATH.read_text(encoding="utf-8"))
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, str] = {}
    for champion in data["champions"]:
        champion_id = str(champion["id"])
        by_id[champion_id] = champion
        # 中文名、称号、英文 ID 以及额外别名都可以用来查询；aliases 可能缺省，
        # 因此用 `or []` 兜底。
        for label in (champion["name"], champion["title"], champion["english_id"], *(champion.get("aliases") or [])):
            normalized = str(label).strip().casefold()
            existing = by_name.get(normalized)
            if existing is not None and existing != champion_id:
                raise ValueError(f"Duplicate champion name alias in mapping: {label}")
            by_name[normalized] = champion_id
    return by_id, by_name


def _resolve_champion_id(value: str | int) -> str | None:
    """Resolve a numeric ID or an exact Chinese/English champion label.

    把用户/模型给的英雄标识解析成 CSV 里真正使用的英雄 ID 字符串。

    参数:
        value: 数字 ID（str 或 int）或精确的中文名/英文名/称号/别名。

    返回:
        解析成功返回 ID 字符串；无法识别返回 None（调用方据此区分"查不到"）。

    说明:
        先查 ID 表再查名称表：数字形式的输入优先按 ID 解释，避免某个别名恰好是数字
        时被误当作名字。
    """
    token = str(value).strip()
    by_id, by_name = _champion_maps()
    return token if token in by_id else by_name.get(token.casefold())


def _add_champion_labels(results: list[dict[str, Any]]) -> None:
    """给统计结果就地补上英雄名称字段，供模型直接引用。

    参数:
        results: metrics.champion_stats / ban_stats 产出的结果项列表（就地修改）。

    说明:
        映射里没有的英雄（理论上不会出现）三个字段都填 None，而不是留空：
        显式的 null 能让模型看出"这个 ID 没有名字"，而不是以为字段被漏掉了。
    """
    by_id, _ = _champion_maps()
    for item in results:
        champion = by_id.get(str(item["champion_id"]))
        item["champion_name"] = champion["name"] if champion else None
        item["champion_title"] = champion["title"] if champion else None
        item["champion_english_id"] = champion["english_id"] if champion else None


@lru_cache(maxsize=1)
def _spell_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Load the pinned spell labels, plus exact-match aliases.

    The mapping file nests the lookup under ``id_to_name`` alongside provenance
    metadata, so the ids must be read from that key rather than the top level.

    载入召唤师技能映射表，返回两张表：

    返回:
        (by_id, by_name)
        by_id: 技能 ID 字符串 -> 中文名。
        by_name: 归一化后的中文名 -> 技能 ID 字符串。

    说明:
        与英雄映射文件不同，这个文件的顶层是元数据（generated_by / sources /
        csv_ids_used…），真正的映射在 `id_to_name` 键下面，所以必须从该键读取并
        校验其类型，否则会把元数据误当成映射。
        同样只做精确匹配（strip + casefold），不做模糊匹配。
    """
    data = json.loads(SPELL_MAPPING_PATH.read_text(encoding="utf-8"))
    id_to_name = data.get("id_to_name")
    if not isinstance(id_to_name, dict):
        raise ValueError(f"Mapping file has no id_to_name table: {SPELL_MAPPING_PATH}")
    by_id = {str(key): str(value) for key, value in id_to_name.items()}
    # 反向表：注意映射文件里 ID 35 与 36 的中文名相同（"已禁用召唤师技能"），
    # 因此按名称反查时字典只会保留其中一个（后写入的那个），这是该文件的固有歧义。
    by_name = {label.strip().casefold(): spell_id for spell_id, label in by_id.items()}
    return by_id, by_name


def _resolve_spell_id(value: str | int) -> str | None:
    """Resolve a numeric spell ID or an exact Chinese spell name.

    把召唤师技能标识解析成 CSV 里使用的技能 ID 字符串。

    参数:
        value: 数字 ID（str 或 int）或精确的中文技能名。

    返回:
        解析成功返回 ID 字符串；无法识别返回 None。

    说明:
        先查 ID 表、再查名称表，与英雄解析保持同一顺序。
    """
    token = str(value).strip()
    by_id, by_name = _spell_maps()
    if token in by_id:
        return token
    resolved = by_name.get(token.casefold())
    if resolved is None:
        return None
    # The mapping also lists retired/disabled spells (35, 36) that the dataset
    # never contains. Rejecting them here keeps "not in this dataset" distinct from
    # "not a spell", which would otherwise both look like an empty result.
    #
    # 注意：上面这段注释描述的"在此处拒绝 35/36"并没有体现在代码里——只要 35/36
    # 出现在 id_to_name 中，本函数就会把它们解析成合法 ID 返回。实测后果是：查询
    # 这两个 ID 时 requested_id 非 None，于是 get_spell_stats 走"已识别但在数据里
    # 没有"的分支，返回 results: [] 且**不带**"未识别"的 note。若确实需要拒绝，
    # 应在返回前显式排除这两个 ID（此处仅如实指出差异，未改动行为）。
    return resolved


def _add_spell_labels(results: list[dict[str, Any]]) -> None:
    """给技能统计结果就地补上中文名（映射缺失时为 None）。"""
    by_id, _ = _spell_maps()
    for item in results:
        item["spell_name"] = by_id.get(str(item["spell_id"]))


@lru_cache(maxsize=4)
def _load_dataset_overview(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    """Count raw rows, columns, unique matches, and duplicate ID groups.

    统计数据集的结构信息（原始行数、列数、去重后对局数、重复情况）。

    参数:
        path: CSV 绝对路径。
        mtime_ns / size: 只作为 lru_cache 的缓存键，函数体不使用。

    返回:
        含 source_file / raw_row_count / column_count / unique_match_count /
        duplicate_row_count / duplicate_game_id_groups /
        duplicate_records_identical 的 dict。

    为什么这里要单独再读一遍 CSV:
        "原始行数"必须逐行数出来（去重后的函数拿不到原始行数），而"去重后对局数"
        则复用 metrics.load_matches，保证与所有统计指标用的是**同一套**去重与冲突
        校验逻辑，不会出现两套口径各自漂移。
    """
    row_count = 0
    game_id_counts: Counter[str] = Counter()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            game_id = (row.get("gameId") or "").strip()
            # 缺 gameId 的行无法参与按对局去重的统计，直接报错并指出行号，便于定位
            # 数据文件问题。行号 = row_count + 1：row_count 统计的是**数据行**序号
            # （从 1 开始），而文件第 1 行是表头，所以物理行号要再加 1。
            if not game_id:
                raise ValueError(f"Missing gameId at CSV row {row_count + 1}")
            game_id_counts[game_id] += 1

    # Reuse the canonical loader to ensure duplicate gameIds have identical rows.
    unique_match_count = len(metrics.load_matches(path))
    # 注意"重复组数"与"重复行数"是两个不同的数：一个 gameId 出现 3 次时，
    # 多出 2 行但只算 1 个重复组（本数据为 437 行 / 424 组）。
    duplicate_groups = sum(count > 1 for count in game_id_counts.values())
    return {
        "source_file": Path(path).name,
        "raw_row_count": row_count,
        "column_count": len(columns),
        "unique_match_count": unique_match_count,
        "duplicate_row_count": row_count - unique_match_count,
        "duplicate_game_id_groups": duplicate_groups,
        # 该字段恒为 True，是因为 metrics.load_matches 遇到内容冲突的重复行会直接抛错：
        # 能走到这里就说明所有重复行逐字段一致。
        "duplicate_records_identical": True,
    }


def get_dataset_overview(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Return deterministic file structure and deduplication counts."""
    # ---- 中文说明 ----
    # 回答"数据规模有多大"这类问题：给出原始行数与去重后对局数两套数字。
    #
    # 参数:
    #     csv_path: 可选的自定义 CSV 路径，默认使用仓库内数据集。
    #
    # 返回:
    #     见 _load_dataset_overview；其中 source_file 在文件位于项目内时会被改写成
    #     相对项目根目录的 POSIX 风格路径。
    #
    # 为什么要把绝对路径改写成相对路径:
    #     结果会进入模型上下文并可能被展示给用户，绝对路径会泄露本机目录结构，而且
    #     换一台机器就失效；相对路径更稳定也更短。使用 as_posix() 保证跨平台输出
    #     一致（Windows 上也显示 / 而不是 \）。
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Match CSV not found: {path}")
    stat = path.stat()
    result = _load_dataset_overview(str(path), stat.st_mtime_ns, stat.st_size)
    if path.is_relative_to(PROJECT_ROOT):
        result["source_file"] = path.relative_to(PROJECT_ROOT).as_posix()
    return result


def _validate_limit(limit: int | None) -> None:
    """校验 limit 参数；非法时抛 ValueError，让模型能收到并改正。

    参数:
        limit: 期望返回的最大条数；None 表示使用调用处决定的默认上限。

    为什么显式排除 bool:
        Python 中 `True` 是 int 的子类，而 `True < 1` 为 False，因此只写
        `limit < 1` 是拦不住 `limit=True` 的——它会被静默当成 1 条返回。
        与其接受一个明显是类型错误的参数，不如直接报错让模型改正。
    """
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer")


# A ranking question is answered from the top few rows, so returning the whole
# champion table is pure waste: all 138 rows serialise to ~55,000 characters
# (~13x the top-10 payload), which the model must read before it can answer and
# which measurably delays the first token. Callers that genuinely want the full
# table can pass an explicit larger limit.
#
# 中文说明：榜单类问题只需要前几行，返回整张英雄表纯属浪费——138 行序列化后约
# 55,000 字符（约为 top-10 的 13 倍），模型必须先读完这些字符才能回答，实测会明显
# 推迟首字返回时间。确实需要整表时，调用方显式传一个更大的 limit 即可。
DEFAULT_RANKING_LIMIT = 20


def get_match_count(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Count unique matches."""
    # ---- 中文说明 ----
    # 返回去重后的唯一对局数（统计单位，见 metrics.match_count）。
    return metrics.match_count(_rows(csv_path))


def get_champion_stats(
    champion_id: str | int | None = None,
    sort_by: str = "appearances",
    limit: int | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return champion appearance and win statistics, optionally filtered/sorted.

    When no champion is named, results are ranked and capped at
    ``DEFAULT_RANKING_LIMIT`` unless an explicit ``limit`` is given.
    """
    # ---- 中文说明 ----
    # 英雄出场/胜率统计。两种调用形态：
    #   1. 指定 champion_id（数字 ID 或精确中英文名）：只返回该英雄一条（查不到则
    #      results 为空列表，同时 champion_id_filter 为 None，模型据此知道是"名字没
    #      认出来"而不是"这个英雄没数据"）；
    #   2. 不指定：按 sort_by 排序后取前 effective_limit 条，DEFAULT_RANKING_LIMIT
    #      作为默认截断长度（原因见上方注释）。
    #
    # 排序口径：先按指标值降序，再按英雄 ID 升序。第二个键是为了让"值相同"的结果
    # 顺序稳定——否则同一份数据两次调用可能给出不同的前 20 名，输出就不可复现了。
    # value 为 None（无样本）时按 -1 参与排序，因此永远排在有样本的英雄后面，
    # 这与"不拿无样本胜率去争榜首"的意图一致。
    #
    # 数字本身由 metrics.champion_stats 计算：appearances 的分母是全部对局（覆盖率），
    # win_rate 的分母是该英雄的出场场数（条件胜率）。两者不能混用。
    _validate_limit(limit)
    if sort_by not in {"appearances", "win_rate"}:
        raise ValueError("sort_by must be 'appearances' or 'win_rate'")
    effective_limit = limit if limit is not None else DEFAULT_RANKING_LIMIT
    all_stats = metrics.champion_stats(_rows(csv_path))
    requested_id = _resolve_champion_id(champion_id) if champion_id is not None else None
    if champion_id is not None:
        # 用 `requested_id in all_stats` 而不是直接取：未识别或数据中不存在的英雄
        # 返回空列表，而不是 KeyError（工具报错会让整轮问答中断）。
        selected = [all_stats[requested_id]] if requested_id in all_stats else []
    else:
        selected = list(all_stats.values())
        metric_name = "appearances" if sort_by == "appearances" else "win_rate"
        selected.sort(
            key=lambda item: (
                # 取负号实现降序；None 视作 -1，保证无样本项排在最后而不是报错。
                -(item[metric_name]["value"] if item[metric_name]["value"] is not None else -1),
                int(item["champion_id"]),
            )
        )
        selected = selected[:effective_limit]
    _add_champion_labels(selected)
    return {
        "metric": "champion_stats",
        "sort_by": sort_by,
        "limit": effective_limit,
        # 原始查询串与解析结果一起回传：模型/用户可以自行判断"是不是名字没认出来"。
        "champion_query": None if champion_id is None else str(champion_id),
        "champion_id_filter": requested_id,
        "mapping_source": "Riot Data Dragon 7.17.1 (zh_CN)",
        # 总英雄数（不受 limit 影响），避免把"只返回了前 20 条"误读成"只有 20 个英雄"。
        "total_champions": len(all_stats),
        "results": selected,
    }


def get_ban_stats(
    champion_id: str | int | None = None,
    sort_by: str = "banned_matches",
    limit: int | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return ban match counts/rates, optionally filtered/sorted.

    When no champion is named, results are ranked and capped at
    ``DEFAULT_RANKING_LIMIT`` unless an explicit ``limit`` is given.
    """
    # ---- 中文说明 ----
    # Ban（禁用）统计，结构与 get_champion_stats 对齐：
    #   指定 champion_id -> 只返回该英雄；未指定 -> 按 sort_by 排序取前 effective_limit。
    #
    # 口径：banned_matches 的分母是去重后的**全部对局数**，含义是"至少一方禁用了该
    # 英雄的对局比例"（每场最多计 1 次），这是面向用户的默认 Ban 率。空禁用位写作
    # -1，由 metrics 侧过滤，不会出现幻影英雄。
    #
    # 排序键里多了一个 `isinstance(..., dict)` 判断：ban_stats 的 sort_by 取值为
    # banned_matches（顶层是整数）与 ban_rate（顶层是结果 dict），两种形状都要能取到
    # 可比较的值，所以这里兼容地只在它是 dict 时取 ["value"]。
    _validate_limit(limit)
    if sort_by not in {"banned_matches", "ban_rate"}:
        raise ValueError("sort_by must be 'banned_matches' or 'ban_rate'")
    effective_limit = limit if limit is not None else DEFAULT_RANKING_LIMIT
    all_stats = metrics.ban_stats(_rows(csv_path))
    requested_id = _resolve_champion_id(champion_id) if champion_id is not None else None
    if champion_id is not None:
        selected = [all_stats[requested_id]] if requested_id in all_stats else []
    else:
        metric_name = "banned_matches" if sort_by == "banned_matches" else "ban_rate"
        selected = list(all_stats.values())
        selected.sort(
            key=lambda item: (
                -(item[metric_name]["value"] if isinstance(item[metric_name], dict) else item[metric_name]),
                int(item["champion_id"]),
            )
        )
        selected = selected[:effective_limit]
    _add_champion_labels(selected)
    return {
        "metric": "ban_stats",
        "sort_by": sort_by,
        "limit": effective_limit,
        "total_champions": len(all_stats),
        "champion_query": None if champion_id is None else str(champion_id),
        "champion_id_filter": requested_id,
        "mapping_source": "Riot Data Dragon 7.17.1 (zh_CN)",
        "results": selected,
    }


def get_first_event_win_rate(event: str, csv_path: str | Path | None = None) -> dict[str, Any]:
    """Return conditional win rate for the team recorded as achieving an event first."""
    # ---- 中文说明 ----
    # 条件胜率：取得该首个事件的队伍的胜率。
    #
    # 参数:
    #     event: FIRST_EVENTS 中的列名，例如 "firstBlood"；其它取值抛 ValueError。
    #
    # 返回:
    #     metrics.first_event_win_rate 的结果；分母是**该事件确有归属**的对局数，
    #     因此小于全局对局数（`first* = 0` 的比赛被排除）。
    #
    # 说明:
    #     这只是统计关联，不是因果；且它的分母与"发生率"的分母不同，回答里必须分别
    #     列出（见 kbase/project_metrics_ban_event.md）。
    return metrics.first_event_win_rate(_rows(csv_path), event)


def get_first_event_occurrence_rate(event: str, csv_path: str | Path | None = None) -> dict[str, Any]:
    """Return event occurrence rate, with the valid-match denominator included."""
    # ---- 中文说明 ----
    # 发生率：该首个事件在数据中出现的比例。
    #
    # 参数:
    #     event: FIRST_EVENTS 中的列名。
    #
    # 返回:
    #     metrics.first_event_occurrence_rate 的结果；**分母是全部有效对局**，
    #     包含没有发生该事件的比赛（这正是它与条件胜率分母不同的原因）。
    return metrics.first_event_occurrence_rate(_rows(csv_path), event)


def get_average_game_duration(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Return mean match duration in seconds."""
    # ---- 中文说明 ----
    # 平均对局时长（秒）。分母是"时长可解析且为正"的对局数；unit 为 seconds，
    # 因此 value 是精确浮点，不会被取整（回答时可同时换算成分钟）。
    return metrics.average_game_duration(_rows(csv_path))


def get_spell_stats(
    spell: str | int | None = None,
    sort_by: str = "picked_teams",
    limit: int | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return summoner-spell pick and win statistics, optionally filtered/sorted.

    ``pick_rate`` is per **team appearance** (two per match): a spell is fielded by
    a team, not by a match. When no spell is named, results are ranked and capped at
    ``DEFAULT_RANKING_LIMIT`` unless an explicit ``limit`` is given.
    """
    # ---- 中文说明 ----
    # 召唤师技能统计。pick_rate 的分母是"队伍出场次数 = 去重后对局数 × 2"，因为技能
    # 由一支队伍携带而不是由一场比赛携带；win_rate_when_picked 的分母是携带该技能的
    # 队伍数（条件胜率，不是全局胜率）。
    #
    # 未识别的技能名走"说明性结果"而不是抛异常：参数本身没有格式错误，只是这个名字
    # 不在映射表里，返回 note 让模型换一个精确名字（数字 ID 或精确中文名，如 闪现）
    # 再试，比整轮失败更有用。注意这与"已识别但数据集中没有"是两种不同结果——后者
    # requested_id 非 None，会走下面的常规分支返回 results: []（不带 note）。
    _validate_limit(limit)
    if sort_by not in {"picked_teams", "pick_rate", "win_rate_when_picked"}:
        raise ValueError("sort_by must be 'picked_teams', 'pick_rate' or 'win_rate_when_picked'")
    effective_limit = limit if limit is not None else DEFAULT_RANKING_LIMIT

    all_stats = metrics.spell_stats(_rows(csv_path))
    requested_id = _resolve_spell_id(spell) if spell is not None else None

    if spell is not None and requested_id is None:
        return {
            "metric": "spell_stats",
            "spell_query": str(spell),
            "spell_id_filter": None,
            "mapping_source": "Riot Data Dragon (zh_CN) summoner spell snapshot",
            "total_spells": len(all_stats),
            "results": [],
            "note": "未识别的召唤师技能；请传入数字 ID 或精确中文名（如 闪现）",
        }

    if requested_id is not None:
        selected = [all_stats[requested_id]] if requested_id in all_stats else []
    else:
        selected = list(all_stats.values())
        # 这里的条件表达式两个分支结果相同（sort_by 的合法取值已在上面校验过），
        # 属于冗余写法；保留原样以免改变行为。注意：不做简化。
        metric_name = sort_by if sort_by != "picked_teams" else "picked_teams"
        selected.sort(
            key=lambda item: (
                # picked_teams 是整数，pick_rate / win_rate_when_picked 是结果 dict，
                # 因此兼容两种形状；末尾 `or 0` 让 None（无样本）当作 0 参与排序。
                -(item[metric_name]["value"] if isinstance(item[metric_name], dict) else item[metric_name]) or 0,
                int(item["spell_id"]),
            )
        )
        selected = selected[:effective_limit]

    _add_spell_labels(selected)
    return {
        "metric": "spell_stats",
        "sort_by": sort_by,
        "limit": effective_limit,
        "spell_query": None if spell is None else str(spell),
        "spell_id_filter": requested_id,
        "mapping_source": "Riot Data Dragon (zh_CN) summoner spell snapshot",
        "total_spells": len(all_stats),
        "results": selected,
    }


def get_spell_win_split(spell: str | int, csv_path: str | Path | None = None) -> dict[str, Any]:
    """Compare team win rate with versus without one summoner spell.

    Both sides cover the same matches, so the difference is meaningful as a
    description of this dataset. It is **not** a causal effect: which champions and
    roles take a spell also drives the outcome.
    """
    # ---- 中文说明 ----
    # "携带某技能"与"未携带该技能"的队伍胜率对比。两侧来自同一批比赛、每支队伍只落进
    # 一侧，所以 teams 之和恒等于 去重后对局数 × 2，可以据此校验。
    #
    # 返回值里另加 caveat 字段，随数字一起进入模型上下文：这是观测数据而非受控实验，
    # 差异同时受英雄与位置选择影响，不可解读为因果、也不得据此建议"该带什么技能"。
    requested_id = _resolve_spell_id(spell)
    if requested_id is None:
        return {
            "metric": "spell_win_split",
            "spell_query": str(spell),
            "spell_id_filter": None,
            "note": "未识别的召唤师技能；请传入数字 ID 或精确中文名（如 闪现）",
        }

    result = metrics.spell_win_split(_rows(csv_path), requested_id)
    by_id, _ = _spell_maps()
    # 与 spell_stats 一样，名称由映射补上；结果里也带上映射来源，
    # 便于回答时说明"技能名来自静态快照，不代表逐场的补丁版本"。
    result["spell_name"] = by_id.get(requested_id)
    result["mapping_source"] = "Riot Data Dragon (zh_CN) summoner spell snapshot"
    result["caveat"] = "观测数据，非受控实验；胜率差异同时受英雄与位置选择影响，不可解读为因果。"
    return result


def get_team_objective_averages(
    objective: str | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return mean end-of-match objective counts per team.

    ``objective`` may be a raw column suffix (``towerKills``) or the Chinese label
    (``防御塔``); omitted, every objective is returned.
    """
    # ---- 中文说明 ----
    # 队伍目标均值（拿塔/水晶/大龙/小龙/峡谷先锋）。objective 有三种可接受的写法，
    # 按下面的顺序逐个尝试：精确的原始列名（towerKills）、OBJECTIVE_LABELS 里精确的
    # 中文名（防御塔）、以及中文名的子串（只说"男爵"也能命中"男爵（大龙）"）。
    # 都匹配不上时返回带 available 列表的说明性结果，让模型知道可选项后重试。
    #
    # 分母口径：每种目标的分母是**该列有可解析数值的比赛数**，不是全部比赛数，
    # 这样缺失值不会被当成 0 而悄悄拉低均值；unit=matches 会让 value 取整，
    # 精确均值要用 numerator / denominator 现算。
    result = metrics.team_objective_averages(_rows(csv_path))

    if objective is None:
        return result

    token = str(objective).strip()
    # 反向表：中文名 -> 列名后缀。
    by_label = {label: key for key, label in metrics.OBJECTIVE_LABELS.items()}
    resolved = token if token in metrics.TEAM_OBJECTIVES else None
    if resolved is None:
        resolved = by_label.get(token)
    if resolved is None:
        # Match the label loosely so "防御塔" style answers work: labels carry
        # parenthetical notes such as "男爵（大龙）".
        # 宽松匹配：标签里有括号注释（如"男爵（大龙）"），用户/模型可能只说"男爵"，
        # 所以退化为子串匹配。`if token` 阻止空串匹配到第一个标签。
        for label, key in by_label.items():
            if token and token in label:
                resolved = key
                break

    if resolved is None:
        return {
            "metric": "team_objective_averages",
            "objective_query": token,
            "objective_filter": None,
            # 即使没认出目标，也回传对局总数与可选项，模型可以据此换个说法重试，
            # 而不是拿到一个无法解释的空结果。
            "unique_match_count": result["unique_match_count"],
            "objectives": [],
            "available": [
                {"objective": key, "objective_label": metrics.OBJECTIVE_LABELS[key]}
                for key in metrics.TEAM_OBJECTIVES
            ],
            "note": "未识别的目标类型；请传入列名（如 towerKills）或中文名（如 防御塔）",
        }

    result["objective_query"] = token
    result["objective_filter"] = resolved
    # 过滤到只剩目标的那一项；其余字段（unique_match_count / note）保持完整。
    result["objectives"] = [
        item for item in result["objectives"] if item["objective"] == resolved
    ]
    return result
