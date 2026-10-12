"""Deterministic metrics for the LoL ranked games CSV.

All rates follow the project metric definitions in ``kbase/`` — see
``kbase/project_metrics_core.md`` and ``kbase/project_metrics_ban_event.md``.
(The original ``docs/metric_dictionary.md`` moved into the knowledge base.)
Functions return explicit numerators and denominators so an Agent can explain
and verify each result.

Champion ids are validated as positive integers: the dataset writes ``-1`` into
unused ban slots, and counting that sentinel would invent a champion.
"""

# ============================================================================
# 中文说明
#
# 本模块是整个项目"数字从哪里来"的唯一出口：所有指标都是这里的纯函数——输入是
# 已经读进内存的 CSV 行，输出是普通 dict，不访问网络、不调用模型、没有随机性。
# 同一份 CSV 必然得到同一组数字。
#
# 为什么统计由 Python 计算，而不让模型估算（本项目最核心的原则）：
#   1. 模型给出的是"看起来合理"的数字，不是可复现的数字：同一个问题问两次可能
#      得到两个答案，评测与用户都无法信任。统计必须在代码里固定下来。
#   2. 项目要求每个数字都能被追问"分子是多少、分母是多少"。因此 `_result(...)`
#      同时返回 numerator / denominator / value / unit，而不是只丢一个百分比出来。
#   3. 口径（谁进分子、谁进分母）是业务约定，写在 kbase/project_metrics_*.md
#      中；这类约定必须由代码固化，不能每次由模型即兴决定。
#   模型在本项目里的职责只有两件事：选择调用哪个工具、把工具返回的数字组织成中文
#   回答。它不参与任何数值计算（见 kbase/project_overview.md）。
#
# 三个贯穿全模块的口径约定（读下面的函数前请先记住）：
#
#   * 统计单位是**去重后的对局**（按 gameId 去重），不是原始 CSV 行数。原始
#     51,490 行里有 437 行重复，只对应 51,053 场不同比赛；所以每个统计函数入口
#     都先调用 `_unique_rows`（见 kbase/project_dedup_rule.md）。
#
#   * 分母为 0 时 value 返回 None，表示"无样本"，**不是 0%**。0% 的含义是"测过，
#     一次都没发生"，None 的含义是"根本没有可用样本"——两者是完全不同的结论，
#     会在回答里给出不同的措辞（见 `_result`）。
#
#   * 分母口径逐指标定义，最常见的不是"全部比赛数"，而是"**该列有可解析数值**的
#     比赛数"：把缺失值当作 0 会悄悄拉低均值，直接把缺失值剔除又必须让调用方看得
#     见剔掉了多少，所以每一处都单独计数并把分母一起回传。
#
# 与其它模块的分工：`tools.py` 负责读取 CSV、把中文名/ID 参数解析成列里的取值，
# 再把行交给本模块；本模块不认识工具的入参格式，也不做任何名称翻译。
# ============================================================================

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

Row = Mapping[str, str]

# 六个"首个事件"字段名，必须与 CSV 列名逐字一致。这个元组的顺序没有任何业务
# 含义，它只用来校验调用方传进来的 event 是否合法（见 first_event_win_rate /
# first_event_occurrence_rate 开头的 `if event not in FIRST_EVENTS`）。
FIRST_EVENTS = (
    "firstBlood",
    "firstTower",
    "firstInhibitor",
    "firstBaron",
    "firstDragon",
    "firstRiftHerald",
)


def load_matches(csv_path: str | Path) -> list[dict[str, str]]:
    """Load CSV rows and deduplicate identical gameIds; reject conflicting IDs.

    读取整个 CSV，按 gameId 去重后返回行列表；同一 gameId 的行内容不一致时抛错。

    参数:
        csv_path: CSV 路径。

    返回:
        去重后的行列表（每行是列名 -> 字符串值的 dict，所有值已 strip）。
        文件缺少 gameId 列值、或同一 gameId 出现不同内容时抛 ValueError。

    说明:
        这是"从磁盘读数据"的唯一实现，`tools.py` 与 `tools._load_dataset_overview`
        都复用它，避免出现两套去重逻辑各自漂移。
    """
    matches: dict[str, dict[str, str]] = {}
    # utf-8-sig 用来吃掉 Windows 工具链可能写入的 BOM；newline="" 交给 csv 模块
    # 自己处理行尾，避免把 CRLF 当成字段内容的一部分。
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        # start=2 让行号与"人在编辑器里看到的行号"一致（第 1 行是表头）。
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            game_id = (row.get("gameId") or "").strip()
            if not game_id:
                raise ValueError(f"Missing gameId at CSV line {line_number}")
            # 统一 strip 所有值：否则 "1" 与 "1 " 会被判定为"内容冲突"，
            # 后续的 isdigit()/数值解析也会因空白而失败。
            normalized = {key: (value or "").strip() for key, value in row.items()}
            previous = matches.get(game_id)
            # 重复行内容不一致时宁可报错也不静默取一条：那意味着数据源本身有问题，
            # 静默择一会让统计建立在不确定的数据上（项目去重规则第 3 条）。
            if previous is not None and previous != normalized:
                raise ValueError(f"Conflicting records for gameId={game_id}")
            matches[game_id] = normalized
    return list(matches.values())


def _unique_rows(rows: Iterable[Row]) -> list[Row]:
    """Deduplicate rows by gameId and guard against conflicting duplicate rows.

    把传入的行按 gameId 去重；同一 gameId 的内容冲突时抛 ValueError。

    为什么它是所有统计的前提:
        CSV 里有 437 行重复行（分布在 424 个 gameId 组），它们对应的是同一场比赛。
        如果不去重，同一场比赛会被计两次：分子分母同时变大，比率会偏移，均值会被
        重复样本加权。因此**每个**统计函数的第一个动作都是调用本函数，让下游拿到的
        一定是"一场比赛一行"，而 "对局数" 这个分母也才有确定含义。

        重复行经过逐字段比对确认内容完全一致；一旦不一致就报错而不是任选一条，
        因为那说明数据源出了问题，任何"随机取一条"的口径都无法解释。
    """
    unique: dict[str, Row] = {}
    for row in rows:
        game_id = str(row.get("gameId", "")).strip()
        if not game_id:
            raise ValueError("Every match row must have a non-empty gameId")
        previous = unique.get(game_id)
        if previous is not None and dict(previous) != dict(row):
            raise ValueError(f"Conflicting records for gameId={game_id}")
        unique[game_id] = row
    return list(unique.values())


def _result(metric: str, numerator: int | float, denominator: int, unit: str) -> dict:
    """把"分子 / 分母"打包成本项目统一的指标结果结构。

    参数:
        metric: 指标名，用于让回答能指认口径（例如 champion_win_rate）。
        numerator: 分子，含义随指标而定，由调用方保证口径正确。
        denominator: 分母。为 0 时 value 为 None。
        unit: "percent"（value 乘以 100）、"matches"（value 取整）或 "seconds"。

    返回:
        含 metric / numerator / denominator / value / unit 五个键的 dict。

    为什么分母为 0 时返回 None 而不是 0:
        0% 是一个统计结论（"观察过，且一次都没发生"），None 是"无样本"。若把无样本
        当成 0%，回答就会出现"该英雄胜率 0%"这种凭空捏造的事实，而真实情况可能是
        这个英雄在数据里根本没出场过。

    为什么 unit="matches" 时要取整:
        这是本项目的展示约定：unit 为 "matches" 的均值（例如队伍目标均值
        team1_towerKills_average）value 显示为整数（5 而不是 5.6983）。
        **精确值保存在 numerator/denominator 里**，需要小数时应当用
        `numerator / denominator` 现算，不能直接把 value 当成精确值来引用
        （见 kbase/project_metrics_core.md"均值取整"一节）。

        注意：kbase/project_metrics_core.md 在同一节里把 average_game_duration
        也列为"同样取整"的例子，但代码中平均时长用的是 unit="seconds"，
        因此它的 value 是精确浮点（约 1832.08），并没有被取整。此处仅如实指出
        文档与实现的差异，未改动任何行为。
    """
    value = None if denominator == 0 else numerator / denominator
    if value is not None and unit == "matches":
        value = int(value)
    if value is not None and unit == "percent":
        value *= 100
    return {
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "unit": unit,
    }


def match_count(rows: Iterable[Row]) -> dict:
    """去重后的对局数。

    返回:
        unit="matches"、denominator 恒为 1 的结果结构（因此 value == numerator）。

    说明:
        denominator 写死 1 是为了让"对局数"与其它指标共用同一套结果结构，下游解析
        逻辑不需要为它开特例。注意它回答的是**去重后**的对局数：原始行数是 51,490，
        对局数是 51,053，两者不是同一个数，数据集概览工具会把两个数一起给出。
    """
    matches = _unique_rows(rows)
    return _result("unique_match_count", len(matches), 1, "matches")


def _is_champion_id(value: str) -> bool:
    """True for a real champion id.

    The dataset writes ``-1`` into unused ban slots; that is a sentinel meaning
    "no champion", not a champion. Counting it would create a phantom champion
    that ranks in ban tables, so only positive integers qualify.

    判断一个单元格是否是"真实英雄 ID"。

    为什么不能只判断"是不是数字":
        空禁用位在数据里写成 `-1`，它是"这个位置没用"的哨兵值而不是英雄。
        如果把 -1 当英雄计数，Ban 榜里会凭空多出一个 ID 为 -1 的"幻影英雄"，
        而且因为每个空位都会命中它，它很可能排到榜首。同理，空字符串与
        `None` 也都不是英雄。因此只有正整数才算。测试
        test_ban_sentinel_minus_one_is_not_a_champion 与
        test_champion_slots_holding_zero_or_empty_are_ignored 锁定该行为。
    """
    text = value.strip()
    return text.isdigit() and int(text) > 0


def _champions_for_team(row: Row, team: int) -> set[str]:
    """取某一场比赛中一支队伍选到的英雄 ID 集合。

    参数:
        row: 一行比赛数据。
        team: 1 或 2，对应列前缀 `t1_` / `t2_`。

    返回:
        英雄 ID 字符串集合；非法/空槽位已在本函数内被 `_is_champion_id` 过滤掉。

    为什么用 set 而不是 list:
        "该队是否选了某英雄"只关心在不在，集合天然把同一队的重复槽位去重；
        后面的 champion_stats 也只需要"出现过"这个事实。
    """
    prefix = f"t{team}_champ"
    return {
        str(row.get(f"{prefix}{slot}id", "")).strip()
        for slot in range(1, 6)
        if _is_champion_id(str(row.get(f"{prefix}{slot}id", "")))
    }


def champion_stats(rows: Iterable[Row]) -> dict[str, dict]:
    """Return pick and win counts/rates for each champion ID, per unique match.

    返回每个英雄 ID 的出场场数、胜场数与两个比率，key 为英雄 ID 字符串。

    分子与分母口径（本项目最容易被问倒的地方）:
        * appearances（出场率/登场率）
          分子 = 该英雄出现在**任一**队伍 5 个 `champ*id` 槽位中的不同对局数；
          分母 = **去重后的全部对局数**（不是"该英雄出场的对局数"）。
          所以它是"对局覆盖率"：本场比赛中这个英雄有没有人玩。**不是**位置选取率、
          也不是玩家选取率——数据里没有分路和玩家字段。
        * win_rate（胜率）
          分子 = 英雄所在队伍获胜的场数；分母 = **该英雄出场的对局数**。
          胜率的分母比出场率的分母小，两者不能混用，输出时必须分别说明。

    两条边界行为:
        1. 同一英雄同场被两队选中时只计 1 场（`teams[1] | teams[2]` 是集合并集）；
           本项目数据中实际不存在这种情况，但口径上仍然只算一场。
        2. 只有当 `winner` 字段确实等于英雄所在队伍时才记胜场；`winner` 不是
           "1"/"2" 的值不会给任何一方加胜场。测试
           test_win_is_not_credited_when_the_holder_lost 锁定这一点。
    """
    matches = _unique_rows(rows)
    picked_games: dict[str, int] = defaultdict(int)
    won_games: dict[str, int] = defaultdict(int)
    for row in matches:
        teams = {1: _champions_for_team(row, 1), 2: _champions_for_team(row, 2)}
        picked = teams[1] | teams[2]
        winner = str(row.get("winner", "")).strip()
        for champion_id in picked:
            picked_games[champion_id] += 1
            # `champion_id in teams[int(winner)]` 是必要的：两队的并集里包含对手的
            # 英雄，仅凭 picked 无法知道这个英雄站在哪一边。
            if winner in ("1", "2") and champion_id in teams[int(winner)]:
                won_games[champion_id] += 1
    total = len(matches)
    return {
        champion_id: {
            "champion_id": champion_id,
            # 分母是全部对局：体现"这个英雄覆盖了多少比例的比赛"。
            "appearances": _result("champion_appearance_matches", games, total, "percent"),
            "wins": won_games[champion_id],
            # 分母是该英雄的出场场数：条件胜率，而非全局胜率。
            "win_rate": _result("champion_win_rate", won_games[champion_id], games, "percent"),
        }
        # 按数值 ID 排序，保证同一份数据每次返回的顺序都一致（输出可复现，
        # 也让测试可以直接比较结构）。
        for champion_id, games in sorted(picked_games.items(), key=lambda item: int(item[0]))
    }


def _spells_for_team(row: Row, team: int) -> set[str]:
    """The distinct summoner spells a team fielded in one match.

    Each of the 5 players has two spell slots (``_sum1`` / ``_sum2``), so a team
    contributes up to 10 slots but far fewer *distinct* spells — Flash alone
    appears in roughly 70% of slots. Deduplicating per team is what makes the
    count below a "spell was present" measure rather than a slot count; without it
    a team with Flash twice would be counted twice.

    The two slots are deliberately treated as unordered: the dataset gives no
    evidence that ``_sum1`` is the primary spell, so ranking them would invent
    information.

    中文说明:
        取一支队伍在一场比赛中**携带过的召唤师技能 ID 集合**。

        字段结构：每队 5 名选手、每名选手固定 2 个技能槽，列名形如
        `t{team}_champ{slot}_sum{position}`，全表共 2 队 × 5 人 × 2 槽 = 20 列。
        因此本函数遍历 10 个槽位，但返回的是**去重后的集合**：一名选手两个槽带同一
        技能、或两名选手带同一技能，该队都只贡献 1 个元素。这正是"队伍是否携带了
        该技能"这个口径的来源；若不去重就变成了"技能槽位占比"，那是另一个指标，
        必须另行命名（见 kbase/project_metrics_summoner_spell.md）。

        两个槽位是**无序**的：数据没有说明 `_sum1` 是主技能还是排序结果，所以不能
        解读为优先级或选取顺序。测试 test_slot_order_is_not_treated_as_meaningful
        通过交换两个槽位的值来锁定"顺序不影响结果"。

        与英雄 ID 相同的正整数规则：空值、`-1` 等非正数都表示"这个槽是空的"，
        不是技能。
    """
    prefix = f"t{team}_champ"
    spells: set[str] = set()
    for slot in range(1, 6):
        for position in (1, 2):
            value = str(row.get(f"{prefix}{slot}_sum{position}", "")).strip()
            # Same positive-integer rule as champion ids: anything else (empty,
            # -1) is a missing slot, not a spell.
            if value.isdigit() and int(value) > 0:
                spells.add(value)
    return spells


def spell_stats(rows: Iterable[Row]) -> dict[str, dict]:
    """Return pick and win counts/rates for each summoner spell.

    Denominator for ``pick_rate`` is **team appearances** (2 per unique match),
    because a spell is fielded by a team, not by a match. Denominator for
    ``win_rate_when_picked`` is the number of teams that fielded that spell.

    This is observational data, not a controlled experiment: a spell's win rate
    also reflects the champions and roles that take it. Callers must not read it
    as a causal effect.

    中文说明:
        返回每个召唤师技能 ID 的携带队伍数、出现场数与两个比率。

        分子/分母口径:
            * pick_rate（出场率）
              分子 = 携带该技能的**队伍出场次数**；
              分母 = 去重后对局数 × 2，即"全部队伍出场次数"。
              分母为什么是 2 倍对局数而不是对局数：技能由**一支队伍**携带，同一场
              比赛里两队可能都带同一技能，此时分子会加 2。若分母只按对局数算，
              比例就可能超过 100%。用"队伍出场次数"当分母，这个指标才恰好是
              "队伍是否携带了该技能"。
            * win_rate_when_picked（被选时胜率）
              分子 = 携带该技能且获胜的队伍数；分母 = 携带该技能的队伍数。
              它是条件胜率，**不等于**全局胜率（全局恒为 50%）。

        `picked_teams` 与 `matches_with_spell` 是两个不同的数：前者是队伍口径，
        后者是"至少有一支队伍携带该技能的对局数"。一场比赛两队都带同一技能时，
        picked_teams 加 2 而 matches_with_spell 只加 1。

        解读边界：这是观测数据而非受控实验。技能胜率差异同时受英雄/定位选择、
        对局内的反向应对、样本量大小影响；闪现这类近乎人人携带的技能没有对照组，
        其胜率必然接近 50%，不能据此得出"无用"的结论。这些限制必须随数字一起给出。
    """
    matches = _unique_rows(rows)
    picked_teams: dict[str, int] = defaultdict(int)
    won_teams: dict[str, int] = defaultdict(int)
    appeared_in_match: dict[str, int] = defaultdict(int)
    for row in matches:
        winner = str(row.get("winner", "")).strip()
        for team in (1, 2):
            spells = _spells_for_team(row, team)
            for spell in spells:
                appeared_in_match[spell] += 1
                picked_teams[spell] += 1
                if winner == str(team):
                    won_teams[spell] += 1

    team_appearances = len(matches) * 2
    return {
        spell: {
            "spell_id": spell,
            "picked_teams": picked_teams[spell],
            "matches_with_spell": appeared_in_match[spell],
            "wins": won_teams[spell],
            "pick_rate": _result("spell_pick_rate", picked_teams[spell], team_appearances, "percent"),
            "win_rate_when_picked": _result(
                "spell_win_rate_when_picked", won_teams[spell], picked_teams[spell], "percent"
            ),
        }
        for spell in sorted(picked_teams, key=lambda value: int(value))
    }


def spell_win_split(rows: Iterable[Row], spell_id: str) -> dict:
    """Compare team win rates with and without a specific summoner spell.

    Returns both sides so a difference can be stated without implying causation:
    ``with`` counts teams that fielded the spell, ``without`` counts teams that did
    not, and both are measured over the same matches.

    中文说明:
        比较"携带某技能"与"未携带该技能"的**队伍胜率**。

        口径:
            * with_spell：分子 = 携带该技能且获胜的队伍数，分母 = 携带该技能的队伍数；
            * without_spell：分子 = 未携带且获胜的队伍数，分母 = 未携带的队伍数。

        为什么两侧一定可比：它们来自**同一批比赛**，且每支队伍只会落进其中一侧，
        所以 `with.teams + without.teams == 去重后对局数 × 2`。这个恒等式是可以用
        来校验结果的（测试 test_spell_win_split_covers_every_team_exactly_once）。

        为什么仍然不能当因果：谁携带什么技能由英雄与位置决定（需要位移的英雄更倾向
        带闪现、打野几乎必带惩戒），因此差异里混着阵容因素。调用方只能作为描述性
        统计呈现（结果里另附 caveat 字段）。
    """
    wanted = str(spell_id).strip()
    # 这里也走"正整数即合法 ID"的规则；非法输入抛错而不是返回空结果，
    # 让调用方立刻知道参数本身有问题。
    if not (wanted.isdigit() and int(wanted) > 0):
        raise ValueError("spell_id must be a positive integer")

    matches = _unique_rows(rows)
    with_teams = with_wins = 0
    without_teams = without_wins = 0
    for row in matches:
        winner = str(row.get("winner", "")).strip()
        for team in (1, 2):
            # 胜负判断写成布尔量，再用 int() 累加，避免 if/else 分支重复计数逻辑。
            won = winner == str(team)
            if wanted in _spells_for_team(row, team):
                with_teams += 1
                with_wins += int(won)
            else:
                without_teams += 1
                without_wins += int(won)

    return {
        "spell_id": wanted,
        "with_spell": {
            "teams": with_teams,
            "wins": with_wins,
            "win_rate": _result("team_win_rate_with_spell", with_wins, with_teams, "percent"),
        },
        "without_spell": {
            "teams": without_teams,
            "wins": without_wins,
            "win_rate": _result("team_win_rate_without_spell", without_wins, without_teams, "percent"),
        },
    }


def ban_stats(rows: Iterable[Row]) -> dict[str, dict]:
    """Return champion ban incidence: matches with >=1 ban / all matches.

    Uses the same positive-id rule as picks, so the ``-1`` sentinel that fills
    unused ban slots never appears as a champion.

    中文说明:
        返回每个英雄的禁用场数与禁用率。

        口径:
            * banned_matches（禁用场数）
              分子 = 该英雄出现在任一队伍 5 个 `ban*` 槽位中的**不同对局数**；
              分母 = 去重后的全部对局数。
              同一英雄在同一场被多次禁用只计 1 场。集合去重在这里不是可选项：
              实测本数据中有 31,893 / 51,053 场（约 62.5%）出现**同一英雄同时出现在
              两队的禁用位**，同一队内部则没有重复。若按槽位计数，这些英雄的禁用场数
              会被多算一次。测试 test_same_champion_banned_twice_in_one_match_counts_once
              锁定该行为。
              注意：跨队重复禁用为何如此常见，无法从数据本身判断（既可能是当时
              的禁用规则，也可能是数据来源的写法），此处仅陈述实测结果。
            * ban_rate（Ban 率）
              同样的"至少一方禁用该英雄的对局比例"。

        为什么这是默认的 Ban 率口径：它是面向用户的概念（"多少比例的比赛里这个
        英雄被禁了"），每场最多计 1 次。如果改用"禁用槽位数 ÷ 可用槽位总数"，
        那是另一种指标，必须另起名字，不能沿用 Ban 率（见
        kbase/project_metrics_ban_event.md）。

        空禁用位写作 `-1`，由 `_is_champion_id` 过滤，因此不会出现 ID 为 -1 的
        幻影英雄（测试 test_ban_sentinel_minus_one_is_not_a_champion）。
    """
    matches = _unique_rows(rows)
    banned_games: dict[str, int] = defaultdict(int)
    for row in matches:
        banned = {
            str(row.get(f"t{team}_ban{slot}", "")).strip()
            for team in (1, 2)
            for slot in range(1, 6)
            if _is_champion_id(str(row.get(f"t{team}_ban{slot}", "")))
        }
        for champion_id in banned:
            banned_games[champion_id] += 1
    return {
        champion_id: {
            "champion_id": champion_id,
            "banned_matches": banned_games[champion_id],
            "ban_rate": _result("champion_ban_match_rate", banned_games[champion_id], len(matches), "percent"),
        }
        for champion_id in sorted(banned_games, key=lambda value: int(value))
    }


def first_event_win_rate(rows: Iterable[Row], event: str) -> dict:
    """Win rate of the team recorded as achieving an event first; excludes event=0.

    取得某个"首个事件"的队伍的条件胜率。

    参数:
        rows: 比赛行。
        event: FIRST_EVENTS 中的一个列名；其它取值抛 ValueError。

    口径:
        分子 = 首个事件由某队取得**且该队最终获胜**的对局数；
        分母 = 首个事件确有归属（字段值为 "1"/"2"）**且 winner 有效**的对局数。

        分母为什么小于全局总样本：`first* = 0` 表示这场比赛根本没有发生该事件
        （例如没有出现男爵），这类比赛没有"取得事件的一方"，无法参与条件胜率，
        必须排除。所以本指标的分母与 first_event_occurrence_rate 的分母**不是同一个
        数**；同一个问题里同时问发生率与条件胜率时，必须分别列出各自的分子分母
        （见 kbase/project_metrics_ban_event.md"两个分母不要混淆"）。

        与获胜之间只报告统计关联，不声称因果。
    """
    if event not in FIRST_EVENTS:
        raise ValueError(f"Unsupported first-event field: {event}")
    numerator = denominator = 0
    for row in _unique_rows(rows):
        event_team = str(row.get(event, "")).strip()
        winner = str(row.get("winner", "")).strip()
        # 只有事件归属与胜负都明确的比赛才进入分母，避免把"数据缺失"当成
        # "没有取得事件"而污染条件胜率。
        if event_team in ("1", "2") and winner in ("1", "2"):
            denominator += 1
            # 布尔值参与整数加法：True 记 1，False 记 0。
            numerator += event_team == winner
    return _result(f"{event}_team_win_rate", numerator, denominator, "percent")


def first_event_occurrence_rate(rows: Iterable[Row], event: str) -> dict:
    """Share of valid matches where the requested first event occurred.

    某个"首个事件"的发生率。

    口径:
        * valid（有效对局）：该 `first*` 字段值为 "0"/"1"/"2" 的对局。值缺失或写成
          别的内容时不进入分母——那属于数据问题，不能算作"没发生"。
        * 分子 = 字段值为 "1"/"2"（确有队伍取得）的对局数。
        * 分母 = **全部有效对局，包含未发生该事件的比赛**。

        分母为什么包含"没发生"的比赛:
            发生率问的是"这件事在数据里有多常见"。若只拿发生过的比赛当分母，比例会
            恒等于 100%，信息量为零。也正因如此，本指标的分母（全部有效对局，本数据
            为 51,053）与 first_event_win_rate 的条件分母（本数据 firstBlood 为
            50,499）是两个不同的数，不能互相替换。

        边界：某事件从未发生时，分母非 0 而分子为 0，value 就是 0.0（"测过且没发生"），
        这与分母为 0 时的 None（无样本）含义不同。
    """
    if event not in FIRST_EVENTS:
        raise ValueError(f"Unsupported first-event field: {event}")
    matches = _unique_rows(rows)
    valid = [row for row in matches if str(row.get(event, "")).strip() in ("0", "1", "2")]
    occurred = sum(str(row.get(event, "")).strip() in ("1", "2") for row in valid)
    return _result(f"{event}_occurrence_rate", occurred, len(valid), "percent")


def average_game_duration(rows: Iterable[Row]) -> dict:
    """Return mean duration in seconds over positive, parseable gameDuration values.

    平均对局时长（秒）。

    口径:
        分子 = 有效对局的 `gameDuration` 秒数之和；
        分母 = **能解析成正整数秒的对局数**（无法解析的、以及非正数的时长被排除）。

        分母为什么不是全部对局数：把解析不了的值当作 0 秒会把均值往低处拉，那等于
        凭空缩短了对局；直接剔除又必须让调用方看得见剔除了多少，所以分母单独计数。
        本数据集的时长都在 190–4,728 秒之间、没有解析失败的值，但代码不依赖这一点。

        unit 是 "seconds" 而不是 "matches"，所以 value 是精确浮点（约 1832.08 秒，
        即约 30.5 分钟），不会被 `_result` 取整。输出时应同时给出秒数与分钟两种格式。
    """
    durations: list[int] = []
    for row in _unique_rows(rows):
        try:
            duration = int(row.get("gameDuration", ""))
        except (TypeError, ValueError):
            # 解析失败就跳过：既不当作 0，也不中断整次统计。
            continue
        if duration > 0:
            durations.append(duration)
    return _result("average_game_duration_seconds", sum(durations), len(durations), "seconds")


# Team-objective columns, grouped by team. These are end-of-match cumulative counts,
# so a mean answers "how many did a team take in a typical game" and says nothing
# about when they were taken or in what order.
#
# 队伍目标类列名（不含队伍前缀，真实列名是 `t{1,2}_<名字>`）。这些是**对局结束时
# 的累计值**，因此均值回答的是"典型一局里一支队伍拿了多少个"，不代表取得时间、
# 取得顺序或过程。这个顺序同时决定返回结果里 objectives 的排列顺序。
TEAM_OBJECTIVES = (
    "towerKills",
    "inhibitorKills",
    "baronKills",
    "dragonKills",
    "riftHeraldKills",
)

# Human-facing names, since the raw column names are not what a question would say.
#
# 面向用户的中文名：用户不会说 `towerKills`，只会说"防御塔"。这里的中文名同时是
# 参数解析（tools.get_team_objective_averages 支持按中文名过滤）与结果展示的来源，
# 所以名称里允许带括号说明（如"男爵（大龙）"），调用方做匹配时需要模糊匹配。
OBJECTIVE_LABELS = {
    "towerKills": "防御塔",
    "inhibitorKills": "水晶",
    "baronKills": "男爵（大龙）",
    "dragonKills": "小龙",
    "riftHeraldKills": "峡谷先锋",
}


def team_objective_averages(rows: Iterable[Row]) -> dict:
    """Mean end-of-match objective counts, per objective and per side.

    The denominator is the number of matches that recorded a parseable value for
    that column, so a column with missing entries cannot silently shrink the mean by
    treating blanks as zero. Every team-objective column in this dataset is fully
    populated, and the reported per-side denominator makes that verifiable rather
    than assumed.

    Results are reported per side (``team1`` / ``team2``) and combined, but **no
    winner/loser split is produced**: ``winner`` identifies the winning side, and
    ``team1``/``team2`` are not colour sides, so a "winning team" aggregate would be
    a different metric needing its own name and definition.

    中文说明:
        每种目标（防御塔/水晶/男爵/小龙/峡谷先锋）分别给出两侧与合并的均值。

        分母口径（本指标最关键的取舍）:
            分母是**该列有可解析数值的比赛数**，而不是全部比赛数。这样缺失值不会被
            当作 0 而悄悄拉低均值。本数据集这 10 列每一场都有数值，返回的每侧
            `matches` 字段正是为了让人**核实**这一点，而不是只能相信"应该没缺失"。
            合并值 combined 的分母是两侧有效计数之和（最多 2 × 对局数）。

        `unit="matches"` 导致 `value` 被取整（例如显示 5 而不是 5.6983）：
        **精确均值要用 `average.numerator / average.denominator` 现算**，回答涉及
        小数时不能直接引用 value。

        为什么不按胜方/败方拆分:
            只输出 team1 / team2 两侧与合并值。`winner` 标识胜方，而 team1/team2
            **不是颜色方**（数据里没有蓝红字段），所以"胜方平均推塔数"属于另一个
            指标，需要独立命名与定义后才能提供；结果里的 note 也如实说明了这一点
            （测试 test_objective_average_does_not_split_by_winner）。
    """
    matches = _unique_rows(rows)
    # (total, count) per side and objective, accumulated in a single pass.
    # 一次遍历同时累积"和"与"有效计数"：把 [total, count] 当成两元素累加器，
    # 避免为每个目标各写一遍遍历。
    accumulator: dict[str, list[int]] = {
        f"team{team}_{objective}": [0, 0]
        for team in (1, 2)
        for objective in TEAM_OBJECTIVES
    }

    for row in matches:
        for team in (1, 2):
            for objective in TEAM_OBJECTIVES:
                value = str(row.get(f"t{team}_{objective}", "")).strip()
                # lstrip("-") 让负数也能通过判断（本数据集中目标数不会是负数，
                # 这里只是宽容地接受带符号的整数写法），非数字一律跳过：
                # 跳过的行不进入分母，所以缺失值不会被当成 0。
                if not value.lstrip("-").isdigit():
                    continue
                cell = accumulator[f"team{team}_{objective}"]
                cell[0] += int(value)
                cell[1] += 1

    objectives: list[dict] = []
    for objective in TEAM_OBJECTIVES:
        # 这里先把两侧的累加器取出来，随后立刻把 sides["teamN"] 换成结构化的 dict，
        # 因此下面 pooled_* 用的是 accumulator 里的原始值，而不是已经被覆盖的 sides。
        sides = {f"team{team}": accumulator[f"team{team}_{objective}"] for team in (1, 2)}
        sides["team1"] = {
            "matches": sides["team1"][1],
            "total": sides["team1"][0],
            "average": _result(
                f"team1_{objective}_average", sides["team1"][0], sides["team1"][1], "matches"
            ),
        }
        sides["team2"] = {
            "matches": sides["team2"][1],
            "total": sides["team2"][0],
            "average": _result(
                f"team2_{objective}_average", sides["team2"][0], sides["team2"][1], "matches"
            ),
        }
        # 合并值用"两侧和 ÷ 两侧有效计数之和"，不等同于两个均值的算术平均
        # （两侧的有效样本数可能不同，先平均再平均会引入偏差）。
        pooled_total = accumulator[f"team1_{objective}"][0] + accumulator[f"team2_{objective}"][0]
        pooled_count = accumulator[f"team1_{objective}"][1] + accumulator[f"team2_{objective}"][1]
        objectives.append(
            {
                "objective": objective,
                "objective_label": OBJECTIVE_LABELS[objective],
                "column_pattern": f"t{{1,2}}_{objective}",
                "team1": sides["team1"],
                "team2": sides["team2"],
                "combined": _result(
                    f"{objective}_average_per_team", pooled_total, pooled_count, "matches"
                ),
            }
        )

    return {
        "metric": "team_objective_averages",
        "unique_match_count": len(matches),
        # note 会随结果一起进入模型上下文，用来提醒"累计值、无时间顺序、不做胜败拆分"，
        # 避免模型把均值解读成过程性结论。测试会断言这段文字包含"不按胜方"。
        "note": "对局结束时的累计值；不代表取得时间或顺序。不按胜方/败方拆分。",
        "objectives": objectives,
    }
