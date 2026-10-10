# 数据源初步审计报告

> **本文件的内容已迁入知识库 `kbase/`，此处保留作为原始记录。**
> 当前生效的审计结果见 `kbase/project_data_audit.md` 与
> `kbase/project_dedup_rule.md`。检索层只索引 `kbase/`。

- 文件：`ShuJuYuan/games.csv`
- 文件大小：9,287,739 bytes
- 审计时间（UTC）：2026-10-07T09:58:24+00:00
- 原始文件以只读方式打开；本脚本不写回或转换源 CSV。

## 文件结构

- 数据行数：51,490
- 字段数：61
- 列名：`gameId`, `creationTime`, `gameDuration`, `seasonId`, `winner`, `firstBlood`, `firstTower`, `firstInhibitor`, `firstBaron`, `firstDragon`, `firstRiftHerald`, `t1_champ1id`, `t1_champ1_sum1`, `t1_champ1_sum2`, `t1_champ2id`, `t1_champ2_sum1`, `t1_champ2_sum2`, `t1_champ3id`, `t1_champ3_sum1`, `t1_champ3_sum2`, `t1_champ4id`, `t1_champ4_sum1`, `t1_champ4_sum2`, `t1_champ5id`, `t1_champ5_sum1`, `t1_champ5_sum2`, `t1_towerKills`, `t1_inhibitorKills`, `t1_baronKills`, `t1_dragonKills`, `t1_riftHeraldKills`, `t1_ban1`, `t1_ban2`, `t1_ban3`, `t1_ban4`, `t1_ban5`, `t2_champ1id`, `t2_champ1_sum1`, `t2_champ1_sum2`, `t2_champ2id`, `t2_champ2_sum1`, `t2_champ2_sum2`, `t2_champ3id`, `t2_champ3_sum1`, `t2_champ3_sum2`, `t2_champ4id`, `t2_champ4_sum1`, `t2_champ4_sum2`, `t2_champ5id`, `t2_champ5_sum1`, `t2_champ5_sum2`, `t2_towerKills`, `t2_inhibitorKills`, `t2_baronKills`, `t2_dragonKills`, `t2_riftHeraldKills`, `t2_ban1`, `t2_ban2`, `t2_ban3`, `t2_ban4`, `t2_ban5`
- 字段数不一致的行：0（DictReader 发现额外字段的行；缺字段另从空值统计观察）
- 重复非空 `gameId` 出现次数（每组首条以外）：437，分布在 424 个重复 ID 组中。
- 对重复 `gameId` 的 61 个字段逐项比对：437 条重复出现均与对应首条完全一致；发现 0 组字段冲突。
- `creationTime` 范围：2017-06-08T03:14:54.922000+00:00 至 2017-09-06T14:15:32.198000+00:00（按毫秒 Unix 时间戳解析）
- `gameDuration` 范围：190 至 4728 秒

## 关键字段取值

### `winner`

| 值 | 数量 |
|---|---:|
| `1` | 26077 |
| `2` | 25413 |


### `firstBlood`

| 值 | 数量 |
|---|---:|
| `0` | 555 |
| `1` | 26113 |
| `2` | 24822 |


### `firstTower`

| 值 | 数量 |
|---|---:|
| `0` | 1213 |
| `1` | 25861 |
| `2` | 24416 |


### `firstInhibitor`

| 值 | 数量 |
|---|---:|
| `0` | 6276 |
| `1` | 23054 |
| `2` | 22160 |


### `firstBaron`

| 值 | 数量 |
|---|---:|
| `0` | 20258 |
| `1` | 14758 |
| `2` | 16474 |


### `firstDragon`

| 值 | 数量 |
|---|---:|
| `0` | 2000 |
| `1` | 24690 |
| `2` | 24800 |


### `firstRiftHerald`

| 值 | 数量 |
|---|---:|
| `0` | 26179 |
| `1` | 12948 |
| `2` | 12363 |


## 缺失字段

| 字段 | 空值行数 |
|---|---:|
|（无）|0|

## 非法数值

| 字段 | 无法解析数量 |
|---|---:|
|（无）|0|

## 初步解释与待核实项

- 与引用 Kaggle 原始数据集的公开说明交叉核对后，数据集为超过 5 万场 EUW 排位赛；`winner` 中 `1=team1`、`2=team2`，各 `first*` 事件中 `1=team1`、`2=team2`、`0=无人/未发生`。来源：[Kaggle 数据集页面](https://www.kaggle.com/datasets/datasnaek/league-of-legends)、[引用该数据集并列出字段定义的 EDA README](https://github.com/aidannguyen23/LoL-Exploratory-Data-Analysis)。
- `creationTime` 按毫秒 Unix 时间戳解析后覆盖 2017-06-08 至 2017-09-06；该范围与历史数据相符，但版本号/补丁号仍需另行映射。
- 以上字段定义来自公开数据集说明的二次引用，尚未取得原始 Kaggle 元数据/JSON 附件作独立验证。
- 这 437 条记录可在分析副本中按 `gameId` 去重，但保留原始 CSV 不变，并在数据字典中记录去重规则。
- `creationTime` 暂按毫秒 Unix 时间戳试算；正式分析前需确认单位、时区和数据覆盖赛季。
- 当前文件只有对局级字段；不能据此分析玩家级、分路级、逐分钟经济或具体装备过程。
- 胜率、Ban 率、登场率等指标的定义和分母需在后续数据字典中明确。
