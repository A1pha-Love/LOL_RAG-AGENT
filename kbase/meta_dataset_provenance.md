---
doc_id: meta_dataset_provenance
title: 数据来源、许可与版本边界
category: meta
source_type: self
version: 7.11.1~7.18.1
source_url: "https://www.kaggle.com/datasets/datasnaek/league-of-legends"
license: "原始数据集 CC0: Public Domain；Data Dragon 为 Riot 官方静态数据"
retrieved_at: 2026-10-09
---

# 数据来源、许可与版本边界
## 对局数据

- 来源：Kaggle 数据集 `datasnaek/league-of-legends`（(LoL) League of Legends Ranked Games）。
- 作者：Mitchell J；许可：**CC0: Public Domain**。
- 本地文件 `ShuJuYuan/games.csv` 与该数据集发布的 games.csv 字节数一致（9,287,739 字节）。
- 内容：2017-06-08 至 2017-09-06（UTC）的 EUW 排位对局，原始表 51,490 行、61 列，按 `gameId` 去重后 51,053 场。

## 随数据集发布的配套文件

| 文件 | 字节 | 用途 |
|---|---:|---|
| `champion_info.json` | 21,713 | 英雄与召唤师技能 ID 对照 |
| `champion_info_2.json` | 33,012 | 英雄与召唤师技能 ID 对照 |
| `summoner_spell_info.json` | 5,564 | 英雄与召唤师技能 ID 对照 |

这些配套文件使用**英文**名称，与 Data Dragon 的中文名称互补；其中记录了一个 id 为 -1 的占位条目，对应 CSV 中表示空禁用位的 `-1` 哨兵值。

## 维度数据

- 来源：Riot Data Dragon 官方静态数据，快照补丁 7.11.1、7.12.1、7.13.1、7.14.1、7.15.1、7.16.1、7.17.1、7.18.1。
- 抓取记录：`ShuJuYuan/ddragon/_manifest.json`，含每个文件的 URL、字节数与 sha256。
- 用途：英雄名称映射、技能与机制说明、装备与召唤师技能目录、版本对照。

## 边界声明

1. Data Dragon 快照**不构成任何单场比赛的补丁证据**；CSV 无逐场 patch 字段。
2. 本数据集不可再生：Riot 官方比赛接口仅保留约两年的对局，2017 年数据无法重新抓取。
3. 数据不含玩家标识、分路、逐分钟过程或装备选择，因此不能回答玩家级、分路级与出装问题。
