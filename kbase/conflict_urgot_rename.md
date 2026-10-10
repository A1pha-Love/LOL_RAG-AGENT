---
doc_id: conflict_urgot_rename
title: 版本冲突样本：英雄 ID 6（厄加特）的更名
category: conflict
source_type: derived
version: 7.14.1~7.15.1
source_url: "https://ddragon.leagueoflegends.com/cdn/7.17.1/data/zh_CN/champion.json"
license: Riot Games 官方静态数据（Data Dragon）；本文为派生对照
retrieved_at: 2026-10-09
---

# 版本冲突样本：英雄 ID 6（厄加特）的更名
## 事实

- 中文名称在 7.14.1 为「首领之傲」，自 7.15.1 起变为「无畏战车」。
- 中文称号「厄加特」与英文标识符 `Urgot` 未变。
- 名称变化伴随该英雄的重做（技能组同时改变），因此**同一 ID 在不同补丁下的技能描述也不同**。

## 为什么这是重要的冲突样本

项目文档 `docs/champion_mapping.md` 记录了这一变化，但未标注发生的确切补丁。依据 Data Dragon 快照，可以确认变化发生在 7.14.1 之后、7.15.1 之前。

## 引用要求

1. 检索到本文时，答案必须说明所依据的补丁版本。
2. 不得据名称变化推断某场对局使用了哪个名称——CSV 无逐场 patch 字段。
3. 讨论该英雄技能时，必须明确技能组来自哪个补丁。
