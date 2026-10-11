---
doc_id: kbase_scheme
title: 知识库结构与分类约定
category: meta
source_type: self
version: 1.0
source_url: internal
license: project-internal
retrieved_at: 2026-10-09
---

# 知识库结构与分类约定

本知识库把项目文档与英雄联盟官方静态数据统一为带元数据的 Markdown，供检索层按标题分块。

## 分类 category

- `project`：项目自有文档（指标字典、英雄映射说明、数据审计、验收记录）。
- `domain`：领域知识，来自 Riot Data Dragon 官方静态数据。
- `conflict`：刻意保留的版本冲突样本，用于检验引用是否准确。
- `meta`：关于知识库自身的说明（本文件、数据来源与许可）。
- `offtopic`：与项目无关的干扰文档，仅用于阈值校准。
- `noise`：其他非知识内容；与 `offtopic` 一样不参与正常检索。

## 干扰文档不参与正常检索

`offtopic` 与 `noise` **不是知识**，它们的存在意义是**被拒绝**，因此
`search_knowledge` 在候选阶段就把这两类排除（`NON_CONTENT_CATEGORIES`）。

**为什么用分类排除而不是分数阈值**：实测三类的 top-1 rerank 分数几乎重叠——

| 类别 | 分数区间 |
|---|---|
| 相关（库内可答） | 0.8099 – 0.9790 |
| 干扰（库内可答但离题） | 0.8453 – 0.9453 |
| 库内不可答 | 0.2840 – 0.6185 |

相关与干扰严重重叠，**不存在**任何能挡住干扰又不误伤相关的分数阈值。
阈值只能区分「库里有答案」与「库里没有答案」，这一点没有改变。

标定脚本需要测量干扰项时显式传入 `include_noise=True`，
证据见 `evals/retrieval_calibration.jsonl`，用 `scripts/calibrate_threshold.py` 复现。
**不要**为了省事把干扰文档从 `kbase/` 删除——它们被排除在检索之外，
但仍然承担着防止阈值被误设的职责。

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

本知识库包含 7.11.1 至 7.18.1 共 8 个补丁的静态快照。

这些快照用于**名称映射、机制说明与版本对照**，不构成任何单场比赛的补丁证据。
原始 CSV 不含逐场 patch 字段，因此任何"按补丁切片"的结论都必须显式标注不确定性。

## 与原始数据的关系

- 原始对局数据：`ShuJuYuan/games.csv`（51,053 场去重对局，不在本知识库内，不做向量化）。
- 维度数据快照：`ShuJuYuan/ddragon/<补丁>/zh_CN/`，抓取记录见 `ShuJuYuan/ddragon/_manifest.json`（含 sha256）。
- 派生映射：`docs/mapping_summoner_spells.json`、`docs/mapping_champions_by_version.json`。
