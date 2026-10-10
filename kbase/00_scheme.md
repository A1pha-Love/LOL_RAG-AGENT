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
