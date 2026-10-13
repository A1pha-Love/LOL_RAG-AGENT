# 第三方内容声明（Third-Party Notices）

本仓库包含或引用了**并非本项目作者创作**的材料。`LICENSE` 中的 MIT 授权仅适用于本项目
自身的源代码、文档与注释；下列内容仍适用其各自的条款，此处按其许可允许的方式分发。

**为什么单独放在这个文件**：GitHub 的许可证识别要求 `LICENSE` 与标准模板近乎精确匹配，
在 MIT 正文后追加说明会导致识别失败并显示为 "Other"。把这些声明放在独立文件是通行做法，
既让许可证被正确识别，也不削弱声明的效力。

---

## 1. Riot Games 静态数据（Data Dragon）

- **涉及路径**
  - `kbase/domain_*.md`（26 份，由 Data Dragon 静态数据整理的机制与目录说明）
  - `docs/champion_mapping_7_17_1.json`
  - `docs/mapping_summoner_spells.json`、`docs/mapping_champions_by_version.json`
  - `kbase/conflict_item_catalog_drift.md`
- **来源**：<https://developer.riotgames.com/docs/lol#data-dragon>
- **版权**：Riot Games, Inc.
- **分发依据**：按 Riot 的社区使用条款（"Legal Jibber Jabber"）用于非商业用途。
- **声明**：本项目**未获 Riot Games 认可**，也不代表 Riot Games 或任何官方参与制作、
  管理 Riot Games 产品的人士的观点或意见。Riot Games 及其相关标识是 Riot Games, Inc.
  的商标或注册商标。
- **说明**：原始快照文件（`ShuJuYuan/ddragon/`，约 32 MB）**未纳入版本库**；仓库内只有
  由它们派生出的映射与文档。

## 2. Kaggle 数据集附带元数据

- **涉及路径**：`ShuJuYuan/champion_info.json`、`champion_info_2.json`、
  `summoner_spell_info.json`
- **来源**：<https://www.kaggle.com/datasets/datasnaek/league-of-legends>
- **作者**：Mitchell J
- **许可**：**CC0 1.0（Public Domain）**
- **说明**：对局数据本身（`games.csv`，同为 CC0）**未纳入版本库**；获取方式见 README
  的「快速开始」一节。

---

## 3. 依赖与模型服务

以下为**运行时依赖**，本项目不重新分发其代码，仅通过包管理器与 API 调用：

- **Python 依赖**：LangGraph、LangChain、OpenAI Python SDK 等，各自适用其原许可
  （版本见 `requirements-lock.txt`，直接依赖见 `pyproject.toml`）。
- **前端依赖**：Vue 3、Vite、TypeScript 等，适用各自原许可（见 `frontend/package.json`）。
- **模型服务**：对话、向量与重排均通过阿里云百炼 DashScope（Qwen）API 调用，
  需使用者自行申请凭据并遵守其服务条款。
