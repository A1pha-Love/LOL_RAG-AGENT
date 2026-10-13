# LoL Ranked Data Agent

面向本地 League of Legends 历史排位赛数据的中文分析 Agent。项目通过 LangGraph 编排单 Agent，用 Qwen 对话模型选择工具；精确统计由 Python 工具完成，项目文档由本地 RAG 检索后作为回答依据。

**核心设计原则：数值一律由 Python 计算，不让模型估算。** 模型只负责选择工具、检索文档并
组织回答；任何统计数字都来自对局数据本身，且附分子/分母口径与文档来源。

---

## 快速开始

### 0. 前提条件

| 需要 | 说明 |
|---|---|
| **Python 3.14** | `pyproject.toml` 要求 `>=3.14,<3.15` |
| **DashScope API Key** | 对话与向量请求共用；在阿里云百炼控制台申请 |
| **对局数据集** | `ShuJuYuan/games.csv`。**该文件不在仓库中**（见下一步）|
| Node.js 20+ | *仅*在需要重新构建前端或开发界面时才需要；仓库已含构建产物 |

### 1. 获取对局数据（必做）

`games.csv` 因体积与许可原因未纳入版本库，需要自行下载：

- 数据集：Kaggle [datasnaek/league-of-legends](https://www.kaggle.com/datasets/datasnaek/league-of-legends)
  （作者 Mitchell J，许可 **CC0: Public Domain**）
- 把其中的 `games.csv` 放到 `ShuJuYuan/games.csv`
- 校验：文件应为 **9,287,739 字节**。若字节数不同，说明下载到的版本不同，统计数字会随之变化

> 仓库中**已包含**数据集附带的元数据文件（`champion_info.json`、`champion_info_2.json`、
> `summoner_spell_info.json`），无需另外下载。

### 2. 创建虚拟环境并安装依赖

在项目根目录执行（下同）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` 是锁定版本，可复现；`pyproject.toml` 声明直接依赖。
下文所有命令都假定当前目录是项目根目录，并统一使用 `.\.venv\Scripts\python.exe`。

### 3. 配置 API Key

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，只需填一项：

```ini
DASHSCOPE_API_KEY=你的Key
```

**其余变量都有可用默认值**（`.env.example` 中逐项注明用途），默认对话模型为
`qwen3.8-flash`、向量模型为 `qwen3.7-text-embedding`。`.env` 已在 `.gitignore` 中，
不会被提交。

### 4. 构建知识索引（必做一次）

RAG 检索依赖本地 SQLite 索引，仓库不含该文件（`reports/knowledge.db`）：

```powershell
.\.venv\Scripts\python.exe scripts\index_knowledge.py
```

该命令会扫描 `kbase/` 下全部 Markdown 文档（44 份 / 约 1171 个分块），调用 Embedding API
生成向量并写入 `reports/knowledge.db`。**这一步消耗向量额度**，只需在首次或 `kbase/`
变更后运行。想先看看会切出什么，可用 `--dry-run`（不调用 API）。

### 5. 启动并打开界面

**方式一（推荐）：双击 `LOL-AGENT.lnk`**

仓库根目录提供了启动入口，它会启动后端、等待就绪后自动打开浏览器，并先做三项前置检查
（虚拟环境、`.env`、前端构建产物），失败时给出可读提示。停止服务只需关闭
「LOL-AGENT backend」窗口。

**方式二：命令行**

```powershell
.\.venv\Scripts\python.exe scripts\serve.py
```

然后访问 <http://127.0.0.1:8000>。后端会同时托管已构建的前端界面，因此单端口即可
（交互式 API 文档在 `/docs`）。

**方式三：不用界面，直接问答**

```powershell
# 单轮问答（不读写会话库）
.\.venv\Scripts\python.exe scripts\chat_session.py --once "这份数据去重后有多少场比赛？"

# 多轮对话（会话持久化到 .cache/sessions.db）
.\.venv\Scripts\python.exe scripts\chat_session.py
```

### 6. 确认部署正确

```powershell
# 三个端点是否连通（对话 / 向量 / 重排）
.\.venv\Scripts\python.exe scripts\check_api_key.py

# 全部本地检查（11 步；不能运行的会标 SKIP，不会伪装成通过）
.\.venv\Scripts\python.exe scripts\run_checks.py
```

---

## 项目状态

本地 MVP 已完成。**收尾时的实际状态、验证方式与已知限制见
[项目结项记录](reports/project_closeout.md)**；`final_acceptance.md` 是 2026-10-08 的
历史验收记录，按约定保持原样，两者时间点不同、数字不同以前者为准。

项目仅用于本地运行，未发布或部署为线上服务。

## 功能

- 对 `games.csv` 提供 11 个确定性统计工具：去重对局数、数据概览、英雄出场与胜率、
  Ban 统计、首个事件发生率与条件胜率、平均时长、召唤师技能选取率与胜率、
  「有/无某技能」的胜率对比、队伍目标均值（防御塔/水晶/男爵/小龙/峡谷先锋）、
  以及项目文档检索。
- 按 `gameId` 去重；原始 CSV 保持只读。数据为 2017-06-08 至 2017-09-06 UTC 的历史 EUW 排位对局。
- 支持通过英雄数值 ID、Data Dragon 7.17.1 的中文名称/标题、英文标识符及登记别名查询；名称映射不代表逐场比赛补丁版本。
- 对指标定义、数据口径和范围问题检索项目文档，并要求回答给出文档来源。
- 对 API 不可用、Embedding 检索不可用和知识索引缺失返回明确错误；命令行输出使用 UTF-8。
- 会拒绝数据不支持的问题（按颜色方比较胜率、用这份 2017 年数据判断当前版本强度、
  某场比赛的出装等），而不是给出看似合理的数字。

## 技术栈

- Python 3.14.x；依赖版本见 `pyproject.toml` 与 `requirements-lock.txt`。
- LangGraph、LangChain OpenAI-compatible client、OpenAI Python SDK。
- Qwen `qwen3.8-flash` 对话模型；DashScope 兼容模式 API。
- Qwen `qwen3.7-text-embedding`，1024 维文本向量；SQLite 保存本地 RAG 索引。
- CSV 数据由 Python 标准库处理；指标计算不依赖模型生成数值。

## 配置

复制 `.env.example` 为 `.env`，填入 API Key 即可——**其余变量都有可用的默认值**：

```powershell
Copy-Item .env.example .env
# 然后编辑 .env，填入 DASHSCOPE_API_KEY
```

只有 `DASHSCOPE_API_KEY` 是必填项。`.env.example` 里逐项说明了用途与默认值，其中
两点最容易踩坑：

- **端点形态**：默认使用通用域名 `https://dashscope.aliyuncs.com`，**仅凭 API Key 认证**。
  若设了 `DASHSCOPE_WORKSPACE_ID`，服务端会改用 workspace 专属域名并要求 Key 属于该
  业务空间；Key 与空间不匹配时会报 `Workspace endpoint access denied`，看起来像 Key 失效，
  实际是空间不匹配。**所以除非确认 Key 属于该空间，否则不要设置它。**
- **`QWEN_EMBEDDING_MODEL`**：换模型会让已存向量全部失效，必须重跑
  `scripts/index_knowledge.py`。

**改配置后必须重启服务**：`.env` 只在进程启动时读取一次，换 Key 或换模型都不会热生效。

不要把 `.env` 提交到版本库（已在 `.gitignore` 中）。API Key 同时用于对话与向量请求；
调用会产生相应服务用量。

## 使用方式

界面上是三栏布局：会话列表 / 对话流 / **检索过程面板**（改写后的查询、工具调用、命中
分块与分数）。检索面板会显示每个分块的来源与 rerank 分数，便于判断回答依据是否可靠。

命令行入口：

```powershell
# 无状态单轮问答（不读写会话库）
.\.venv\Scripts\python.exe scripts\chat_session.py --once "这份数据去重后有多少场比赛？"

# 多轮对话（会话持久化到 .cache/sessions.db）
.\.venv\Scripts\python.exe scripts\chat_session.py
.\.venv\Scripts\python.exe scripts\chat_session.py --list
.\.venv\Scripts\python.exe scripts\chat_session.py --session <id> -q "那他的胜率呢？"

# HTTP 后端（流式问答，SSE）
.\.venv\Scripts\python.exe scripts\serve.py            # 交互式 API 文档在 /docs
```

多轮对话会在检索前把「历史 + 当前问题」改写成独立查询，因此
「那他的胜率呢？」这类省略主语的追问仍能检索到正确文档；
改写失败时回退到原问题，不会中断整轮。

**后端只监听回环地址。** 它没有身份认证，对外暴露会让任何人都能读取会话记录并消耗 API 配额。

## 前端界面（Vue 3 + Vite + TypeScript）

**仓库已包含构建产物（`frontend/dist`），通常无需自行构建。** 只有在改动前端源码后
才需要重新构建。需要 Node.js 20+。

```powershell
cd frontend
npm install
npm run build          # 产物写入 frontend/dist，由后端托管
```

开发模式（热更新；`/api` 由 Vite 代理到后端）：

```powershell
cd frontend
npm run dev            # http://127.0.0.1:5173
```

**npm 注意事项**：仓库根与 `frontend/` 下都放了 `.npmrc`，把 npm 缓存指向项目内
`.cache/npm`。默认缓存位于工作区之外，在受限沙箱下会导致每次安装都以 `EPERM`
失败。此外 npm 需要用管道创建子进程，沙箱会拒绝该操作，因此 `npm install`
/ `npm run build` 需在放宽权限的终端中执行。

## 目录说明

- `src/lol_agent/`：Agent、统计工具、Embedding 适配器和本地 RAG 检索。核心代码已按
  [`docs/annotation_style_guide.md`](docs/annotation_style_guide.md) 补充详细中文注释，
  适合作为学习 RAG / Agent 工程的入口。
- `docs/`：指标字典、英雄映射说明及 7.17.1 映射数据；`docs/annotation_style_guide.md`
  说明本项目的代码注释规范。
- `kbase/`：检索用的项目文档（指标口径、数据审计、版本冲突样本等），是 RAG 的知识源。
- `reports/data_audit.md`：数据审计结果；`reports/project_closeout.md`：收尾状态、验证
  方式与已知限制；`reports/knowledge.db`：本地构建的检索索引（**不在仓库中**）。
- `evals/`：统计参考案例、RAG/Agent 案例及检索阈值校准数据。
- `scripts/`：CLI、索引、评测与检查入口。开发过程中的一次性工具已归档到
  `scripts/archive/`（用途见该目录 README，**不要在日常流程中运行**）。

## 数据与回答边界

- 不按蓝方/红方拆分或比较胜率；`team1/team2` 仅是单局内队伍标识。
- 数据没有玩家、分路、逐分钟过程或逐场 patch 信息；不能据此回答当前版本或分路排名。
- 装备数据**不在对局 CSV 内**（61 个字段中没有任何装备字段），因此不能回答某场比赛
  出了什么装备，也不能给出装备的选取率或胜率。装备数值只能作为静态目录事实给出。
- 首个事件与获胜只报告统计关联，不声称因果。
- `docs/metric_dictionary.md` 是归档记录（正文保留原样）。它先前标注的"初稿"所依据的
  核实项——用数据集自带的元数据附件独立验证字段定义——**已完成**，结论与复现方式见
  `kbase/project_data_audit.md` 的「元数据附件的独立验证（已完成）」，或直接运行
  `scripts/verify_dataset_metadata.py`。
- 装备数值随补丁变化，而 CSV 没有逐场 `patch` 字段，因此任何装备数值都必须连同补丁
  版本与物品 ID 一起给出。见 `kbase/conflict_item_catalog_drift.md`。

## 本地检查

```powershell
.\.venv\Scripts\python.exe scripts\run_checks.py          # 全部检查（11 步）
.\.venv\Scripts\python.exe scripts\run_checks.py --fast   # 提交前用的快速子集
```

其中 8 步完全离线（语法编译、单元测试、索引一致性、元数据交叉验证、装备漂移核验、
前端单测等），**不调用任何模型 API**；另有 3 步需要模型或本地服务器：

- `SSE 分帧契约`、`并发轮次隔离`：需要本地服务器在 `127.0.0.1:8000` 运行
- `Agent 行为阈值门禁`：默认回放 `.cache/eval_agent_cache.json` 中记录的回答，
  **不调用模型**；缓存为空时才需要调用

无法运行的步骤会标记为 `SKIP` 并附原因，**不会伪装成 PASS**。若账号额度耗尽，
依赖该额度的步骤同样标记为 `SKIP`；额度可能在运行中途耗尽，此时会在步骤结束后
依据输出中的额度标记改判为 `SKIP`，避免把「没跑成」报成「有缺陷」。

## 评测

统计参考案例见 `evals/cases.jsonl`（17 条）；Agent 与 RAG 行为案例见
`evals/rag_cases.jsonl`（15 条），口径与用量记录见 [评测说明](evals/README.md)。

Agent 行为由 `scripts/eval_agent.py` 按阈值门禁（工具选择、未调用禁止工具、检索命中、
数值断言、未出现禁止声称），基线记录在 `evals/agent_baseline.json`：此前通过、现在失败的
用例会被报告为**回归**并阻止；已记录的已知失败与抖动项则单独列出，不会伪装成通过。
默认运行回放缓存，不消耗额度；`--refresh` 才重新提问。

**基线绑定模型**：基线记录它所属的 `chat_model` 与 `temperature`，评测缓存键也包含模型名。
换模型后旧基线不会继续沿用，需要 `--refresh` 后重建（`scripts/write_agent_baseline.py`）。

## 许可证

本项目采用 **MIT 许可证**，见 [LICENSE](LICENSE)。

**第三方内容不适用 MIT 条款**，已在该文件末尾逐项列出来源与条款：

- **Riot Games 静态数据（Data Dragon）**：仓库内由该数据派生的文档与映射
  （`kbase/domain_*.md`、`docs/mapping_*.json`、`docs/champion_mapping_7_17_1.json` 等）
  版权属 Riot Games，按其社区使用条款分发。本项目**未获 Riot Games 认可**，也不代表其
  观点。原始快照（`ShuJuYuan/ddragon/`）未纳入版本库。
- **Kaggle 数据集附带元数据**：`ShuJuYuan/champion_info*.json`、
  `summoner_spell_info.json`，作者 Mitchell J，许可 **CC0 1.0（公有领域）**。
  对局数据本身（`games.csv`，同为 CC0）未纳入版本库，获取方式见上文快速开始。

> 版权署名已设为 **A1pha-Love**（<https://github.com/A1pha-Love>）。项目不再更新，因此
> 提交历史中的 author 仍为本地占位身份（`lol-ranked-data-agent <lol-agent@localhost>`），
> GitHub 不会把这些提交关联到该账号。这只影响提交历史的显示，不影响许可证效力。

## 致谢

- 数据集：[datasnaek/league-of-legends](https://www.kaggle.com/datasets/datasnaek/league-of-legends)
  （Mitchell J，CC0 1.0）
- 静态数据：Riot Games [Data Dragon](https://developer.riotgames.com/docs/lol#data-dragon)
- 模型与检索：阿里云百炼 DashScope（Qwen 对话 / Embedding / Rerank）
- Agent 编排：LangGraph、LangChain
