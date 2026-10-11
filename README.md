# LoL Ranked Data Agent

面向本地 League of Legends 历史排位赛数据的中文分析 Agent。项目通过 LangGraph 编排单 Agent，用 Qwen 对话模型选择工具；精确统计由 Python 工具完成，项目文档由本地 RAG 检索后作为回答依据。

## 项目状态

本地 MVP 已完成并通过阶段性验收，功能、验证记录和已知限制见 [最终验收记录](reports/final_acceptance.md)。项目仅用于本地运行，未发布或部署为线上服务。

## 功能

- 对 `games.csv` 提供确定性统计：去重对局数、数据概览、英雄出场与胜率、Ban 统计、首个事件发生率与条件胜率、平均时长。
- 按 `gameId` 去重；原始 CSV 保持只读。数据为 2017-06-08 至 2017-09-06 UTC 的历史 EUW 排位对局。
- 支持通过英雄数值 ID、Data Dragon 7.17.1 的中文名称/标题、英文标识符及登记别名查询；名称映射不代表逐场比赛补丁版本。
- 对指标定义、数据口径和范围问题检索项目文档，并要求回答给出文档来源。
- 对 API 不可用、Embedding 检索不可用和知识索引缺失返回明确错误；命令行输出使用 UTF-8。

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

不要把 `.env` 提交到版本库（已在 `.gitignore` 中）。API Key 同时用于对话与向量请求；
调用会产生相应服务用量。

## 安装和运行

在项目根目录使用已配置好的虚拟环境：

```powershell
# 无状态单轮问答（不读写会话库）
.\.venv\Scripts\python.exe scripts\chat_session.py --once "这份数据去重后有多少场比赛？"

# 多轮对话（会话持久化到 .cache/sessions.db）
.\.venv\Scripts\python.exe scripts\chat_session.py
.\.venv\Scripts\python.exe scripts\chat_session.py --list
.\.venv\Scripts\python.exe scripts\chat_session.py --session <id> -q "那他的胜率呢？"

# HTTP 后端（流式问答，SSE）
.\.venv\Scripts\python.exe scripts\serve.py            # http://127.0.0.1:8000/docs
```

多轮对话会在检索前把「历史 + 当前问题」改写成独立查询，因此
「那他的胜率呢？」这类省略主语的追问仍能检索到正确文档；
改写失败时回退到原问题，不会中断整轮。

**后端只监听回环地址。** 它没有身份认证，对外暴露会让任何人都能读取会话记录并消耗 API 配额。

## 前端界面（Vue 3 + Vite + TypeScript）

三栏布局：会话列表 / 对话流 / **检索过程面板**（改写后的查询、工具调用、命中分块与分数）。

```powershell
# 开发模式（热更新；/api 由 Vite 代理到后端）
cd frontend
npm install
npm run dev            # http://127.0.0.1:5173
```

生产模式为单端口：先构建，再由后端托管界面。

```powershell
cd frontend
npm run build          # 产物写入 frontend/dist
cd ..
.\.venv\Scripts\python.exe scripts\serve.py
```

后端启动时若发现 `frontend/dist` 存在，会在同一端口挂载界面，直接访问
`http://127.0.0.1:8000` 即可，无需再开 Vite。

**npm 注意事项**：仓库根与 `frontend/` 下都放了 `.npmrc`，把 npm 缓存指向项目内
`.cache/npm`。默认缓存位于工作区之外，在受限沙箱下会导致每次安装都以 `EPERM`
失败。此外 npm 需要用管道创建子进程，沙箱会拒绝该操作，因此 `npm install`
/ `npm run build` 需在放宽权限的终端中执行。

离线检查（不调用任何模型 API，零配额）：

```powershell
.\.venv\Scripts\python.exe scripts\run_checks.py          # 全部离线检查
.\.venv\Scripts\python.exe scripts\run_checks.py --fast   # 提交前用的快速子集
```

如果本地知识索引不存在或源文档发生变化，构建/更新索引：

```powershell
.\.venv\Scripts\python.exe scripts\index_knowledge.py
```

索引命令会扫描 `kbase/` 下的全部 Markdown 文档（当前 42 份，按 `category` 分为
`project` / `domain` / `conflict` / `meta` / `offtopic`），按标题路径分块并调用
Embedding API；原始 CSV 不会被向量化。生成的 `reports/knowledge.db` 是本地文件，
已由 `.gitignore` 排除。`scripts/index_knowledge.py --dry-run` 可在不调用 API 的情况下
预览分块结果，`--check` 可检查索引是否已过期。

## 目录说明

- `src/lol_agent/`：Agent、统计工具、Embedding 适配器和本地 RAG 检索。
- `docs/`：指标字典、英雄映射说明及 7.17.1 映射数据。
- `reports/data_audit.md`：数据审计结果；`reports/final_acceptance.md`：验收结论、验证证据与限制。
- `evals/`：统计参考案例、RAG/Agent 案例及检索阈值校准数据。
- `scripts/`：CLI、索引、审计、检索校准和评测入口。

## 数据与回答边界

- 不按蓝方/红方拆分或比较胜率；`team1/team2` 仅是单局内队伍标识。
- 数据没有玩家、分路、逐分钟过程或逐场 patch 信息；不能据此回答当前版本或分路排名。
- 首个事件与获胜只报告统计关联，不声称因果。
- `docs/metric_dictionary.md` 仍标为初稿：部分原始字段含义依赖公开二手说明，正式对外发布前应再核对原始数据集元数据。

## 评测

统计参考案例见 `evals/cases.jsonl`；Agent 与 RAG 案例、用量记录和评测口径见 [评测说明](evals/README.md)。完整 10 案例回归按额度优先策略只运行过一次；最新提示词变更后的两个受影响案例已定向复测，详情见最终验收记录。
