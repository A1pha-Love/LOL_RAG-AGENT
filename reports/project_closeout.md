# 项目结项记录（2026-10-12）

本文档记录项目**收尾时的实际状态**。`final_acceptance.md`（2026-10-08）是当时的历史验收，
按约定保持原样不改写；两者记录的时间点不同，数字不同属正常，以前者为准。

---

## 一、项目是什么

面向本地 League of Legends 历史排位赛数据的中文分析 Agent。通过 LangGraph 编排单 Agent，
由 Qwen 对话模型选择工具；**精确统计由 Python 工具完成，不让模型估算**；项目文档经本地
RAG 检索后作为口径依据，并要求回答给出来源。

**运行状态**：仅本地运行，未发布、未部署。无认证、无限流、无导出、无反馈入口——这是
有意的范围界定，不是缺失。

## 二、收尾时的规模

| 项 | 数量 |
|---|---|
| 提交 | 37 |
| 后端测试 | **208 例**（11 个文件）|
| 前端测试 | **70 例**（8 个文件）|
| 本地检查步骤 | **11 步**（其中 8 步完全离线）|
| Agent 工具 | 11 |
| HTTP 路由 | 10（7 个 `/api` 接口 + `/` + SPA 兜底 + 静态资源挂载）|
| 知识库 | 44 份文档 / 1171 个分块 |
| `scripts/` 脚本 | 19 个（另有 21 个一次性工具归档在 `scripts/archive/`）|
| 统计参考案例 | 17 条（`evals/cases.jsonl`）|
| Agent 行为案例 | 15 条（`evals/rag_cases.jsonl`）|
| 检索标定案例 | 31 条（`evals/retrieval_calibration.jsonl`）|
| 数据 | 51,490 原始行 / 61 列 → 51,053 唯一对局（437 条重复行，分布在 424 组）|

## 三、如何验证（可复现）

```powershell
# 1. 全部本地检查（11 步）
.\.venv\Scripts\python.exe scripts\run_checks.py

# 2. Agent 行为门槛（默认回放缓存，不消耗额度）
.\.venv\Scripts\python.exe scripts\eval_agent.py
.\.venv\Scripts\python.exe scripts\eval_agent.py --refresh      # 重新提问

# 3. 单独验证各项
.\.venv\Scripts\python.exe scripts\verify_dataset_metadata.py   # 元数据交叉验证
.\.venv\Scripts\python.exe scripts\verify_item_drift.py         # 装备漂移数字核对
.\.venv\Scripts\python.exe scripts\verify_retrieval_offline.py  # 检索断言（离线）
.\.venv\Scripts\python.exe scripts\verify_api_endpoints.py      # 接口冒烟（需服务器）
.\.venv\Scripts\python.exe scripts\check_api_key.py             # 三个端点连通性

# 4. 前端
cd frontend; npm test; npm run typecheck; npm run build
```

`scripts/` 下是收尾后仍在用的工具；开发过程中的一次性探针与已完成的一次性构建已归档到
`scripts/archive/`，用途见该目录的 README——保留是为了记录当时的结论如何得出。

**收尾时的结果**：检查 11/11 通过；后端 208 例、前端 70 例全部通过；Agent 行为门槛
14/15 达标 + 1 个已记录的已知失败（见下）；离线检索断言 13/13。

**Agent 行为门槛的判分方式**：五个语义阈值逐例强制——工具选择、未调用禁止工具、
检索命中、数值断言、未出现禁止声称。答案措辞（`answer_must_include`）是子串启发式，
仅测量、不作门槛，避免把措辞差异当成缺陷；但「一项都未命中」会单独列出，因为那是漏答
的特征。基线与回归判定见 `evals/agent_baseline.json`：此前通过、现在失败的用例报为
**回归**并阻止；已记录的失败与抖动项单独列出，不伪装成通过。

## 四、已知限制（交付时必须一并说明）

按"是否影响结论正确性"分为两类。

### A. 影响答案完整性（1 项）

**`spell_pick_rate_denominator`：模型从文档作答，不调用统计工具。**
问"召唤师技能出场率的分母是比赛数还是队伍出场次数"，模型调用检索后回答
「分母是队伍出场次数（即去重后比赛数 × 2）」，并正确引用
`project_metrics_summoner_spell.md`。

- **答案内容正确**：与工具返回值 102106 数学等价，且直接回答了提问。
- **未给出具体数值**（102106 / 99.998%），而该用例要求调用 `get_spell_stats` 并给出数值。
- 在 temperature=0 下两次一致，属**确定性行为**，不是抖动。

该用例的断言存在**内部不一致**：`answer_must_include` 接受多种措辞（"队伍出场"/"队伍出现"/
"每场两支队伍"），而数值断言只认字面 102106，不接受等价的「× 2」写法。属结项时的待决事项，
未擅自放宽。

### B. 边界与降级路径（不影响正常使用）

1. **fallback 检索阈值处于重叠区**：用 31 条标定案例实测最佳 embedding 相似度，
   相关类（n=15）为 **0.6018–0.8799**，不可答类（n=13）为 **0.3420–0.6333**，
   两者**确有重叠**，因此 `MIN_RETRIEVAL_SCORE = 0.3703` 无法同时保证安全与完整。
   **仅在 reranker 不可用时生效**；reranker 正常时不走该分支。
   （复算方式：`scripts/calibrate_retrieval.py`。）
2. **口语化检索的阈值边界**：实测「平均推掉多少座防御塔」rerank 0.6873，略低于
   `MIN_RERANK_SCORE = 0.70` 而被挡掉（尽管命中的正是目标小节）。这类问句走统计工具
   不受影响；影响的是"问口径"的口语化问法。已记录于 `project_metrics_core.md`。
3. **中文标签未与英文附件逐字比对**：项目快照为 Data Dragon 7.17.1（zh_CN），数据集
   附件为 7.17.2（en_US），语言不同。可严格比对的部分——英雄集合与英文标识符→数值 ID
   映射——**已全量核对（138/138 零差异）**。
4. **装备数据不在对局 CSV 内**：61 个字段中无任何装备字段，也无逐场 patch 字段。
   因此不能回答某场比赛的出装、装备选取率或胜率；装备数值只能作为静态目录事实给出，
   且必须连同补丁版本与物品 ID。
5. **项目快照与数据集附件版本不同**：锁定 7.17.1，附件为 7.17.2；已实测差异仅为版本号。

## 五、数据与回答边界（不可逾越）

- 不按蓝方/红方拆分或比较胜率；`team1/team2` 仅是单局内队伍标识。
- 无玩家、分路、逐分钟过程、逐场 patch 信息；不能据此回答当前版本或分路排名。
- 首个事件与获胜只报告统计关联，不声称因果。
- 装备数据不在 CSV 内（见上）。
- 数据为 2017-06-08 至 2017-09-06 UTC 的历史对局，`seasonId` 恒为 9。
- **会话顺序按真实活动变化**：打开（查看）会话不改变列表顺序，只有该会话产生新对话后才
  升到非置顶组首位，且始终不越过置顶组。此前"打开即上移"是实现缺陷，已修（曾导致列表
  在光标下跳动，看起来像点错了会话）。

## 六、运行方式

```powershell
# 首次准备
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
Copy-Item .env.example .env      # 填入 DASHSCOPE_API_KEY
.\.venv\Scripts\python.exe scripts\index_knowledge.py   # 构建知识索引（需额度）

# 使用
.\.venv\Scripts\python.exe scripts\chat_session.py --once "去重后有多少场比赛？"
.\.venv\Scripts\python.exe scripts\serve.py             # HTTP + 界面
# 浏览器打开 http://127.0.0.1:8000
```

**改配置后必须重启服务**：`.env` 在进程启动时读取一次，换 key 或改模型不会热生效。

## 七、明确不做的事

以下均为**有意的范围界定**，非未完成项：

- 对外部署、容器化、进程守护
- 认证与配额限制
- 会话导出、用户反馈入口

若将来需要对外提供服务，才需补上述能力；当前形态仅面向本地单人使用。

## 八、结论

**本地 MVP 完成，检查与测试全绿，已知限制已逐条记录并可复现核验。**

代码中无 `TODO` / `FIXME` / 未实现分支；无失败的测试；唯一的已知失败已在上文说明其
性质（答案正确但未给出具体数值，且断言内部不一致）。五项设计边界为有意选择，不视为欠账。
