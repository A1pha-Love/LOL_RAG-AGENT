# 归档脚本

这些脚本是开发过程中的**一次性工具**：为某次排查而写的探针、或已完成的阶段性构建。
它们不再需要被运行，保留在这里是为了留下"当时的结论是怎么得出来的"这一记录——其中
若干结论被 `reports/project_closeout.md` 与 `kbase/` 引用。

**不要在收尾后的日常流程中使用它们**；检查套件与日常入口仍在上层 `scripts/` 目录。
若需要复算某个结论，对应的现行工具是：

| 结论 | 现行复算方式 |
|---|---|
| 检索来源是否能被召回 | `scripts/verify_retrieval_offline.py` |
| 检索阈值标定区间 | `scripts/calibrate_retrieval.py`（未归档）|
| 数据集字段与元数据一致 | `scripts/verify_dataset_metadata.py` |
| 装备漂移数字 | `scripts/verify_item_drift.py` |
| Agent 行为是否达标 | `scripts/eval_agent.py` |
| 全部检查 | `scripts/run_checks.py` |

各脚本的用途：

- `audit_data.py`：生成 reports/data_audit.md 的原始数据审计（已完成的阶段性产物）
- `build_dimension_mappings.py`：从抓取的维度数据构建 ID 映射（一次性构建）
- `build_kbase_from_dimension.py`：从维度数据生成知识库文档（一次性构建）
- `calibrate_threshold.py`：测量 rerank 分数分布以推导阈值（结论已写入知识库）
- `check_api_stream.py`：流式 API 的人工联通性检查（已被 verify_sse_contract.py 取代）
- `check_retrieval_event.py`：确认定义类问题会返回 retrieval 事件（一次性验证）
- `check_spell_tools.py`：召唤师技能工具接入时的抽查脚本
- `check_web_app.py`：内置界面的端到端人工验收（已被 verify_api_endpoints.py 取代）
- `compare_strategies.py`：比较检索打分策略（加权和 vs 倒数排名）
- `decode_key.py`：解析 sk-ws- key 中内嵌的 workspace id（排查 403 时用）
- `fetch_dimension_data.py`：抓取维度扩充所需的 Data Dragon 数据
- `measure_history_size.py`：测量会话历史的实际字符数（用于确定裁剪预算）
- `measure_rerank.py`：测量 embedding-only 与 rerank 的 Recall@k
- `measure_rewrite.py`：测量检索查询改写对真实追问的影响
- `measure_stream_timing.py`：测量流式分块到达时机（排查首字延迟）
- `probe_endpoints.py`：对比 workspace 专用域名与通用域名（排查 403）
- `probe_node_env.py`：确认 npm 是否可用（排查沙箱 EPERM）
- `refresh_index_hashes.py`：刷新索引源哈希（修复过一次元数据过期，见其文档字符串）
- `relax_retrieval_expectations.py`：一次性迁移：为用例加入 alternatives 与 optional_tools
- `time_question.py`：给单个问题端到端计时（排查延迟）
- `validate_retrieval_candidates.py`：写入标定用例前校验候选

## 归档动作本身

把上述脚本移入本目录的迁移脚本（rchive_one_off_scripts.py）已在执行后删除：
它是一次性动作，重复运行没有意义，而各文件的用途已记录在本文档中。
