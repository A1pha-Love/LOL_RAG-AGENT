"""Deterministic analytics for the LoL ranked match dataset."""

# ============================================================================
# 中文说明
#
# 包入口。本包 = 本地英雄联盟排位数据的确定性分析 Agent：
#
#   metrics.py      指标口径与纯计算（分母/分子、去重、无样本语义）
#   tools.py        给模型调用的 11 个工具：读 CSV、解析中文/ID 参数、补中文标签
#   conversation.py 会话与轮次的持久化
#   knowledge.py    知识库索引与两阶段检索
#   reranker.py     rerank 服务封装
#   embeddings.py   向量化适配器（1024 维，固定模型）
#   config.py       DashScope 域名与 API key 的唯一出口
#   agent.py        LangGraph 单 Agent：改写查询 -> 选工具 -> 生成回答
#   streaming.py    一轮问答的事件流（SSE 的服务端无关实现）
#
# 贯穿全局的原则：**所有统计数字由 Python 计算，模型只负责选择工具与组织语言**；
# 指标口径固定写在各模块与 kbase/ 文档里，不由模型即兴决定。
#
# 这里刻意不 re-export 任何子模块：使用方按需 `from . import metrics`，
# 避免 `import lol_agent` 时连带加载向量库、HTTP 客户端等重依赖。
# ============================================================================
