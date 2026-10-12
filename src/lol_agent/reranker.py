"""Cross-encoder reranking for retrieved chunks, via DashScope `qwen3-rerank`.

Retrieval with embeddings answers "which chunks look similar"; a reranker scores
each candidate *against the query jointly*, which is what separates a broad
document-level chunk from the focused sub-section that actually answers the
question.

The endpoint is workspace-scoped and differs from the OpenAI-compatible path
used for chat and embeddings:

    POST https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-api/v1/reranks

Note the docs' warning: `relevance_score` is a **relative** score for one
request, so it orders candidates within a query but is not comparable across
queries. It must not be used as a global no-answer cutoff.

Reranking is best-effort: any failure raises `RerankUnavailable`, and callers
fall back to the embedding order rather than failing the request.
"""

# ============================================================================
# 中文说明
#
# 职责：把「向量召回」得到的候选片段交给 DashScope 的 cross-encoder 重排模型
# （qwen3-rerank），用「查询 + 文档拼在一起打分」的方式重新排序。
#
# 在整体链路中的位置：
#   knowledge.search_knowledge 先用 embedding 余弦相似度粗排，取
#   RERANK_CANDIDATE_POOL 个候选 → 调用本模块 rerank() 精排 → 按 rerank_score
#   排序并按 MIN_RERANK_SCORE 截断。
#
# 为什么需要这一层：embedding 是「双塔」结构，查询与文档各自独立编码，只能回答
# 「这两段文字像不像」；cross-encoder 让查询与文档在同一遍前向计算里互相注意，
# 因而能把「文档级的大分块」和「真正回答问题的那一小节」区分开。项目实测的结果
# （写在 knowledge.py 的常量注释里）是：只靠余弦相似度时，可回答类与不可回答类的
# 分数区间互相重叠，无法靠单一阈值分开，所以精排是主路径而不是可选优化。
#
# 关键约定与踩过的坑：
#   1. 端点与 chat/embedding 不同：rerank 走 workspace 专属域名（见
#      config.rerank_url()），因此不能复用 OpenAI 兼容的那套 base_url。
#   2. 模块开头的英文 docstring 已记录官方警告：relevance_score 是**单次请求内**
#      的相对分，跨 query 不可比。项目仍然在 knowledge.MIN_RERANK_SCORE 处用了一个
#      全局截断值，依据是 31 条校准集上的实测分布（见 knowledge.py 注释），
#      属于基于自有数据的经验取值，而非官方保证的用法。
#   3. 只要重排不可用（缺配置、网络/超时、HTTP 错误、返回体结构异常、条目格式错误），
#      一律抛 RerankUnavailable，由调用方降级回向量顺序；重排是「尽力而为」，
#      不能让整轮问答失败。
#   4. 返回 (原始下标, 分数) 而不是重排后的文本列表：调用方必须把分数写回原来的
#      候选对象，而下标是唯一可靠的对应关系（服务端返回的顺序和条数都不该被依赖）。
# ============================================================================

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from .config import rerank_url, require_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_RERANK_MODEL = "qwen3-rerank"
# The reranker reads the question, so the instruction steers what "relevant"
# means. This is the documented default for question-answering retrieval.
# 中文说明：instruct 是重排接口的一个入参，模型会读它来决定「相关」的判定标准。
# 这里用的是官方为「问答式检索」给出的推荐句式。注意项目里的检索请求既有自然语言
# 问句，也有「这个字段是什么意思」式的定义型查询，此处并未按查询类型分支。
QA_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."
# 中文说明：重排是一次同步 HTTP 调用，超时后会被下面 rerank() 的兜底 except 转成
# RerankUnavailable，从而降级为向量顺序——也就是说超时会让本轮检索质量下降，但
# 不会让请求挂死。具体秒数是经验取值。
REQUEST_TIMEOUT_SECONDS = 60


class RerankUnavailable(RuntimeError):
    """The rerank endpoint could not be used (missing config, network, or API error)."""

    # 中文说明：把「重排这一层用不了」这一件事收敛成一个异常类型。调用方
    # （knowledge.search_knowledge）只需要捕获它就能实现降级，不必区分是缺 API Key、
    # DNS 失败还是返回体变了形状。


def rerank_model() -> str:
    # 中文说明：模型名每次现读环境变量，而不是在导入时固化——这样测试或脚本可以在
    # 导入之后再改环境变量。末尾的 `or DEFAULT_RERANK_MODEL` 用来兜住
    # `QWEN_RERANK_MODEL=""`（空串）的情况：空串会被当作「没配」，而不是当成模型名。
    return os.getenv("QWEN_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip() or DEFAULT_RERANK_MODEL


def rerank_enabled() -> bool:
    """Reranking is on unless explicitly disabled."""
    # 中文说明：默认开启。只有显式把 RERANK_ENABLED 设成 0/false/no/off 之一才关闭，
    # 于是「环境变量忘了配」走的是质量更好的精排路径；大小写与首尾空格都被容忍，
    # 因为 .env 与不同 shell 里写 "False"、"off " 都很常见。
    return os.getenv("RERANK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _endpoint() -> str:
    # 中文说明：包一层函数是为了把 URL 的拼装细节留在 config 里（region、WorkspaceId
    # 从环境变量来），本模块只关心协议本身。
    return rerank_url()


def rerank(
    query: str,
    documents: list[str],
    *,
    top_n: int | None = None,
    instruct: str = QA_INSTRUCT,
) -> list[tuple[int, float]]:
    """Return [(original_index, relevance_score)] ordered best-first.

    Raises `RerankUnavailable` on any failure so the caller can fall back.
    """
    # 中文说明：
    #   用途：对一批候选文档按「与 query 的相关性」重新打分并排序。
    #   参数：query     —— 本次检索真正使用的查询文本；多轮会话里它已经是
    #                      conversation.rewrite_query 改写后的独立查询，而不是用户原话。
    #         documents —— 候选片段正文列表，其**顺序**就是返回下标的参照系，
    #                      所以调用方传入的列表不能在两次使用之间被重排。
    #         top_n     —— 只要前若干条；服务端要求它不超过文档数，见下面的 min()。
    #         instruct  —— 相关的判定标准（默认问答式检索的官方句式）。
    #   返回：(原始下标, relevance_score) 的列表，按分数降序；空输入返回空列表。
    #   异常：任何失败都抛 RerankUnavailable——包括「缺 API Key」，它本来是
    #         RuntimeError，也被归一化成同一个异常，调用方只需捕获一种。
    if not documents:
        # 中文说明：空列表直接短路，省掉一次必然无意义的 HTTP 往返（也避免服务端对
        # 空 documents 报参数错误，从而把「本来没有候选」误报成「重排不可用」）。
        return []
    try:
        api_key = require_api_key()
    except RuntimeError as exc:
        raise RerankUnavailable(str(exc)) from exc
    payload: dict[str, object] = {
        "model": rerank_model(),
        "query": query,
        "documents": documents,
        "instruct": instruct,
    }
    if top_n is not None:
        # 中文说明：服务端约束 top_n 不能超过文档条数，这里用 min() 兜住，避免因为
        # 多要几条而吃到 400 参数错误（那会白降级一次）。
        payload["top_n"] = min(top_n, len(documents))

    # 中文说明：直接用标准库 urllib 发 POST，请求体是 JSON，鉴权沿用 OpenAI 兼容的
    # Bearer 头；只有 URL 换成了 workspace 专属的 rerank 端点。
    request = urllib.request.Request(
        _endpoint(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        # 中文说明：HTTPError 单独先捕获——它身上带着服务端返回的错误正文（配额不足、
        # 鉴权失败、参数非法都会在这里体现）。正文截断到 200 字符，足够定位问题，
        # 又不会把整页响应塞进异常信息和日志。
        # `decode(..., "replace")`：错误正文未必是合法 UTF-8，用 replace 保证解码本身
        # 不会再抛异常（在异常处理里再抛异常会把原始错误盖掉）。
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise RerankUnavailable(f"HTTP {exc.code}: {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        # 中文说明：兜底分支覆盖超时、DNS/SSL 失败、连接被重置、响应不是 JSON 等。
        # 这里的宽 except 是设计而非疏忽（noqa: BLE001 即为此）：重排是尽力而为的一层，
        # 任何意外都必须变成「降级」而不是「整轮问答报错」。
        raise RerankUnavailable(f"{type(exc).__name__}: {exc}") from exc

    # qwen3-rerank puts `results` at the top level; gte-rerank-v2 nests it.
    # 中文说明：两种返回体形状都试一遍，这样换用 gte-rerank-v2 时不必改代码。
    results = body.get("results")
    if results is None and isinstance(body.get("output"), dict):
        results = body["output"].get("results")
    if results is None:
        # 中文说明：形状对不上时把实际出现的顶层键名（前 6 个）写进错误信息便于排查；
        # 只取键名、不打印正文，避免把整段文档内容写进日志。
        raise RerankUnavailable(f"unexpected response shape: {sorted(body)[:6]}")

    ranked: list[tuple[int, float]] = []
    for item in results:
        try:
            # 中文说明：服务端可能把数字给成字符串，因此显式 int()/float() 转换。
            # 注意：只要有**一条**不符合预期，就整批抛 RerankUnavailable 走降级，
            # 而不是跳过坏条目继续用其余分数——这样不会出现「下标与分数对应关系被
            # 悄悄破坏」的结果。
            ranked.append((int(item["index"]), float(item["relevance_score"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankUnavailable(f"malformed result entry: {item}") from exc
    # 中文说明：本地再排一次序，不依赖服务端返回时已按分数排列（也不能依赖，因为
    # 服务端返回顺序并非契约的一部分）。本函数对调用方承诺的就是 best-first。
    ranked.sort(key=lambda pair: -pair[1])
    return ranked
