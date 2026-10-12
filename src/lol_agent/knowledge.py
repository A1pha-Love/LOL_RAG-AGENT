"""Local retrieval over the project's curated Markdown knowledge base.

The knowledge base lives in ``kbase/`` (workspace root). Every document is a
Markdown file carrying YAML front matter, discovered automatically — there is
no hardcoded source list any more.

Chunking rules (replacing the old fixed-character-window splitter):

* headings define sections, and a **heading path** is carried in the chunk text
  so a retrieved fragment is self-describing (``文档 > 节 > 小节``);
* Markdown tables are atomic — a table row is never separated from its header;
* undersized pieces are merged up to a minimum size, because a 13-character
  chunk competes with real content for retrieval slots and carries no context.

The index stores chunk metadata plus source hashes, so staleness is detectable
instead of silent.
"""

# ============================================================================
# 中文说明
#
# 职责：本地知识库（kbase/*.md）的发现与解析、按标题路径分块、SQLite 索引构建、
# 两阶段检索（向量召回 → rerank 精排）、阈值过滤，以及来源可追溯。
#
# 在整体架构中的位置：agent.py 里的 search_project_knowledge 工具调用本模块的
# search_knowledge()；conversation.py 先把多轮追问改写成可独立理解的查询，再交给
# 这里检索；reranker.py 提供第二阶段的精排。索引由 scripts/index_knowledge.py 显式
# 重建，运行期只读——所以「索引不存在/过期」是一种需要用户动手的显式错误，
# 而不是被静默兜住的状态。
#
# 本文件里最值得学习的几处设计取舍（详见各处的行内注释）：
#   1. 为什么按「标题路径」分块，而不是固定字符窗口；
#   2. chunk_id 里为什么带 ordinal，以及它带来的「文档一改，后续 id 全部位移」；
#   3. 两阶段检索与 RERANK_CANDIDATE_POOL 的意义；
#   4. MIN_RERANK_SCORE 与 MIN_RETRIEVAL_SCORE 两个阈值各管什么、已知的重叠问题；
#   5. 干扰文档（offtopic/noise）为什么在 SQL 层排除，而不是排完序再过滤。
# ============================================================================

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embeddings import build_embeddings
from .reranker import RerankUnavailable, rerank, rerank_enabled

# 中文说明：parents[0]=lol_agent 包目录，parents[1]=src，parents[2]=仓库根目录。
# 这样路径不依赖「从哪个目录启动进程」，无论 CLI、脚本还是 Web 后端都指向同一份
# kbase/ 与索引文件。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KBASE_DIR = PROJECT_ROOT / "kbase"
# 中文说明：索引是构建产物（可随时重建），因此与源文档分开放在 reports/ 下。
# 注意对比 conversation.py：会话数据库刻意放在 .cache/，因为那个文件需要在运行时可写；
# 而知识索引只在重建索引时写，运行期只读。
INDEX_PATH = PROJECT_ROOT / "reports" / "knowledge.db"

# --- chunking -------------------------------------------------------------
# 中文说明（分块参数的取舍，都在这里集中说明）：
#   MAX_CHUNK_CHARS —— 单块字符上限。块越大，一次能完整带上的信息越多，但检索粒度越粗
#       （命中后模型要读进大量无关文字），embedding 也越容易被「语义平均」掉。
#   MIN_CHUNK_CHARS —— 单块下限。英文模块 docstring 已写明原因：十几个字符的碎片会跟
#       真正的内容抢检索位次，而且自身没有上下文。低于此值的块会在 _merge_blocks 的
#       第二遍里折进「标题路径相同」的邻居。
#   OVERLAP_CHARS —— 只用于超长块的滑动切分（_split_oversized）：相邻窗口重叠这么多
#       字符，避免一句关键话正好落在窗口边界上、两侧都读不完整。
#   CHUNKER_VERSION —— 写进索引元数据，标记「这批分块是哪个版本的分块器产物」；
#       分块逻辑一变就应改这个字符串，旧索引因此可以被识别出来。
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200
# 注意：MAX_TABLE_CHARS 在本模块（以及仓库内其他 .py 文件）都没有被引用；
# 表格分块目前实际只受 MAX_CHUNK_CHARS 约束。此处仅如实指出，未改动代码。
MAX_TABLE_CHARS = 2400
OVERLAP_CHARS = 160
CHUNKER_VERSION = "heading-path-v2"

# --- retrieval ------------------------------------------------------------
# 中文说明：DashScope 的 Embedding 接口单次请求最多接受 20 条输入，所以这个批量大小是
# 接口约束而不是性能旋钮；embed_texts() 依赖它（那里的英文 docstring 也记录了同一约束）。
EMBEDDING_BATCH_SIZE = 20
# 中文说明：向量维度。它只被写进索引元数据，供人工核对「索引是用哪个模型建的」；
# 运行期真正的维度校验在 search_knowledge 里（索引向量长度 vs 查询向量长度）。
EMBEDDING_DIMENSIONS = 1024
# Scoring is vector-only by default. Measured on evals/retrieval_calibration.jsonl
# (17 scored cases, heading-path chunker), recall@3 was:
#   vector only          10/17
#   vector + bigram       9/17
#   RRF(vector,bigram,bm25)  4/17
# Retrieval here is Chinese-query over Chinese prose, and SQLite FTS5 ships no
# CJK tokenizer, so term matching contributes noise rather than signal. The
# FTS5 table is still built so the comparison stays reproducible via
# scripts/compare_strategies.py; the weights below keep it out of the score.
# 中文说明：similarity = VECTOR_WEIGHT*余弦 + BM25_WEIGHT*bm25 + LEXICAL_WEIGHT*bigram。
# 当前后两项权重都是 0，所以 similarity 在数值上就等于余弦相似度。这些实验结论（为什么
# 关掉词法项）见上面的英文注释——中文要点是：本项目的检索是「中文问题 → 中文正文」，而
# SQLite FTS5 不自带 CJK 分词器，词项匹配带来的主要是噪声。FTS5 表仍然构建，是为了让
# scripts/compare_strategies.py 能复现那次对比实验，而不是因为它参与打分。
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 0.0
LEXICAL_WEIGHT = 0.0
# No-answer cutoff when reranking. Measured top-1 rerank score over all 31 cases in
# evals/retrieval_calibration.jsonl (reproduce with scripts/calibrate_threshold.py):
#
#   relevant (in scope)      n=15   0.7929 – 0.9790
#   unanswerable             n=13   0.2840 – 0.6185
#   offtopic (excluded)      n=3    0.8453 – 0.9453
#
# What this cutoff does and does not do:
#
#   * It separates "the corpus has something on this" from "it has nothing". The
#     feasible band is [0.6185, 0.7929]; 0.70 sits inside it, roughly centred
#     (+0.0815 above the ceiling, -0.0929 below the floor).
#   * It does NOT separate relevant from offtopic — those ranges overlap almost
#     entirely, so no score can do it. Distractors are therefore excluded by
#     category (see NON_CONTENT_CATEGORIES) and must stay out of any separability
#     calculation, or the cutoff looks impossible to choose.
#   * A high score means "this chunk is on the same topic", not "the question is
#     fully answerable". The case `specific_patch_notes` scores 0.7929 because the
#     corpus can state when the rework happened and that it accompanied a rename,
#     even though it holds no line-by-line patch notes. It is labelled relevant for
#     that reason; do not reclassify such cases as unanswerable.
#
# The band is wide enough that the exact value is not delicate, but it is still a
# provisional choice from 31 queries: review failures before adjusting it.
# 中文说明（全项目最需要理解的一个阈值）：
#   * 上面的英文注释给出了 31 条校准用例上的实测分布：可回答 0.7929–0.9790、
#     不可回答 0.2840–0.6185、干扰文档 0.8453–0.9453。0.70 落在「可行区间」
#     [0.6185, 0.7929] 之内，作用是区分「库里有相关材料」与「库里什么都没有」。
#   * 它**不能**区分「可回答」与「干扰文档」：两者的分数区间几乎完全重叠，任何分数
#     阈值都做不到。干扰文档因此改用「按分类结构性排除」处理（见
#     NON_CONTENT_CATEGORIES），并且必须从可分性计算里剔除，否则这个阈值看起来
#     「怎么选都不可能」。
#   * 高分的含义是「这段文字和问题同一主题」，而不是「问题能被完整回答」。英文注释里
#     `specific_patch_notes`（0.7929）就是这种情况，它被标为 relevant 是刻意的。
#   * 平台口径的张力：reranker.py 的模块注释保留了官方警告——relevance_score 是单次
#     请求内的相对分、跨 query 不可比。这里仍把它当全局阈值使用，依据是上面这套自有
#     校准集的实测分布，属于基于自有数据的经验选择。
#   注意：样本只有 31 条查询，英文注释末句也强调「调整阈值前先复盘失败案例」。
MIN_RERANK_SCORE = 0.70
# Candidate pool handed to the reranker, and the cutoff used when reranking is
# unavailable (the vector score is a different scale and cannot share the above).
# 中文说明：交给 reranker 精排的候选池大小。向量召回是「粗排」，它给出的名次并不可靠，
# 所以不能只把 top_k 条送进精排——池子太小会把真正正确的片段挡在门外，而召回阶段没进池
# 的片段，精排再准也救不回来；池子太大则徒增一次 HTTP 请求的体积和延迟。20 是本数据规模
# 下的折中值。
# 注意：池子大小与 top_k 是两件事。search_knowledge 用 max(RERANK_CANDIDATE_POOL, top_k)
# 计算池子，保证「调用方要得比默认池子还多」时不会反而少给候选。
RERANK_CANDIDATE_POOL = 20

# Cutoff for the embedding-only fallback path, used only when the reranker is
# unavailable. Measured on the same cases with scripts/calibrate_retrieval.py:
#
#   relevant (in scope)   0.5921 – 0.8799
#   unanswerable          0.3419 – 0.6334
#
# These OVERLAP (relevant floor 0.5921 sits below the unanswerable ceiling 0.6334),
# so embedding similarity alone cannot be the no-answer gate — which is exactly why
# reranking is the primary path. The fallback is best-effort: it still filters
# obvious misses, and it cannot be tuned to be both safe and complete. Treat a
# fallback turn as "retrieval degraded" rather than "no answer exists".
# 中文说明：这个阈值只用于「rerank 不可用」的降级路径，作用在 similarity（余弦相似度）
# 这个完全不同的量纲上，因此与 MIN_RERANK_SCORE 不可互换。上面英文注释给出的关键实测
# 事实是：可回答 0.5921–0.8799 与 不可回答 0.3419–0.6334 **区间重叠**（前者的下界低于
# 后者的上界），所以单靠 embedding 相似度做不了「无答案」判定——这正是精排成为主路径的
# 原因。降级路径只能尽力而为：它仍能挡掉明显不相关的命中，但无法既安全又完整。因此一次
# 降级回合应被理解为「检索质量下降」，而不是「知识库确实没有答案」。
MIN_RETRIEVAL_SCORE = 0.3703

# 中文说明：文档 front matter 中 category 的允许取值。project / domain / conflict / meta
# 属于真知识；offtopic / noise 是「存在就是为了被拒绝」的干扰样本（见下面
# NON_CONTENT_CATEGORIES）。取值必须精确匹配——load_document 不做同义词或大小写兜底，
# 目的是让写错的分类在构建索引时就暴露，而不是变成检索时无法解释的怪现象。
VALID_CATEGORIES = {"project", "domain", "conflict", "meta", "offtopic", "noise"}

# Categories that are deliberately not knowledge: official and project documents
# answer questions, these exist to be *rejected*.
#
# `kbase/00_scheme.md` defines offtopic as "与项目无关的干扰文档，仅用于阈值校准",
# so they are calibration fixtures, not content. Their rerank scores overlap the
# relevant range almost entirely (0.8453–0.9453 versus 0.8099–0.9790), which means
# no score cutoff can exclude them without also dropping legitimate hits — the
# measurement is in evals/retrieval_calibration.jsonl and reproduced by
# scripts/calibrate_threshold.py. They are therefore excluded structurally, by
# category, rather than by score.
#
# Calibration scripts pass include_noise=True to measure them anyway.
# 中文说明：这两类不是知识，而是校准用的干扰项，所以默认在检索时被「按分类」排除
# （理由与实测区间见上方英文注释）。校准脚本通过 include_noise=True 仍能测到它们，
# 因此这个集合只影响默认检索路径。
NON_CONTENT_CATEGORIES = frozenset({"offtopic", "noise"})
# 中文说明：front matter 里必须存在的三个键。缺任何一个都会让 load_document 直接抛
# KnowledgeDocumentError——即「宁可构建索引时报错，也不要静默产出没有来源、无法引用的
# 脏分块」。这也是 answer 端能标注文档路径与标题的前提。
REQUIRED_META = ("doc_id", "title", "category")


class KnowledgeIndexError(RuntimeError):
    """The local knowledge index is missing, empty, or incompatible."""

    # 中文说明：索引层的问题——文件不存在、内容为空、schema 不兼容、数据库不可读、
    # 向量维度对不上。上层的处置动作是「重建索引」（scripts/index_knowledge.py）。


class KnowledgeAPIError(RuntimeError):
    """The Qwen Embedding request failed."""

    # 中文说明：模型服务侧的问题——Embedding 调用失败（网络、地域、Workspace、API Key）。
    # 与 KnowledgeIndexError 分开是刻意的：两者让用户做的事完全不同，一个是跑脚本重建，
    # 一个是查配置与网络。agent.py 对这两类错误都会转成给用户看的提示，而不是抛栈。


class KnowledgeDocumentError(RuntimeError):
    """A knowledge-base document is malformed."""

    # 中文说明：知识库文档本身不合规——缺少必需 front matter 字段，或 category 不在
    # VALID_CATEGORIES 内。构建索引时就会暴露，避免脏数据进入检索。


@dataclass(frozen=True)
class KnowledgeDocument:
    # 中文说明：一份知识库文档的不可变快照（frozen=True，构建后不再改动）。
    # 字段来源：path/body 来自磁盘，其余来自 front matter；content_hash 用于过期检测。
    # 注意：content_hash 是对**整份原文（含 front matter）**取哈希，因此
    # check_index_staleness 必须用同样的方式计算，否则会把每份文档都判成「已变更」。
    path: Path
    doc_id: str
    title: str
    category: str
    version: str
    source_url: str
    license: str
    source_type: str  # 文档来源类型（front matter 里的 source_type，缺省 "unknown"）
    retrieved_at: str  # 该文档的取回时间，原样存进索引，供审计/溯源
    body: str  # 去掉 front matter 之后的正文
    content_hash: str


@dataclass(frozen=True)
class KnowledgeChunk:
    # 中文说明：检索的最小单位。除了正文 text，还带齐了「回答里要标注的来源信息」
    # （source 路径、heading_path、category、version），所以命中之后不必回查索引。
    chunk_id: str  # 形如 f"{doc_id}#{ordinal:04d}-{digest}"，详见 chunk_document
    doc_id: str
    source: str  # 相对仓库根的 POSIX 风格路径，例如 kbase/00_scheme.md
    heading: str  # 分块的标题路径；没有标题时退化为文档标题（保证引用时不会为空）
    heading_path: str  # 纯标题路径，无标题时是空串（注意与 heading 的差别）
    category: str
    version: str
    source_url: str
    license: str
    text: str
    ordinal: int  # 在本**文档内**的序号，从 0 开始；用于生成 chunk_id
    # 中文说明：compare=False 表示它不参与 dataclass 的相等性比较与 repr——附加信息
    # （由调用方按需挂载）变化时，不应该让两个「同一分块」被判定为不相等。
    meta: dict[str, Any] = field(default_factory=dict, compare=False)


# --------------------------------------------------------------------------
# front matter + document discovery
# --------------------------------------------------------------------------

# 中文说明：front matter 必须是文件最开头的 `---` 行到下一个 `---` 行。`\r?\n` 同时
# 容忍 Windows(CRLF) 与 Unix(LF) 换行；re.DOTALL 让 `.` 也能匹配换行，从而一次吃到
# 整个块；结尾的 `\r?\n?` 是可选的，允许没有空行紧跟正文。
_FRONT_MATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a document into (metadata, body). Tolerates a missing block."""
    # 中文说明：
    #   用途：把一份文档拆成 (front matter 字典, 正文)。
    #   参数：text —— 整份文档原文（读取时的编码处理见 load_document）。
    #   返回：(meta, body)。没有 front matter 时返回 ({}, 原文) 而不是报错——「格式是否
    #         合规」由 load_document 判断，这里只负责切分。
    #   实现说明：这是手写的扁平行解析，不是完整 YAML 实现——只认 `key: value` 一行一项，
    #   因此所有值都是字符串；空行、以 # 开头的行、不含冒号的行都会被跳过。
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        # 跳过空行、注释行（lstrip 后判断，所以缩进的 # 也算注释）、以及没有冒号的行：
        # front matter 里允许写给人看的说明文字，解析器不该因此失败。
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        # 用 partition(":") 只按**第一个**冒号切分，而不是 split(":")：值里本身可能含
        # 冒号，例如 source_url: https://example.com/x，用 split 会把 https 当成 key 的一部分。
        key, _, value = line.partition(":")
        value = value.strip()
        # 去掉成对的引号：给含冒号或 # 的值加引号是写 front matter 的常见习惯。
        # 条件要求首尾字符相同且是引号，长度 ≥ 2，因此单字符值不会被误删。
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        meta[key.strip()] = value
    # 用 match.end() 直接切正文：正则已经把结束分隔符和它后面的换行一起消费掉了，
    # 不必再去查找第二个 "---"（那样会把正文里出现的 `---`（Markdown 分隔线）也算进去）。
    return meta, text[match.end():]


def _sha256(text: str) -> str:
    # 中文说明：统一按 UTF-8 编码再哈希。这个函数是「内容是否变更」的唯一判据，所以
    # 所有调用点必须喂同样的输入（load_document 与 check_index_staleness 都是对整份
    # 原始文件文本，含 front matter）——口径不一致会让每次检查都报「全部文档已变更」。
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_documents(kbase_dir: Path | None = None) -> list[Path]:
    """Return every Markdown document in the knowledge base, sorted by name."""
    # 中文说明：
    #   用途：递归找出知识库目录下所有 .md 文件（英文 docstring 说 "sorted by name"，
    #         实际上 sorted 是对完整路径排序，效果是按路径字典序）。
    #   参数：kbase_dir —— 覆盖默认的 KBASE_DIR，测试与脚本用得到。
    #   返回：排序后的文件路径列表；目录不存在时抛 KnowledgeIndexError 而不是返回空列表，
    #         这样「路径配错了」会立刻暴露，而不是伪装成「知识库是空的」。
    #   为什么必须排序：让遍历顺序确定，索引构建与报告因此可复现。注意 chunk_id 里的
    #   ordinal 是**文档内**计数（每个文档从 0 重新开始），所以跨文档顺序变化不会改动
    #   chunk_id，只会影响插入顺序与统计输出。
    root = kbase_dir or KBASE_DIR
    if not root.is_dir():
        raise KnowledgeIndexError(f"知识库目录不存在：{root}")
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def load_document(path: Path) -> KnowledgeDocument:
    """Read one document, validating the front matter it must carry."""
    # 中文说明：
    #   用途：读取并校验一份文档，产出 KnowledgeDocument。
    #   参数：path —— .md 文件路径。
    #   返回：KnowledgeDocument；缺少必需元数据或 category 非法时抛
    #         KnowledgeDocumentError。
    # 用 utf-8-sig 读取：Windows 上的编辑器经常写入 UTF-8 BOM。若按普通 utf-8 解码，
    # BOM 会变成正文的第一个字符（\ufeff），于是 `^---` 与 `^#` 这类行首正则全部失配，
    # front matter 会「凭空消失」、第一级标题也识别不到。utf-8-sig 会在有 BOM 时剥掉它，
    # 没有 BOM 时行为与 utf-8 相同。
    raw = path.read_text(encoding="utf-8-sig")
    meta, body = parse_front_matter(raw)
    missing = [key for key in REQUIRED_META if not meta.get(key)]
    if missing:
        raise KnowledgeDocumentError(f"{path.name}: 缺少必需元数据 {missing}")
    category = meta["category"].strip()
    if category not in VALID_CATEGORIES:
        raise KnowledgeDocumentError(
            f"{path.name}: category={category!r} 不在允许集合 {sorted(VALID_CATEGORIES)} 内"
        )
    return KnowledgeDocument(
        path=path,
        doc_id=meta["doc_id"].strip(),
        title=meta["title"].strip(),
        category=category,
        # version/source_url/license/source_type/retrieved_at 都是可选元数据：给出
        # 兜底默认值（"n/a"/""/"unknown"），使引用时字段始终存在、不必到处判空。
        version=meta.get("version", "n/a").strip(),
        source_url=meta.get("source_url", "").strip(),
        license=meta.get("license", "").strip(),
        source_type=meta.get("source_type", "unknown").strip(),
        retrieved_at=meta.get("retrieved_at", "").strip(),
        body=body,
        # 哈希整份 raw（含 front matter）：元数据改动也算「文档变更」，理应触发重建索引。
        content_hash=_sha256(raw),
    )


# --------------------------------------------------------------------------
# heading-aware block splitting
# --------------------------------------------------------------------------

# MULTILINE is required: these patterns are applied to a whole document body,
# so `^` and `$` must match at line boundaries.
# 中文说明（逐个正则的用途与坑）：
#   _HEADING_RE   —— 匹配 1~6 级 ATX 标题，捕获 (井号串, 标题文本)。行首 `[ \t]*`
#       容忍缩进；行尾 `\r?$` 兼容 CRLF——不写的话，CRLF 文件里标题文本会多出一个 \r。
#   _TABLE_ROW_RE —— 「以 | 开头且以 | 结尾」的行，即 Markdown 表格行（表头、分隔行、
#       数据行都满足）。用它来判断某一行是否属于表格。
#   _FENCE_RE     —— 代码围栏（```）。
#       注意：_FENCE_RE 在本模块内没有被引用（代码围栏目前不参与分块判断）。此处仅如实
#       指出，未改动代码。
#   _BACKTICK_RE  —— 连续反引号，用于清理标题里的行内代码标记。
_HEADING_RE = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.*?)[ \t]*\r?$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*\r?$", re.MULTILINE)
_FENCE_RE = re.compile(r"^[ \t]*```", re.MULTILINE)
_BACKTICK_RE = re.compile(r"`+")


def _heading_title(raw: str) -> str:
    """Normalise a heading's text: drop trailing markup and inline backticks."""
    # 中文说明：把标题里的反引号全部删掉再 strip。标题会被拼成 heading_path 并写进分块
    # 正文，带反引号既影响可读性，也不利于按标题做精确匹配。（英文 docstring 说的是
    # "trailing markup"，实现上删除的是所有行内反引号。）
    return _BACKTICK_RE.sub("", raw).strip()


def _split_blocks(body: str) -> list[tuple[list[str], str]]:
    """Split a body into (heading_path, text) blocks.

    Correct by construction: every heading is located first, then the text
    between consecutive headings is sliced out and labelled with the heading
    chain in force at that point. Nothing is accumulated across headings, so a
    block can never be labelled with a section it does not belong to. A section
    that contains further headings is dropped, because those nested sections
    already contributed their own blocks.
    """
    # 中文说明：
    #   用途：按标题把正文切成 (标题路径, 文本) 块，标题路径是从一级标题累积下来的列表。
    #   参数：body —— 文档正文（front matter 已被 parse_front_matter 去掉）。
    #   返回：按文档顺序排列的块列表；标题之间没有正文时不会产出空块。
    #   为什么这样切（对应英文 docstring 的 "correct by construction"）：先把所有标题定位
    #   出来，再对「相邻两个标题之间的区间」贴上当时生效的标题栈。文本从不跨标题累积，
    #   所以一个块不可能被贴上不属于它的章节名——这是「先切片、再贴标签」相对于「边扫边
    #   累积」的关键优势。含子标题的父章节不会产出多余内容，因为它的正文只到下一个标题为止。
    #   注意：第一个标题之前的文字会以空路径（[]）产出一个块，即文档前言。
    matches = list(_HEADING_RE.finditer(body))
    stack: list[str] = []
    blocks: list[tuple[list[str], str]] = []

    def emit(path: list[str], text: str) -> None:
        # 中文说明：只产出非空块。空块会让标题路径白占一个检索位次，且没有内容可引用。
        text = text.strip()
        if text:
            blocks.append((path, text))

    if not matches:
        # 中文说明：整篇没有标题时，全文作为唯一块、路径为空；后面 chunk_document 会用文档
        # 标题兜底，因此引用信息仍然完整。
        emit([], body)
        return blocks

    # 中文说明：第一个标题之前的正文（前言/摘要）单独成块，路径为空列表。
    emit([], body[: matches[0].start()])

    for index, match in enumerate(matches):
        depth = len(match.group(1))
        title = _heading_title(match.group(2))
        # 中文说明：stack[: depth - 1] 先砍掉「比当前标题更深或同级」的层级再压入自己，
        # 于是栈内容始终等于当前位置的真实层级链（遇到 ## 保留前 1 项，遇到 ### 保留前 2 项…）。
        stack = stack[: depth - 1]
        stack.append(title)

        # 中文说明：块的结束位置取「下一个标题的起点」，最后一个块延伸到正文末尾。
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        # 中文说明：传 list(stack) 而不是 stack——stack 随后还会被就地修改，若传引用，
        # 先前已产出的块会跟着一起变（Python 里列表别名最典型的坑）。
        emit(list(stack), body[match.end():end])

    # 中文说明：再过一遍表格切分，把表格从它所在的块里独立出来（见 _split_tables）。
    return _split_tables(blocks)


def _split_tables(blocks: list[tuple[list[str], str]]) -> list[tuple[list[str], str]]:
    """Keep each Markdown table in one block, separate from surrounding prose.

    Consecutive table rows must stay together: splitting them makes the table's
    leading rows (the ones carrying ``| 值 | 数量 |``) unreachable, because the
    surrounding prose then forms its own chunk and the header row is orphaned.
    """
    # 中文说明：
    #   用途：把一个块里的 Markdown 表格与周围散文分开，且**整张表保持为一整块**。
    #   参数：blocks —— _split_blocks 的产物。
    #   返回：块列表，表格行连续成段，散文另成段。
    #   为什么表格必须原子化（英文 docstring 已给出实测理由）：表格拆开后，承载字段名的
    #   表头行会与数据行分离——检索命中数据行时读不出「这一列是什么」，而命中表头行时又
    #   没有任何数据。表格在本文档里正是「字段定义」这类问题的答案载体，拆散等于让它们失效。
    result: list[tuple[list[str], str]] = []
    for path, text in blocks:
        lines = text.splitlines()
        # 快速路径：块里一行表格都没有，原样保留，避免无谓的逐行处理。
        if not any(_TABLE_ROW_RE.match(line) for line in lines):
            result.append((path, text))
            continue

        segment: list[str] = []
        segment_is_table = False

        def flush_segment() -> None:
            # 中文说明：nonlocal 是必需的——下面会整体重新赋值 segment，若不声明，
            # 这里只会创建一个同名局部变量，外层列表永远不会被清空（典型的闭包踩坑）。
            nonlocal segment
            joined = "\n".join(segment).strip()
            if joined:
                result.append((path, joined))
            segment = []

        for line in lines:
            is_table = bool(_TABLE_ROW_RE.match(line))
            # Start a new segment only when the kind changes AND the previous
            # segment was prose; a blank line inside a table is not a boundary.
            # 中文说明（上面英文注释的展开）：只有在「类型发生切换」且「上一段是散文」时才
            # 断段。`not (segment_is_table and not line.strip())` 这一项专门放行「表格中间
            # 的空行」——它是排版留白，不是表格的结束，否则一张表会被空行切成两截。
            if segment and segment_is_table != is_table and not (segment_is_table and not line.strip()):
                flush_segment()
                segment_is_table = is_table
            elif not segment:
                # 中文说明：新段的类型由它的第一行决定。
                segment_is_table = is_table
            segment.append(line)
        flush_segment()
    return result


def _split_oversized(text: str, limit: int) -> list[str]:
    """Deterministic windowed split with overlap, used only for huge blocks."""
    # 中文说明：
    #   用途：把超长文本按固定窗口切成多片，相邻窗口重叠 OVERLAP_CHARS 个字符。
    #   参数：text —— 待切文本；limit —— 每片的最大字符数。
    #   返回：按原顺序排列的片段列表；text 比 limit 短时返回单元素列表。
    #   为什么叫 deterministic：切点只由字符数决定，不依赖句子边界或随机性，因此同一份
    #   输入永远得到同一批分块——否则每次重建索引都会产生新的 chunk_id，索引与评测无法比对。
    #   代价是按字符切会切断句子，所以才需要重叠：让被切在边界上的那句话至少在一侧完整。
    #   `max(1, ...)` 是防御性写法：若 limit <= OVERLAP_CHARS，步长会变成 0 或负数，
    #   range 会报错或退化成死循环；下限 1 保证一定会向前推进。
    step = max(1, limit - OVERLAP_CHARS)
    return [text[start:start + limit] for start in range(0, len(text), step)]

def _merge_blocks(blocks: list[tuple[list[str], str]]) -> list[tuple[str | None, str]]:
    """Pack blocks into chunks, one entity per chunk.

    The rule is intentionally narrow so it cannot glue unrelated content:

    * a block is merged only when the buffer represents its **direct parent**
      (``## 英雄`` then ``### 技能``) or it shares the buffer's exact path;
    * anything else — a sibling champion, an unrelated section, a shallower or
      deeper branch — flushes the buffer first.

    The buffer always holds the path of the block that opened it, and that label
    is passed to ``emit`` explicitly, so a chunk can never be named after the
    entity that *follows* it.
    """
    # 中文说明：
    #   用途：把 (标题路径, 文本) 块打包成「一个实体一个分块」，并顺手做尺寸控制。
    #   参数：blocks —— _split_blocks/_split_tables 的产物，顺序与文档顺序一致。
    #   返回：[(标签或 None, 正文)]，标签是 " > " 连接后的标题路径（空路径时为 None）。
    #   merges() 的英文 docstring 说明了合并规则为什么必须这么窄（只合并路径完全相同的
    #   兄弟块），中文补充一点实践后果：它是「同一实体（例如同一个英雄、同一张表）的内容
    #   尽量待在一个分块里」与「不同实体的内容绝不混进同一个分块」这两条要求的最小实现。
    packed: list[tuple[str | None, str]] = []
    buffer: list[str] = []
    buffer_path: list[str] = []
    buffer_size = 0

    def emit(label: list[str]) -> None:
        # 中文说明：用**传入的 label** 落盘，而不是读 buffer_path。这是刻意的：调用方在
        # 刷新 buffer 之后马上会把 buffer_path 换成新块的路径，如果这里去读实例状态，
        # 就可能把「后一个实体」的标题贴到「前一个实体」的内容上（英文 docstring 末段
        # 强调的 "a chunk can never be named after the entity that follows it"）。
        nonlocal buffer, buffer_size
        text = "\n\n".join(buffer).strip()
        if text:
            packed.append((" > ".join(label) if label else None, text))
        buffer = []
        buffer_size = 0

    def merges(buffer_path: list[str], block_path: list[str]) -> bool:
        """Whether `block_path` belongs inside the chunk opened at `buffer_path`.

        The block must be a *sibling* of the buffer's path (same depth, same
        path) — sections under one entity pack together. A block nested under
        the buffer is deliberately NOT merged: a document-level chunk such as
        ``# 数据字段定义与取值域`` must not swallow every ``##`` section beneath
        it, or the whole document collapses into one chunk and its sections
        become unreachable.
        """
        # 中文说明：判定条件只有「路径完全相等」这一条。最需要记住的反例写在英文
        # docstring 里：文档级标题（# 数据字段定义与取值域）绝不能吞掉它下面的每个 ##
        # 小节——否则整篇文档塌成一个分块，各个小节再也无法被单独检索到。
        return buffer_path == block_path

    for path, block in blocks:
        path_text = " > ".join(path)

        if len(block) > MAX_CHUNK_CHARS:
            # 中文说明：超长块不参与合并，先刷掉缓冲区，再直接滑动切分后落盘（每片都是
            # 独立分块，标签相同）。这样一张巨大的表格或超长小节不会把整篇文档挤成一个块。
            emit(buffer_path)
            buffer_path = []
            # Leave room for the heading-path prefix added by chunk_document,
            # so no emitted chunk can bust the advertised maximum.
            # 中文说明：reserve 预留给 chunk_document 拼接的「文档标题 > 标题路径」前缀
            # （+64 是分隔符与余量的经验值），否则「切分时的上限」加上前缀后仍会越界。
            reserve = len(path_text) + 64
            for piece in _split_oversized(block, max(MIN_CHUNK_CHARS, MAX_CHUNK_CHARS - reserve)):
                packed.append((path_text or None, piece))
            continue

        if not buffer:
            # 中文说明：空缓冲区由当前块「开张」，并以它的路径作为后续块的归属判据。
            buffer_path = list(path)
        elif buffer_size + len(block) + 2 > MAX_CHUNK_CHARS or not merges(buffer_path, path):
            # 中文说明：两种刷新条件——装不下了（+2 是块之间 join("\n\n") 的两个换行），
            # 或者这个块不属于当前分块的标题路径（换实体了）。
            emit(buffer_path)
            buffer_path = list(path)

        buffer.append(block)
        buffer_size += len(block) + 2

    emit(buffer_path)

    # Second pass: fold a runt into its neighbour so no chunk is trivially small.
    # Merging only across an identical heading path cannot re-cross an entity
    # boundary, so two champions still never share a chunk.
    # 中文说明（第二遍：把小碎块折进邻居）：只向**前一个**分块合并，且要求两者标题路径
    # 完全相同、合并后不超上限。限定「路径相同」这一条保证了不会越过实体边界——这也正是
    # 它不会把两个英雄拼成一个分块的原因（英文注释同义）。
    merged: list[tuple[str | None, str]] = []
    for heading_path, text in packed:
        if merged and len(text) < MIN_CHUNK_CHARS:
            prev_path, prev_text = merged[-1]
            if (
                prev_path == heading_path
                and len(prev_text) + len(text) + 2 <= MAX_CHUNK_CHARS
            ):
                merged[-1] = (prev_path, f"{prev_text}\n\n{text}")
                continue
        merged.append((heading_path, text))
    return merged


def chunk_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    """Turn one document into retrieval chunks with metadata attached.

    The body keeps its own leading H1: dropping it would orphan the first
    section (its heading would look like a new top level and every following
    sibling would appear nested under it). Instead the H1 is recognised here and
    absorbed into the title prefix rather than repeated.
    """
    # 中文说明：
    #   用途：把一份文档变成带完整元数据的分块列表——这是「检索最小单位」的生产点。
    #   参数：document —— load_document 产出的文档对象。
    #   返回：KnowledgeChunk 列表，顺序即文档顺序；同一块被切成多片时，各片是独立分块。
    #   为什么要给正文加「标题前缀」：embedding 只看正文时，「7.17.1」这种小标题下的
    #   一段话可能完全看不出属于哪份文档的哪一节；把「文档 > 节 > 小节」拼在正文前面，
    #   片段本身就自带上下文，检索与引用都更准（这也是本分块策略相对固定窗口的核心优势）。
    body = document.body
    chunks: list[KnowledgeChunk] = []
    ordinal = 0
    for heading_path, text in _merge_blocks(_split_blocks(body)):
        if heading_path:
            parts = [part.strip() for part in heading_path.split(">")]
            # 中文说明：正文自带的一级标题（H1）文本通常与 front matter 的 title 相同，
            # 前缀里会再加一次 title，所以这里把重复的那一段去掉，避免出现
            # 「标题 > 标题 > 小节」。用等值比较而不是无条件去掉第一段，是因为有些文档
            # 正文并没有 H1（或与 title 不同），那时不该误删一个真实的章节名。
            if parts and parts[0] == document.title:
                parts = parts[1:]
            prefix = document.title if not parts else f"{document.title} > {' > '.join(parts)}"
        else:
            # 中文说明：空路径的块（文档前言、整篇无标题的文档）只挂文档标题，
            # 保证任何分块都能回答「你来自哪份文档」。
            prefix = document.title
        payload = f"{prefix}\n{text}"
        # Final guard: the advertised maximum counts the prefix too, so a
        # composed chunk can never exceed it.
        # 中文说明：最后一道尺寸闸门——MAX_CHUNK_CHARS 是对**最终 payload**（含前缀）的
        # 承诺，所以这里必须在拼好前缀之后再判一次；超了就把正文重新按前缀预留量切分。
        payloads = (
            [payload]
            if len(payload) <= MAX_CHUNK_CHARS
            else [
                f"{prefix}\n{piece}"
                for piece in _split_oversized(text, max(MIN_CHUNK_CHARS, MAX_CHUNK_CHARS - len(prefix) - 1))
            ]
        )
        for payload in payloads:
            # 中文说明（chunk_id 的构成与「位移」问题，本项目最值得记住的约束之一）：
            #   chunk_id = f"{doc_id}#{ordinal:04d}-{digest}"，三段各有职责：
            #     * doc_id   —— 命名空间，跨文档不会冲突；
            #     * ordinal  —— 分块在本**文档内**的位置，从 0 起递增，左侧补零到 4 位，
            #                   于是字符串排序与数值顺序一致，日志和数据库里都好读；
            #     * digest   —— 对 doc_id + ordinal + payload 取 SHA-256 的前 16 个十六进制
            #                   字符，用来区分「序号相同但内容变了」的分块。
            #   **位移**：只要在文档中间插入或删除一个分块，其后所有分块的 ordinal 都会变化，
            #   chunk_id 随之全部改变。本项目因此不受影响，因为索引是整库重建的
            #   （build_knowledge_index 先 DELETE 再 INSERT），不存在跨版本的悬挂引用；
            #   但若将来做增量更新，或把 chunk_id 当作长期稳定标识（外部引用、缓存键、
            #   评测集里写死），就必须先解决这个位移问题。
            #   哈希输入用 NUL（\0）分隔三段：doc_id 与正文里都不会出现 NUL，因此不同的
            #   (doc_id, ordinal, payload) 组合不可能拼成同一个字符串（避免拼接歧义）。
            #   只取前 16 个十六进制字符（64 位）是「足够区分 + id 更短」的取舍。
            digest = hashlib.sha256(
                f"{document.doc_id}\0{ordinal}\0{payload}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.doc_id}#{ordinal:04d}-{digest}",
                    doc_id=document.doc_id,
                    # 中文说明：存相对仓库根的 POSIX 风格路径。相对路径让索引与工作目录无关，
                    # as_posix() 则保证 Windows 的反斜杠不会写进数据库与最终引用文本。
                    source=document.path.relative_to(PROJECT_ROOT).as_posix(),
                    # 中文说明：heading 是「给人看的标题」，无标题时退化为文档标题，永不为空；
                    # heading_path 是严格的路径字段，无标题就是空串。两者语义不同，别互换。
                    heading=heading_path or document.title,
                    heading_path=heading_path or "",
                    category=document.category,
                    version=document.version,
                    source_url=document.source_url,
                    license=document.license,
                    text=payload,
                    ordinal=ordinal,
                )
            )
            # 中文说明：自增放在 payloads 循环内部——一个块被 _split_oversized 切成多片时，
            # 每片各占一个 ordinal（否则同文档内会出现重复序号，digest 也无法兜住）。
            ordinal += 1
    return chunks


def load_knowledge_chunks(kbase_dir: Path | None = None) -> list[KnowledgeChunk]:
    """Parse every knowledge-base document into chunks."""
    # 中文说明：每次调用都**重新解析磁盘**，不读索引、不调用任何 API，因此可在测试与校验
    # 脚本里随意使用（tests/test_knowledge.py、scripts/verify_kbase.py、
    # scripts/archive/refresh_index_hashes.py 都依赖它）。
    # 注意：build_knowledge_index 内部重复了同样的遍历与分块（它还需要 documents 列表来写
    # index_sources 表），两条路径的分块结果必须一致——分块的唯一事实来源是 chunk_document，
    # 所以改分块逻辑时两边会同时改变。
    chunks: list[KnowledgeChunk] = []
    for path in discover_documents(kbase_dir):
        chunks.extend(chunk_document(load_document(path)))
    return chunks


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

# 中文说明（三张表的分工，SQL 里不便写注释，所以放在这里）：
#   knowledge_chunks —— 检索主表，一行 = 一个分块。embedding 以 JSON 文本存（SQLite 没有
#       原生向量类型），读取时再 json.loads 还原成 float 列表；这个「存文本、读时解析」的
#       选择意味着向量维度错误只能在读取时发现，因此 search_knowledge 里保留了维度校验。
#   index_meta       —— 键值对，记录「这份索引是怎么来的」：分块器版本、embedding 模型、
#       向量维度、权重与阈值、文档/分块数量。排查「索引过期 / 口径不一致」时先看它。
#   index_sources    —— 每个源文件的哈希、字节数、文档 id、分类与该文档产出的分块数；
#       check_index_staleness 靠它比对磁盘现状。
#   索引建在 category 与 doc_id 上（按分类过滤、按文档排查用得上）；数据量不大时是否真的
#   走索引由 SQLite 自行决定。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    source       TEXT NOT NULL,
    heading      TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    category     TEXT NOT NULL,
    version      TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    license      TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    content      TEXT NOT NULL,
    embedding    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_sources (
    source       TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    byte_size    INTEGER NOT NULL,
    doc_id       TEXT NOT NULL,
    category     TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    retrieved_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_category ON knowledge_chunks(category);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON knowledge_chunks(doc_id);
"""

# 中文说明：FTS5 虚表，用于词法检索实验。chunk_id 标了 UNINDEXED（只存不索引，它是外键，
# 不需要分词），source / heading_path / content 三列参与匹配。构建它只是为了让
# scripts/compare_strategies.py 的对比实验可以复现——默认打分路径中 BM25 权重为 0
# （见 VECTOR_WEIGHT / BM25_WEIGHT 处的说明）。
# 另外：并非所有 SQLite 构建都启用了 FTS5，所以建表失败会被 _ensure_schema 容忍
# （降级记录为 "fts5 unavailable"），检索照常工作。
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    source,
    heading_path,
    content
);
"""

# Latin words/numbers only: FTS5 has no CJK tokenizer, so Chinese matching is
# left to the bigram similarity term.
# 中文说明：只抽取拉丁字母/数字/下划线。直接后果是：**纯中文查询在这里抽不出任何词项**，
# _fts_query 返回空串、_bm25_scores 返回 {}——这是常态而不是异常（本项目绝大多数查询是中文）。
_TERM_RE = re.compile(r"[a-z0-9_]+")


def _fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from the query's Latin terms."""
    # 中文说明：
    #   用途：把查询里的拉丁词项拼成一个 FTS5 MATCH 表达式。
    #   参数：query —— 用户查询或改写后的检索查询。
    #   返回：形如 "a" OR "b" 的表达式；没有拉丁词项时返回空串，调用方据此跳过 BM25。
    # 用 OR 连接而不是依赖 FTS5 默认的 AND 语义：任一词项命中即算候选，优先保证召回。
    # 给每个词项加双引号，是让它们被当作字面量而不是 FTS5 的语法记号（AND / OR / NEAR / *），
    # 否则查询里出现这些词会让 MATCH 表达式直接变成语法错误。词项本身已被 _TERM_RE 限制为
    # [a-z0-9_]，不可能带进引号或操作符。
    # dict.fromkeys 用于去重**并保持出现顺序**（字典有序），使表达式不会因重复词而变长。
    terms = _TERM_RE.findall(query.casefold())
    return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))

# 中文说明：与 _SCHEMA 里 knowledge_chunks 的列集合一一对应，用作「索引是否属于当前版本」
# 的判据（比较用的是集合相等，因此不依赖列顺序）。
_REQUIRED_CHUNK_COLUMNS = {
    "chunk_id", "doc_id", "source", "heading", "heading_path", "category",
    "version", "source_url", "license", "ordinal", "content", "embedding",
}


def _existing_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    # 中文说明：PRAGMA table_info 每行描述一列，其中下标 1 是列名，所以取 row[1]。
    # 表不存在时 PRAGMA 一般返回空结果，返回空集合正好表达「还没有这张表」；这里的
    # except 是防御性兜底（数据库文件损坏或不是 SQLite 文件时可能抛 sqlite3.Error）。
    try:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _ensure_schema(connection: sqlite3.Connection) -> str:
    """Create the schema, migrating an incompatible legacy index if needed."""
    # 中文说明：
    #   用途：保证连接上的 schema 可用，必要时迁移旧索引，并返回一句人类可读的处理说明。
    #   参数：connection —— 已打开的 SQLite 连接。
    #   返回：note 字符串，最终写进 index_meta.schema_note 并由构建报告打印出来，
    #         例如 "created; fts5 indexed" 或 "migrated from legacy schema [...]; fts5 unavailable"。
    columns = _existing_columns(connection, "knowledge_chunks")
    note = "created"
    # 中文说明：只有「表已存在（columns 非空）且列集合与要求不同」才判定为旧版索引并 DROP。
    # 注意这里选择的是「整表 DROP 重建」而不是逐列 ALTER：该表内容属于派生数据，随时可以
    # 从 kbase/ 重新生成，所以重建的代价只是再跑一次索引。
    if columns and columns != _REQUIRED_CHUNK_COLUMNS:
        connection.execute("DROP TABLE IF EXISTS knowledge_chunks")
        note = f"migrated from legacy schema {sorted(columns)}"
    # executescript 一次执行多条语句（建表 + 建索引）；CREATE ... IF NOT EXISTS 使它对
    # 「已经建好的库」也是幂等的。
    connection.executescript(_SCHEMA)
    try:
        connection.executescript(_FTS_SCHEMA)
        note += "; fts5 indexed"
    except sqlite3.Error:
        # FTS5 is optional: retrieval falls back to vector + bigram scoring.
        # 中文说明（含一处提醒）：FTS5 不可用时只记录状态、不中断构建。
        # 注意：这条英文注释写的是「回退到 vector + bigram 打分」，但当前权重里
        # BM25_WEIGHT 与 LEXICAL_WEIGHT 都是 0，因此实际只剩向量打分；注释保留原文不动。
        note += "; fts5 unavailable"
    return note


def build_knowledge_index(*, dry_run: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Embed the knowledge base and replace the local SQLite retrieval index.

    With ``dry_run=True`` the chunking runs and is reported without any
    Embedding API call — the cheap way to inspect a chunking change.
    """
    # 中文说明：
    #   用途：把整个知识库重新分块、向量化，并**整体替换**本地 SQLite 索引。
    #   参数：dry_run —— True 时只做分块与统计、不调用任何 API（英文 docstring 已说明，
    #                     中文补充：这是检查「改了分块参数之后结果变成什么样」的廉价方式，
    #                     不消耗 Embedding 额度）。
    #         verbose —— 是否打印进度；脚本默认开着，测试或程序化调用可以关掉。
    #   返回：report 字典：文档/分块数量、分块字符的最值与均值、小碎块数量、索引相对路径、
    #         schema 处理说明。调用方（scripts/index_knowledge.py）据此打印与判断。
    # 先把文档全部读进来：任何一份文档的 front matter 不合规都会在这里抛
    # KnowledgeDocumentError，即「要么全部合格、要么一份索引都不产出」，不会留下半成品索引。
    documents = [load_document(path) for path in discover_documents()]
    if not documents:
        raise KnowledgeIndexError("知识库中没有找到任何 .md 文档。")

    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    if not chunks:
        raise KnowledgeIndexError("知识库文档没有产出任何分块。")

    # 中文说明：这几个尺寸指标是分块质量的体检项——小碎块数量突然上升、或最大值越过
    # MAX_CHUNK_CHARS，通常说明分块器被改坏了，而不需要真去调 API 检索才能发现。
    sizes = [len(chunk.text) for chunk in chunks]
    report: dict[str, Any] = {
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_chars_min": min(sizes),
        "chunk_chars_mean": round(sum(sizes) / len(sizes), 1),
        "chunk_chars_max": max(sizes),
        "small_chunks_under_min": sum(1 for size in sizes if size < MIN_CHUNK_CHARS),
        "dry_run": dry_run,
    }

    if dry_run:
        if verbose:
            print(f"DRY RUN — {report['documents']} 份文档，{report['chunks']} 个分块，未调用 API")
            print(
                f"  分块字符 最小/平均/最大: {report['chunk_chars_min']} / "
                f"{report['chunk_chars_mean']} / {report['chunk_chars_max']}"
            )
            print(f"  小于 {MIN_CHUNK_CHARS} 字符的分块: {report['small_chunks_under_min']}")
            # 中文说明：按分类统计分块数并按数量降序打印，用于确认 offtopic/noise 这类
            # 干扰文档确实在库里（它们不是知识，但必须存在才能做阈值校准）。
            by_category: dict[str, int] = {}
            for chunk in chunks:
                by_category[chunk.category] = by_category.get(chunk.category, 0) + 1
            for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
                print(f"  {category:10} {count:>5}")
        return report

    # 中文说明：embedding 客户端只在这里构建一次并复用（每次调用都新建会重复读配置、
    # 重复建立 HTTP 客户端）。真正的分批在 embed_texts 里完成。
    embeddings = build_embeddings()
    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(texts)
    if len(vectors) != len(chunks):
        # 中文说明：数量对不上必须立刻失败。若放任下去，下面的 zip 会把后半部分分块与错误
        # 的向量配对，索引会在无人察觉的情况下整体错乱。
        raise KnowledgeAPIError(
            f"Embedding 返回数量（{len(vectors)}）与分块数量（{len(chunks)}）不一致。"
        )

    # 中文说明：首次运行时 reports/ 目录可能还不存在，先建出来。
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 注意：`with sqlite3.connect(...)` 只负责事务提交/回滚，**不会关闭连接**
    # （conversation.py 里有详细说明）。这里是一次性脚本，进程退出即释放，因此不额外处理。
    with sqlite3.connect(INDEX_PATH) as connection:
        note = _ensure_schema(connection)
        # 中文说明：重建语义 = 先清空三张表再全量写入，而不是增量更新。索引是派生数据，
        # 全量重建不会有碎片和「半更新」状态；代价是每次都要重新 embedding 全部文本。
        # 这也是 chunk_id 的「ordinal 位移」问题在本项目里不成问题的原因：不存在跨版本引用。
        connection.execute("DELETE FROM knowledge_chunks")
        connection.execute("DELETE FROM index_sources")
        connection.execute("DELETE FROM index_meta")
        try:
            connection.execute("DELETE FROM chunk_fts")
        except sqlite3.Error:
            # 中文说明：FTS5 不可用时这条 DELETE 会失败，忽略即可（与 _ensure_schema 的
            # 容忍策略一致）。
            pass
        # 中文说明：用 executemany 批量插入（比逐条 execute 快一个量级）。
        # zip(..., strict=True) 是第二道长度校验：两列表长度不等时它会抛 ValueError，
        # 而不是像默认行为那样悄悄截断到较短的一方（strict 参数需要 Python 3.10+）。
        connection.executemany(
            """INSERT INTO knowledge_chunks
               (chunk_id, doc_id, source, heading, heading_path, category, version,
                source_url, license, ordinal, content, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    chunk.chunk_id, chunk.doc_id, chunk.source, chunk.heading,
                    chunk.heading_path, chunk.category, chunk.version,
                    chunk.source_url, chunk.license, chunk.ordinal, chunk.text,
                    # 中文说明：向量以 JSON 数组文本存进 TEXT 列——SQLite 没有向量/数组类型，
                    # 读取时由 search_knowledge 用 json.loads 还原成 float 列表。
                    json.dumps(vector),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

        try:
            # 中文说明：写入 FTS5 表的 content 是「标题 + 正文」拼接，于是按标题里的中文/英文
            # 词做词法匹配时也有机会命中（heading_path 另有一列）。整段包在 try/except 里，
            # FTS5 不可用就跳过，与 _ensure_schema 的容忍策略保持一致。
            connection.executemany(
                "INSERT INTO chunk_fts (chunk_id, source, heading_path, content) VALUES (?, ?, ?, ?)",
                [
                    (
                        chunk.chunk_id, chunk.source, chunk.heading_path,
                        f"{chunk.heading} {chunk.text}",
                    )
                    for chunk in chunks
                ],
            )
        except sqlite3.Error:
            pass

        # 中文说明：先按文档统计分块数。这个数字写进 index_sources.chunk_count，能让
        # 「某份文档一个分块都没产出」这类分块异常在索引里直接可见。
        per_document: dict[str, int] = {}
        for chunk in chunks:
            per_document[chunk.doc_id] = per_document.get(chunk.doc_id, 0) + 1
        connection.executemany(
            """INSERT INTO index_sources
               (source, content_hash, byte_size, doc_id, category, chunk_count, retrieved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    document.path.relative_to(PROJECT_ROOT).as_posix(),
                    document.content_hash,
                    # 中文说明：这一列是**正文**（不含 front matter）的 UTF-8 字节数，单位是
                    # 字节而非字符；而 content_hash 是对整份文件（含 front matter）取的哈希。
                    # 两者口径不同是故意的：byte_size 只是给人看的规模指标，hash 才是过期判据。
                    len(document.body.encode("utf-8")),
                    document.doc_id,
                    document.category,
                    per_document.get(document.doc_id, 0),
                    document.retrieved_at,
                )
                for document in documents
            ],
        )
        # 中文说明：把「这份索引是用什么口径建的」写进元数据：换 embedding 模型、改阈值或
        # 改分块器之后，旧索引能一眼看出来，而不是靠人去猜。
        meta = {
            "chunker_version": CHUNKER_VERSION,
            "embedding_model": getattr(embeddings, "model", "unknown"),
            "embedding_dimensions": str(EMBEDDING_DIMENSIONS),
            "max_chunk_chars": str(MAX_CHUNK_CHARS),
            "min_chunk_chars": str(MIN_CHUNK_CHARS),
            "vector_weight": str(VECTOR_WEIGHT),
            "lexical_weight": str(LEXICAL_WEIGHT),
            "min_retrieval_score": str(MIN_RETRIEVAL_SCORE),
            "document_count": str(len(documents)),
            "chunk_count": str(len(chunks)),
            "schema_note": note,
        }
        # 中文说明：sorted(meta.items()) 让写入顺序确定——两次构建的 index_meta 行顺序一致，
        # 比较两次索引的差异时不会因为字典顺序而产生噪声。
        connection.executemany(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)", sorted(meta.items())
        )

    # 中文说明：报告里给出相对仓库根的路径，日志更短，而且与「从哪个目录启动」无关。
    report["index_path"] = INDEX_PATH.relative_to(PROJECT_ROOT).as_posix()
    report["schema_note"] = note
    if verbose:
        print(
            f"索引已重建：{report['chunks']} 个分块 / {report['documents']} 份文档"
            f" -> {report['index_path']}"
        )
        print(f"  schema: {note}")
    return report


def embed_texts(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """Embed texts in API-sized batches.

    DashScope rejects a request with more than 20 inputs, so batching lives here
    rather than at each call site.
    """
    # 中文说明：
    #   用途：把一批文本向量化，内部按接口上限自动分批。
    #   参数：texts —— 待向量化文本；batch_size —— 每批条数，默认 EMBEDDING_BATCH_SIZE(20)。
    #   返回：向量列表，**顺序与输入一致**（用 extend 顺序拼接；调用方的 zip 依赖这个顺序，
    #         不能在这里打乱或排序）。空输入返回空列表。
    #   把「必须分批」这条平台约束收在一个地方，调用方就不必知道 20 这个数字，
    #   也不必担心换模型时上限变化。
    embeddings = build_embeddings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embeddings.embed_documents(texts[start:start + batch_size]))
    return vectors


def _index_meta(connection: sqlite3.Connection) -> dict[str, str]:
    # 中文说明：读取索引的自述元数据。表不存在或不可读时返回空字典而不是抛异常——
    # check_index_staleness 需要能在「索引残缺/是旧版」的情况下仍然产出一份报告，
    # 让调用方据此提示用户重建，而不是直接崩掉。
    try:
        return {row[0]: row[1] for row in connection.execute("SELECT key, value FROM index_meta")}
    except sqlite3.Error:
        return {}


def check_index_staleness() -> dict[str, Any]:
    """Compare indexed source hashes against the files currently on disk.

    Returns a report so callers can warn instead of silently answering from an
    out-of-date index.
    """
    # 中文说明：
    #   用途：把索引里记录的源文件哈希与磁盘上现在的文件逐一比对，判断索引是否过期。
    #   参数：无（固定使用 INDEX_PATH 与 KBASE_DIR）。
    #   返回：报告字典，字段含义：
    #         index_exists       —— 索引文件是否存在；
    #         stale              —— 综合结论，任一类差异都会置为 True；
    #         missing_documents  —— 索引里有、磁盘上已不存在（被删除或改名）；
    #         new_documents      —— 磁盘上有、索引里没有（新增的 .md）；
    #         changed_documents  —— 两边都有但内容哈希不同；
    #         meta               —— 索引的自述元数据（见 _index_meta）；
    #         reason             —— 给人看的中文结论。
    #   为什么返回报告而不是抛异常或返回 bool：调用方（apps/api.py、scripts/index_knowledge.py、
    #   verify_dataset_metadata.py 等）需要据此**提醒**用户重建索引，而不是直接拒绝服务；
    #   把判断留给调用方，口径才一致。
    result: dict[str, Any] = {
        "index_exists": INDEX_PATH.is_file(),
        "stale": False,
        "missing_documents": [],
        "changed_documents": [],
        "new_documents": [],
        "meta": {},
    }
    # 中文说明：三级早退——索引文件不存在 / 数据库不可读 / 没有来源记录，都直接判为过期并
    # 给出 reason。「根本没有索引」本身就是一种过期状态，不需要再往下比对。
    if not result["index_exists"]:
        result["stale"] = True
        result["reason"] = "索引文件不存在"
        return result

    try:
        with sqlite3.connect(INDEX_PATH) as connection:
            indexed = {
                row[0]: row[1]
                for row in connection.execute("SELECT source, content_hash FROM index_sources")
            }
            result["meta"] = _index_meta(connection)
    except sqlite3.Error as exc:
        result["stale"] = True
        result["reason"] = f"索引不可读：{exc}"
        return result

    if not indexed:
        result["stale"] = True
        result["reason"] = "索引没有来源记录（旧版索引，需要重建）"
        return result

    # 中文说明：读取口径必须与 load_document 完全一致（utf-8-sig 读取整份原文 → sha256），
    # 否则这里会把每一份文档都判成 changed，过期检查就彻底失去意义。
    on_disk: dict[str, str] = {}
    for path in discover_documents():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        on_disk[relative] = _sha256(path.read_text(encoding="utf-8-sig"))

    # 中文说明：三个集合运算构成完整的差异视图。注意 missing_documents 是「索引视角」的命名，
    # 即「索引里有、磁盘上没了」；rglob 会递归发现新文件，所以新增文档也能被识别。
    result["missing_documents"] = sorted(set(indexed) - set(on_disk))
    result["new_documents"] = sorted(set(on_disk) - set(indexed))
    result["changed_documents"] = sorted(
        source for source in set(indexed) & set(on_disk) if indexed[source] != on_disk[source]
    )
    result["stale"] = bool(
        result["missing_documents"] or result["new_documents"] or result["changed_documents"]
    )
    result["reason"] = "索引与知识库一致" if not result["stale"] else "知识库已变更，索引需要重建"
    return result


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def _lexical_similarity(query: str, text: str) -> float:
    """Reward exact Chinese phrase/word overlap alongside vector similarity."""
    # 中文说明：
    #   用途：一个轻量的「字面重合度」近似——中文没有空格分词，所以用「整段中文 + 所有相邻
    #         二字组合（bigram）」构成特征集合。
    #   参数：query —— 查询；text —— 候选分块正文。
    #   返回：0~1 的比例，含义是「查询特征有多大比例在文本里出现过」；查询抽不出特征时返回 0.0。
    #   为什么整段中文也要收进集合（result = set(tokens)）：这样「连续短语整体出现」与
    #   「零散二字组合出现」都会被计入，前者的重合度天然更高。
    #   注意：这是集合运算，不考虑词频、位置与文本长度，也不是 BM25 那种长度归一化打分。
    #   当前 LEXICAL_WEIGHT = 0，所以它虽然会被算出来并写进结果（便于脚本观察），
    #   但并不影响最终的 similarity。
    def features(value: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value.casefold())
        result: set[str] = set(tokens)
        for token in tokens:
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                # 中文串切成所有相邻二字组合。range(len(token) - 1) 使单字 token 切不出
                # 任何 bigram，于是它就以整字形式参与匹配（这也是单字查询仍能工作的原因）。
                result.update(token[i:i + 2] for i in range(len(token) - 1))
        return result

    query_features = features(query)
    if not query_features:
        return 0.0
    # 分母是查询特征数，所以这是「召回覆盖率」而不是对称相似度：文本越长，覆盖查询特征的
    # 机会越多，这个指标就越偏向长文本。
    return len(query_features & features(text)) / len(query_features)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    # 中文说明：两个向量的余弦相似度。
    #   zip(..., strict=True) 要求两向量等长，长度不一致会抛 ValueError——这是把「索引向量
    #   与查询向量维度不匹配」尽早暴露出来的第一道防线（search_knowledge 里另有一次显式校验，
    #   用于给出更友好的中文错误信息）。
    #   任一模长为 0 时返回 0.0，而不是触发除零错误：零向量没有方向，定义上相似度就是 0。
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _bm25_scores(connection: sqlite3.Connection, query: str) -> dict[str, float]:
    """Return {chunk_id: normalised BM25 score} for the query's Latin terms.

    Normalisation maps the best match to 1.0 and worse matches linearly below,
    which keeps the value comparable with the cosine and bigram terms.
    """
    # 中文说明：
    #   用途：对查询里的拉丁词项跑一次 FTS5 的 BM25 查询。
    #   参数：connection —— 由调用方传入，以便复用 search_knowledge 已打开的连接
    #                     （避免为一次打分再开一次数据库）；query —— 查询文本。
    #   返回：{chunk_id: 归一化分数}，最高分恒为 1.0；没有拉丁词项或 FTS5 不可用时返回 {}。
    expression = _fts_query(query)
    if not expression:
        # 中文说明：纯中文查询会走到这里（_TERM_RE 只认拉丁词项），返回空字典是常态。
        return {}
    try:
        # 中文说明：FTS5 的 bm25() **越小越相关**，所以取负号把它翻成「越大越好」，与余弦、
        # bigram 两项同向。后面的 1.0/2.0/1.0 是 FTS5 的列权重参数。
        # LIMIT 50 把 BM25 的候选限定在一小批内：既让归一化有明确范围，也避免为全库排序。
        rows = connection.execute(
            "SELECT chunk_id, -bm25(chunk_fts, 1.0, 2.0, 1.0) AS score "
            "FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY score DESC LIMIT 50",
            (expression,),
        ).fetchall()
    except sqlite3.Error:
        # 中文说明：FTS5 虚表不存在或 MATCH 表达式不被支持时静默降级。因为 BM25 目前的权重
        # 是 0，这个降级对最终结果没有影响。
        return {}
    if not rows:
        return {}
    best = max(score for _, score in rows)
    if best <= 0:
        # 中文说明：最高分都不为正，等价于「没有有效词项命中」。宁可返回空字典，也不要把
        # 纯噪声归一化成 1.0——那样噪声会被当成满分匹配。
        return {}
    # 中文说明：除以本批最高分做归一化，让量纲与余弦、bigram 可比（都是 0~1、越大越好）。
    # 注意这只是同一批候选内部的相对值，跨查询不可比。
    return {chunk_id: score / best for chunk_id, score in rows}


def score_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    bm25: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Attach vector/bm25/lexical sub-scores and the combined score.

    Shared by search and by the calibration script so both always agree on how
    a chunk is scored.
    """
    # 中文说明：
    #   用途：给每个候选块补上 vector_score / lexical_score / bm25_score 与合并后的 similarity。
    #   参数：query —— 用于 bigram 相似度；chunks —— 字典列表，每个字典必须含 chunk_id、
    #         content、vector，以及调用方注入的 query_vector（它是「查询级」信息，不属于块本身）；
    #         bm25 —— 可选的 {chunk_id: 分数}。
    #   返回：**同一个列表对象**（就地修改每个字典，不新建对象）。
    #   为什么单独抽成一个函数（英文 docstring 已说明）：search_knowledge 与
    #   scripts/calibrate_retrieval.py 共用同一套算式，避免出现「校准用的分数」与
    #   「线上检索的分数」不是一回事。
    bm25 = bm25 or {}
    for chunk in chunks:
        # 中文说明：只有块里还没有该分数时才计算，因此第二次调用会沿用第一次的结果而不重算。
        # 当前仓库里的调用方都不会预先填充分数，所以实际总是走计算分支；这个入口的价值在于
        # 「重复调用 score_chunks」是安全的。注意沿用下来的是 round(...,6) 之后的值，
        # 所以重复调用与单次调用可能出现 1e-6 量级的差异。
        vector_score = chunk.get("vector_score")
        if vector_score is None:
            vector_score = _cosine_similarity(chunk["query_vector"], chunk["vector"])
        lexical_score = chunk.get("lexical_score")
        if lexical_score is None:
            lexical_score = _lexical_similarity(query, chunk["content"])
        bm25_score = bm25.get(chunk["chunk_id"], 0.0)
        # 中文说明：统一保留 6 位小数——既让日志/JSON 可读，也避免浮点尾数刷屏；
        # round 不改变排序（差异远大于 1e-6 才会影响名次）。
        chunk["vector_score"] = round(vector_score, 6)
        chunk["lexical_score"] = round(lexical_score, 6)
        chunk["bm25_score"] = round(bm25_score, 6)
        # 中文说明：合并分数用三个显式权重计算。当前只有 VECTOR_WEIGHT 是 1，另外两个是 0，
        # 所以 similarity 在数值上等于 vector_score（权重说明见文件头部常量区）。
        chunk["similarity"] = round(
            VECTOR_WEIGHT * vector_score
            + BM25_WEIGHT * bm25_score
            + LEXICAL_WEIGHT * lexical_score,
            6,
        )
    return chunks


def search_knowledge(
    query: str,
    top_k: int = 6,
    *,
    category: str | None = None,
    include_below_threshold: bool = False,
    include_noise: bool = False,
    rerank_candidates: bool | None = None,
) -> list[dict[str, Any]]:
    """Return relevant snippets with source, heading path and metadata.

    Retrieval is two-stage: embedding similarity selects a candidate pool, then
    a cross-encoder (`qwen3-rerank`) reorders it against the query. The rerank
    stage is best-effort — if it is unavailable the embedding order is returned
    and `rerank_fallback` records why, rather than failing the request.

    Distractor documents (`offtopic`, `noise`) are excluded by default: they are
    calibration fixtures, and their scores overlap the relevant range too much for
    a cutoff to separate them. Pass ``include_noise=True`` only to measure them.
    """
    # 中文说明：
    #   用途：RAG 检索链路的唯一入口，被 agent.py 的 search_project_knowledge 工具调用。
    #   参数：query —— 检索查询。多轮会话里它不是用户原话，而是 conversation.rewrite_query
    #                  改写后的独立查询（由 agent.current_retrieval_query 注入），因为
    #                  「他的胜率呢」这类追问本身没有可检索的主语。
    #         top_k —— 期望返回条数，默认 6；返回值另有 20 条的硬上限（见函数末尾）。
    #         category —— 只检索某一类文档（VALID_CATEGORIES 之一）。
    #         include_below_threshold —— True 时跳过阈值过滤，把低分候选也返回。这是给
    #                  校准/测试观察原始分数用的，不应把低分结果当作答案依据。
    #         include_noise —— True 时把 offtopic/noise 干扰文档也纳入候选，仅供阈值校准。
    #         rerank_candidates —— 显式覆盖「本轮是否用精排」；None 表示按环境变量
    #                  RERANK_ENABLED 决定（这是评测与校准脚本的控制入口）。
    #   返回：命中字典列表，按相关性降序。每个字典含 source / heading_path / category /
    #         version / content，以及 similarity、rerank_score、rerank_rank、rerank_applied；
    #         发生降级时还带 rerank_fallback（原因）。
    #   两阶段检索（英文 docstring 的核心，这里展开）：
    #     第一阶段「召回」用 embedding 余弦相似度粗排，取 RERANK_CANDIDATE_POOL 条作为候选池；
    #     第二阶段「精排」把候选池交给 cross-encoder 按「查询 + 文档一起打分」重排，再按
    #     MIN_RERANK_SCORE 截断。粗排只保证「正确的片段有没有进池」，不保证名次；池子必须大于
    #     top_k，因为粗排完全可能把真正正确的片段排在第 10 名开外，只有进了池才有机会被精排
    #     提到前面——一旦没进池，精排再准也救不回来。
    if not isinstance(query, str) or not query.strip():
        # 中文说明：先判类型再判空——查询可能来自模型生成的工具参数，不保证是字符串。
        raise ValueError("Search query cannot be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        # 中文说明：isinstance(top_k, bool) 必须单独先判：Python 里 bool 是 int 的子类，
        # True 能通过 int 检查并被当成 1，于是 `top_k=True` 会静默变成「只取 1 条」。
        raise ValueError("top_k must be a positive integer.")
    if category is not None and category not in VALID_CATEGORIES:
        raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}")
    if category is not None and include_noise:
        # Asking for one distractor category while also excluding distractor
        # categories is contradictory; refusing beats silently honouring one.
        # 中文说明：指定 category 与 include_noise=True 语义冲突（前者已经限定了类别）。
        # 直接拒绝，好过默默只执行其中一个、让调用方以为两个条件都生效了。
        raise ValueError("category 与 include_noise=True 不能同时使用")
    if not INDEX_PATH.is_file():
        raise KnowledgeIndexError("知识库索引不存在，请先运行 scripts/index_knowledge.py。")

    # 中文说明：use_rerank 决定走哪条路径，也决定后面用哪个阈值、池子开多大。
    use_rerank = rerank_enabled() if rerank_candidates is None else bool(rerank_candidates)
    # 中文说明：用 max 而不是直接用 RERANK_CANDIDATE_POOL——调用方要的条数可能比池子还大
    # （评测脚本会传 top_k=40），池子至少要装得下请求量。不用精排时池子没有意义，取 top_k 即可。
    pool_size = max(RERANK_CANDIDATE_POOL, top_k) if use_rerank else top_k

    # 中文说明：只取需要的列。embedding 也在这里取出，是因为余弦相似度在 Python 里算
    # （SQLite 没有向量类型，无法在 SQL 里算）。
    sql = (
        "SELECT chunk_id, source, heading, heading_path, category, version, source_url, "
        "content, embedding FROM knowledge_chunks"
    )
    # 中文说明：条件与参数分开累积，最后用 ? 占位符拼进 SQL（参数化查询）。全程不做字符串
    # 插值，既避免注入，也省掉手工转义引号的麻烦。
    conditions: list[str] = []
    parameters: list[Any] = []
    if category is not None:
        # 中文说明：调用方明确指定分类时只取该类，此时「排除干扰类」不再适用——用户要的就是它。
        conditions.append("category = ?")
        parameters.append(category)
    elif not include_noise:
        # Excluded in SQL rather than after ranking: a distractor removed here can
        # never consume one of the top_k slots, so the caller still receives top_k
        # real hits instead of a short list.
        # 中文说明（本项目最容易被忽略的一处设计）：干扰文档在**SQL 层**就被排除，而不是
        # 先排序再在 Python 里过滤。原因很实在：干扰文档的相似度/精排分数与正常内容高度重叠
        # （精排 0.8453–0.9453，见 NON_CONTENT_CATEGORIES 的英文注释），如果先取候选池再过滤，
        # 一条高分干扰文档就会占掉一个名额；而名额是按 top_k 截断的，过滤后调用方可能只剩
        # 一两条结果甚至空列表。在 SQL 层排除，等于让干扰文档**从未参与**名额竞争。
        placeholders = ", ".join("?" for _ in NON_CONTENT_CATEGORIES)
        conditions.append(f"category NOT IN ({placeholders})")
        # 中文说明：NON_CONTENT_CATEGORIES 是 frozenset，迭代顺序不保证；先 sorted 再展开，
        # 保证参数与占位符顺序稳定（可复现，日志里也好比对）。
        parameters.extend(sorted(NON_CONTENT_CATEGORIES))
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    try:
        with sqlite3.connect(INDEX_PATH) as connection:
            rows = connection.execute(sql, tuple(parameters)).fetchall()
            # 中文说明：同一个连接里顺带算 BM25，避免为一次打分再打开一次数据库。
            # 当前 BM25 权重为 0，所以这一步的结果暂时不影响排序（见常量区说明）。
            bm25 = _bm25_scores(connection, query)
    except sqlite3.Error as exc:
        raise KnowledgeIndexError("知识库索引无法读取，请重新运行 scripts/index_knowledge.py 重建。") from exc
    if not rows:
        # 注意：这里把「索引表为空」和「category 过滤后没有命中」都汇报成「索引为空」，
        # 错误文案与实际原因不一定对应（分类下没有文档时也会走到这里）。如实指出，未改动代码。
        raise KnowledgeIndexError("知识库索引为空，请运行 scripts/index_knowledge.py 生成索引。")

    try:
        # 中文说明：只需为查询本身做一次 embedding——候选向量早在建索引时就存好了，
        # 这也是「检索成本与语料规模成正比、但只有一次 API 调用」的原因。
        query_vector = build_embeddings().embed_query(query)
    except Exception as exc:
        # 中文说明：把 SDK 抛出的各种底层异常统一收敛成 KnowledgeAPIError，并给出可执行的
        # 中文排查方向（网络、地域、Workspace、API Key）。上层因此只需处理一种异常。
        raise KnowledgeAPIError(
            "Qwen Embedding 服务暂时不可用，请检查网络、地域、Workspace 和 API Key 配置。"
        ) from exc

    try:
        candidates: list[dict[str, Any]] = []
        for (
            chunk_id, source, heading, heading_path, chunk_category,
            version, source_url, content, vector_json,
        ) in rows:
            vector = json.loads(vector_json)
            # 中文说明：维度校验必须在这里做。索引里的向量是「建索引时的模型」产出的，
            # 若期间换了 embedding 模型（维度可能仍是 1024，但语义空间不同），余弦值会失真；
            # 长度不同至少能立刻发现。错误信息同时给出两个长度，便于判断是哪一侧的问题。
            if len(vector) != len(query_vector):
                raise ValueError(
                    f"向量维度不一致（索引 {len(vector)} vs 查询 {len(query_vector)}），请重建索引。"
                )
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "heading": heading,
                    "heading_path": heading_path,
                    "category": chunk_category,
                    "version": version,
                    "source_url": source_url,
                    "content": content,
                    "vector": vector,
                }
            )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        # 中文说明：这三类异常覆盖「向量不是合法 JSON」「JSON 里的元素不是数字」
        # 「维度不一致」，统一翻译成「请重建索引」——因为从用户角度看，处置动作都是重建。
        # 用普通 dict 而不用 dataclass 承载候选：这些字典一路传到最后会直接变成给模型和
        # 前端面板的结果对象，中途要不断加字段（vector/query_vector/similarity/rerank_*），
        # 用 dict 加字段最省事。
        raise KnowledgeIndexError(
            "知识库索引格式或向量维度异常，请重新运行 scripts/index_knowledge.py 重建。"
        ) from exc

    # Only score chunks that BM25 or the vector pass could plausibly rank, but
    # the corpus is small enough that scoring everything is simplest and exact.
    # 中文说明：上面这段英文注释的意思是「本可以只给可能入围的块打分，但语料太小，全部打分
    # 最简单也最精确」。实际效果就是**全库打分**：每个查询都要对全部分块（当前上千条）算一次
    # 余弦相似度。这个规模下完全够用，但语料涨到十万级时这里会成为瓶颈（需要近似最近邻或
    # 向量数据库）。query_vector 注入到每个候选，是因为 score_chunks 需要「块 + 查询」两个向量。
    for candidate in candidates:
        candidate["query_vector"] = query_vector
    ranked = score_chunks(query, candidates, bm25)
    for candidate in ranked:
        # 中文说明：算完就把两个向量从结果里摘掉。向量又长又对外无用，留着只会让结果对象
        # （最终会进模型的工具返回值、前端检索面板、SSE 载荷）白白变大。
        candidate.pop("vector", None)
        candidate.pop("query_vector", None)

    # 中文说明：排序键是 (分数降序, chunk_id 升序)。第二项不可省——分数相同的块必须有一个
    # 确定的先后（这里用 chunk_id），否则同一查询在不同进程/不同次运行可能给出不同顺序，
    # 测试也就无法断言具体结果。
    ranked.sort(key=lambda item: (-item["similarity"], item["chunk_id"]))

    rerank_fallback = None
    if use_rerank:
        # 中文说明：粗排名次只用来决定「谁进池」，池内顺序此时无所谓——精排会整体重排。
        # 截断池子同时也限制了发往 rerank 接口的文本总量（20 条上下是这个数据规模的折中）。
        pool = ranked[:pool_size]
        try:
            ordering = rerank(query, [item["content"] for item in pool], top_n=pool_size)
        except RerankUnavailable as exc:
            # 中文说明：精排不可用只记录原因，不抛异常——本轮降级为向量顺序返回。
            rerank_fallback = str(exc)
        else:
            # 中文说明：用 try/except/else 而不是把成功分支也塞进 try：只有真的没有异常时才
            # 改写排序结果，避免把「排序代码自身的错误」也当成「精排不可用」吞掉。
            # rerank() 返回 (原始下标, 分数)，下标对应 pool 的位置，所以按下标写回原候选对象。
            for position, (index, relevance) in enumerate(ordering, start=1):
                pool[index]["rerank_score"] = round(relevance, 6)
                # 中文说明：名次从 1 开始（enumerate(start=1)），供 UI 直接展示。
                pool[index]["rerank_rank"] = position
            # The reranker decides the order, so sort by its score here; keeping
            # the embedding order would silently discard the rerank entirely.
            # 中文说明（这条英文注释值得反复读）：精排的全部意义就是改顺序。若这里仍按
            # similarity 排序，前面所有精排调用都白做，而且 rerank_score 还在结果里，
            # 表面上看「精排已生效」——属于最难发现的一类 bug。
            # 注意：这里是下标访问 item["rerank_score"]，所以 `is not None` 只能兜住
            # 「键存在但值为 None」；若 rerank 返回的条目数少于 pool 的条数，未被打分的条目
            # 会让这一行抛 KeyError。此处仅如实指出，未改动代码。
            pool.sort(
                key=lambda item: (
                    -(item["rerank_score"] if item["rerank_score"] is not None else -1.0),
                    item["chunk_id"],
                )
            )
            # 中文说明：精排生效时最终候选**只剩池内这 pool_size 条**，池外候选被丢弃。
            # 也就是说召回上限由 pool_size 决定，MIN_RERANK_SCORE 只在池内起作用——
            # 这是「候选池大小」比阈值更需要小心调整的原因（池子漏掉的，阈值再宽松也找不回）。
            ranked = pool

    for item in ranked:
        # 中文说明：给所有结果补齐这两个键（值可能是 None），让返回结构在任何路径下都一致：
        # 调用方（agent 工具、前端面板）不必为「有没有精排」写两套取值逻辑。
        item.setdefault("rerank_score", None)
        item.setdefault("rerank_rank", None)
        # 中文说明：只有「本来想精排、且没有降级」才算 applied=True。
        item["rerank_applied"] = rerank_fallback is None and use_rerank
        # 中文说明：降级原因只在真正降级时出现，调用方据此把本轮标为「检索质量下降」。
        if rerank_fallback:
            item["rerank_fallback"] = rerank_fallback

    # 中文说明：两个阈值分别对应两条路径，而且作用在**不同的键**上——
    #   精排路径   → MIN_RERANK_SCORE   比 rerank_score（cross-encoder 的相关性分）
    #   降级/关闭 → MIN_RETRIEVAL_SCORE 比 similarity（余弦相似度）
    # 两条路径的分数分布完全不同（见两个常量处的实测区间），所以不能共用一个阈值，
    # 也不能在降级时沿用精排阈值。
    if use_rerank and rerank_fallback is None:
        cutoff = MIN_RERANK_SCORE
        key = "rerank_score"
    else:
        cutoff = MIN_RETRIEVAL_SCORE
        key = "similarity"

    if include_below_threshold:
        # 中文说明：跳过阈值过滤，把候选全部返回（含低分）。校准脚本正是靠它看到原始分数
        # 分布，进而决定阈值该放在哪里；常规问答路径不要打开。
        matched = ranked
    else:
        # 中文说明：要求分数存在且达到阈值。`is not None` 的判断是必要的——降级路径上
        # rerank_score 是 None，直接比较会抛 TypeError。
        matched = [
            item for item in ranked
            if item[key] is not None and item[key] >= cutoff
        ]
    # 中文说明：最外层再夹一次上限：返回值同时不超过 top_k 与 20。
    # 这里的 20 是硬编码的（与 RERANK_CANDIDATE_POOL 数值相同但含义不同——那个是池子大小）。
    # 因此即使调用方传 top_k=40（scripts/eval_agent.py 就是这么调的），最多也只拿到 20 条。
    return matched[: min(top_k, 20)]
