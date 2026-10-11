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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KBASE_DIR = PROJECT_ROOT / "kbase"
INDEX_PATH = PROJECT_ROOT / "reports" / "knowledge.db"

# --- chunking -------------------------------------------------------------
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200
MAX_TABLE_CHARS = 2400
OVERLAP_CHARS = 160
CHUNKER_VERSION = "heading-path-v2"

# --- retrieval ------------------------------------------------------------
EMBEDDING_BATCH_SIZE = 20
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
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 0.0
LEXICAL_WEIGHT = 0.0
# No-answer cutoff when reranking. Measured top-1 rerank score on
# evals/retrieval_calibration.jsonl (17 scored + 6 unanswerable cases):
#   answerable (in scope)  0.8099 – 0.9790
#   corpus-answerable but offtopic  0.8453 – 0.9453
#   unanswerable           0.2840 – 0.6185
# 0.70 sits in the gap. It only separates "the corpus has an answer" from "it
# does not"; keeping off-topic but answerable questions out is a prompt-level
# concern, not a retrieval-score one.
MIN_RERANK_SCORE = 0.70
# Candidate pool handed to the reranker, and the cutoff used when reranking is
# unavailable (the vector score is a different scale and cannot share the above).
RERANK_CANDIDATE_POOL = 20
MIN_RETRIEVAL_SCORE = 0.3703

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
NON_CONTENT_CATEGORIES = frozenset({"offtopic", "noise"})
REQUIRED_META = ("doc_id", "title", "category")


class KnowledgeIndexError(RuntimeError):
    """The local knowledge index is missing, empty, or incompatible."""


class KnowledgeAPIError(RuntimeError):
    """The Qwen Embedding request failed."""


class KnowledgeDocumentError(RuntimeError):
    """A knowledge-base document is malformed."""


@dataclass(frozen=True)
class KnowledgeDocument:
    path: Path
    doc_id: str
    title: str
    category: str
    version: str
    source_url: str
    license: str
    source_type: str
    retrieved_at: str
    body: str
    content_hash: str


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    source: str
    heading: str
    heading_path: str
    category: str
    version: str
    source_url: str
    license: str
    text: str
    ordinal: int
    meta: dict[str, Any] = field(default_factory=dict, compare=False)


# --------------------------------------------------------------------------
# front matter + document discovery
# --------------------------------------------------------------------------

_FRONT_MATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a document into (metadata, body). Tolerates a missing block."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        meta[key.strip()] = value
    return meta, text[match.end():]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_documents(kbase_dir: Path | None = None) -> list[Path]:
    """Return every Markdown document in the knowledge base, sorted by name."""
    root = kbase_dir or KBASE_DIR
    if not root.is_dir():
        raise KnowledgeIndexError(f"知识库目录不存在：{root}")
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def load_document(path: Path) -> KnowledgeDocument:
    """Read one document, validating the front matter it must carry."""
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
        version=meta.get("version", "n/a").strip(),
        source_url=meta.get("source_url", "").strip(),
        license=meta.get("license", "").strip(),
        source_type=meta.get("source_type", "unknown").strip(),
        retrieved_at=meta.get("retrieved_at", "").strip(),
        body=body,
        content_hash=_sha256(raw),
    )


# --------------------------------------------------------------------------
# heading-aware block splitting
# --------------------------------------------------------------------------

# MULTILINE is required: these patterns are applied to a whole document body,
# so `^` and `$` must match at line boundaries.
_HEADING_RE = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.*?)[ \t]*\r?$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*\r?$", re.MULTILINE)
_FENCE_RE = re.compile(r"^[ \t]*```", re.MULTILINE)
_BACKTICK_RE = re.compile(r"`+")


def _heading_title(raw: str) -> str:
    """Normalise a heading's text: drop trailing markup and inline backticks."""
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
    matches = list(_HEADING_RE.finditer(body))
    stack: list[str] = []
    blocks: list[tuple[list[str], str]] = []

    def emit(path: list[str], text: str) -> None:
        text = text.strip()
        if text:
            blocks.append((path, text))

    if not matches:
        emit([], body)
        return blocks

    emit([], body[: matches[0].start()])

    for index, match in enumerate(matches):
        depth = len(match.group(1))
        title = _heading_title(match.group(2))
        stack = stack[: depth - 1]
        stack.append(title)

        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        emit(list(stack), body[match.end():end])

    return _split_tables(blocks)


def _split_tables(blocks: list[tuple[list[str], str]]) -> list[tuple[list[str], str]]:
    """Keep each Markdown table in one block, separate from surrounding prose.

    Consecutive table rows must stay together: splitting them makes the table's
    leading rows (the ones carrying ``| 值 | 数量 |``) unreachable, because the
    surrounding prose then forms its own chunk and the header row is orphaned.
    """
    result: list[tuple[list[str], str]] = []
    for path, text in blocks:
        lines = text.splitlines()
        if not any(_TABLE_ROW_RE.match(line) for line in lines):
            result.append((path, text))
            continue

        segment: list[str] = []
        segment_is_table = False

        def flush_segment() -> None:
            nonlocal segment
            joined = "\n".join(segment).strip()
            if joined:
                result.append((path, joined))
            segment = []

        for line in lines:
            is_table = bool(_TABLE_ROW_RE.match(line))
            # Start a new segment only when the kind changes AND the previous
            # segment was prose; a blank line inside a table is not a boundary.
            if segment and segment_is_table != is_table and not (segment_is_table and not line.strip()):
                flush_segment()
                segment_is_table = is_table
            elif not segment:
                segment_is_table = is_table
            segment.append(line)
        flush_segment()
    return result


def _split_oversized(text: str, limit: int) -> list[str]:
    """Deterministic windowed split with overlap, used only for huge blocks."""
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
    packed: list[tuple[str | None, str]] = []
    buffer: list[str] = []
    buffer_path: list[str] = []
    buffer_size = 0

    def emit(label: list[str]) -> None:
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
        return buffer_path == block_path

    for path, block in blocks:
        path_text = " > ".join(path)

        if len(block) > MAX_CHUNK_CHARS:
            emit(buffer_path)
            buffer_path = []
            # Leave room for the heading-path prefix added by chunk_document,
            # so no emitted chunk can bust the advertised maximum.
            reserve = len(path_text) + 64
            for piece in _split_oversized(block, max(MIN_CHUNK_CHARS, MAX_CHUNK_CHARS - reserve)):
                packed.append((path_text or None, piece))
            continue

        if not buffer:
            buffer_path = list(path)
        elif buffer_size + len(block) + 2 > MAX_CHUNK_CHARS or not merges(buffer_path, path):
            emit(buffer_path)
            buffer_path = list(path)

        buffer.append(block)
        buffer_size += len(block) + 2

    emit(buffer_path)

    # Second pass: fold a runt into its neighbour so no chunk is trivially small.
    # Merging only across an identical heading path cannot re-cross an entity
    # boundary, so two champions still never share a chunk.
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
    body = document.body
    chunks: list[KnowledgeChunk] = []
    ordinal = 0
    for heading_path, text in _merge_blocks(_split_blocks(body)):
        if heading_path:
            parts = [part.strip() for part in heading_path.split(">")]
            if parts and parts[0] == document.title:
                parts = parts[1:]
            prefix = document.title if not parts else f"{document.title} > {' > '.join(parts)}"
        else:
            prefix = document.title
        payload = f"{prefix}\n{text}"
        # Final guard: the advertised maximum counts the prefix too, so a
        # composed chunk can never exceed it.
        payloads = (
            [payload]
            if len(payload) <= MAX_CHUNK_CHARS
            else [
                f"{prefix}\n{piece}"
                for piece in _split_oversized(text, max(MIN_CHUNK_CHARS, MAX_CHUNK_CHARS - len(prefix) - 1))
            ]
        )
        for payload in payloads:
            digest = hashlib.sha256(
                f"{document.doc_id}\0{ordinal}\0{payload}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.doc_id}#{ordinal:04d}-{digest}",
                    doc_id=document.doc_id,
                    source=document.path.relative_to(PROJECT_ROOT).as_posix(),
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
            ordinal += 1
    return chunks


def load_knowledge_chunks(kbase_dir: Path | None = None) -> list[KnowledgeChunk]:
    """Parse every knowledge-base document into chunks."""
    chunks: list[KnowledgeChunk] = []
    for path in discover_documents(kbase_dir):
        chunks.extend(chunk_document(load_document(path)))
    return chunks


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

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
_TERM_RE = re.compile(r"[a-z0-9_]+")


def _fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from the query's Latin terms."""
    terms = _TERM_RE.findall(query.casefold())
    return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))

_REQUIRED_CHUNK_COLUMNS = {
    "chunk_id", "doc_id", "source", "heading", "heading_path", "category",
    "version", "source_url", "license", "ordinal", "content", "embedding",
}


def _existing_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _ensure_schema(connection: sqlite3.Connection) -> str:
    """Create the schema, migrating an incompatible legacy index if needed."""
    columns = _existing_columns(connection, "knowledge_chunks")
    note = "created"
    if columns and columns != _REQUIRED_CHUNK_COLUMNS:
        connection.execute("DROP TABLE IF EXISTS knowledge_chunks")
        note = f"migrated from legacy schema {sorted(columns)}"
    connection.executescript(_SCHEMA)
    try:
        connection.executescript(_FTS_SCHEMA)
        note += "; fts5 indexed"
    except sqlite3.Error:
        # FTS5 is optional: retrieval falls back to vector + bigram scoring.
        note += "; fts5 unavailable"
    return note


def build_knowledge_index(*, dry_run: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Embed the knowledge base and replace the local SQLite retrieval index.

    With ``dry_run=True`` the chunking runs and is reported without any
    Embedding API call — the cheap way to inspect a chunking change.
    """
    documents = [load_document(path) for path in discover_documents()]
    if not documents:
        raise KnowledgeIndexError("知识库中没有找到任何 .md 文档。")

    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    if not chunks:
        raise KnowledgeIndexError("知识库文档没有产出任何分块。")

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
            by_category: dict[str, int] = {}
            for chunk in chunks:
                by_category[chunk.category] = by_category.get(chunk.category, 0) + 1
            for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
                print(f"  {category:10} {count:>5}")
        return report

    embeddings = build_embeddings()
    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(texts)
    if len(vectors) != len(chunks):
        raise KnowledgeAPIError(
            f"Embedding 返回数量（{len(vectors)}）与分块数量（{len(chunks)}）不一致。"
        )

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(INDEX_PATH) as connection:
        note = _ensure_schema(connection)
        connection.execute("DELETE FROM knowledge_chunks")
        connection.execute("DELETE FROM index_sources")
        connection.execute("DELETE FROM index_meta")
        try:
            connection.execute("DELETE FROM chunk_fts")
        except sqlite3.Error:
            pass
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
                    json.dumps(vector),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

        try:
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
                    len(document.body.encode("utf-8")),
                    document.doc_id,
                    document.category,
                    per_document.get(document.doc_id, 0),
                    document.retrieved_at,
                )
                for document in documents
            ],
        )
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
        connection.executemany(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)", sorted(meta.items())
        )

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
    embeddings = build_embeddings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embeddings.embed_documents(texts[start:start + batch_size]))
    return vectors


def _index_meta(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return {row[0]: row[1] for row in connection.execute("SELECT key, value FROM index_meta")}
    except sqlite3.Error:
        return {}


def check_index_staleness() -> dict[str, Any]:
    """Compare indexed source hashes against the files currently on disk.

    Returns a report so callers can warn instead of silently answering from an
    out-of-date index.
    """
    result: dict[str, Any] = {
        "index_exists": INDEX_PATH.is_file(),
        "stale": False,
        "missing_documents": [],
        "changed_documents": [],
        "new_documents": [],
        "meta": {},
    }
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

    on_disk: dict[str, str] = {}
    for path in discover_documents():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        on_disk[relative] = _sha256(path.read_text(encoding="utf-8-sig"))

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
    def features(value: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value.casefold())
        result: set[str] = set(tokens)
        for token in tokens:
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                result.update(token[i:i + 2] for i in range(len(token) - 1))
        return result

    query_features = features(query)
    if not query_features:
        return 0.0
    return len(query_features & features(text)) / len(query_features)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _bm25_scores(connection: sqlite3.Connection, query: str) -> dict[str, float]:
    """Return {chunk_id: normalised BM25 score} for the query's Latin terms.

    Normalisation maps the best match to 1.0 and worse matches linearly below,
    which keeps the value comparable with the cosine and bigram terms.
    """
    expression = _fts_query(query)
    if not expression:
        return {}
    try:
        rows = connection.execute(
            "SELECT chunk_id, -bm25(chunk_fts, 1.0, 2.0, 1.0) AS score "
            "FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY score DESC LIMIT 50",
            (expression,),
        ).fetchall()
    except sqlite3.Error:
        return {}
    if not rows:
        return {}
    best = max(score for _, score in rows)
    if best <= 0:
        return {}
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
    bm25 = bm25 or {}
    for chunk in chunks:
        vector_score = chunk.get("vector_score")
        if vector_score is None:
            vector_score = _cosine_similarity(chunk["query_vector"], chunk["vector"])
        lexical_score = chunk.get("lexical_score")
        if lexical_score is None:
            lexical_score = _lexical_similarity(query, chunk["content"])
        bm25_score = bm25.get(chunk["chunk_id"], 0.0)
        chunk["vector_score"] = round(vector_score, 6)
        chunk["lexical_score"] = round(lexical_score, 6)
        chunk["bm25_score"] = round(bm25_score, 6)
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
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if category is not None and category not in VALID_CATEGORIES:
        raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}")
    if category is not None and include_noise:
        # Asking for one distractor category while also excluding distractor
        # categories is contradictory; refusing beats silently honouring one.
        raise ValueError("category 与 include_noise=True 不能同时使用")
    if not INDEX_PATH.is_file():
        raise KnowledgeIndexError("知识库索引不存在，请先运行 scripts/index_knowledge.py。")

    use_rerank = rerank_enabled() if rerank_candidates is None else bool(rerank_candidates)
    pool_size = max(RERANK_CANDIDATE_POOL, top_k) if use_rerank else top_k

    sql = (
        "SELECT chunk_id, source, heading, heading_path, category, version, source_url, "
        "content, embedding FROM knowledge_chunks"
    )
    conditions: list[str] = []
    parameters: list[Any] = []
    if category is not None:
        conditions.append("category = ?")
        parameters.append(category)
    elif not include_noise:
        # Excluded in SQL rather than after ranking: a distractor removed here can
        # never consume one of the top_k slots, so the caller still receives top_k
        # real hits instead of a short list.
        placeholders = ", ".join("?" for _ in NON_CONTENT_CATEGORIES)
        conditions.append(f"category NOT IN ({placeholders})")
        parameters.extend(sorted(NON_CONTENT_CATEGORIES))
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    try:
        with sqlite3.connect(INDEX_PATH) as connection:
            rows = connection.execute(sql, tuple(parameters)).fetchall()
            bm25 = _bm25_scores(connection, query)
    except sqlite3.Error as exc:
        raise KnowledgeIndexError("知识库索引无法读取，请重新运行 scripts/index_knowledge.py 重建。") from exc
    if not rows:
        raise KnowledgeIndexError("知识库索引为空，请运行 scripts/index_knowledge.py 生成索引。")

    try:
        query_vector = build_embeddings().embed_query(query)
    except Exception as exc:
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
        raise KnowledgeIndexError(
            "知识库索引格式或向量维度异常，请重新运行 scripts/index_knowledge.py 重建。"
        ) from exc

    # Only score chunks that BM25 or the vector pass could plausibly rank, but
    # the corpus is small enough that scoring everything is simplest and exact.
    for candidate in candidates:
        candidate["query_vector"] = query_vector
    ranked = score_chunks(query, candidates, bm25)
    for candidate in ranked:
        candidate.pop("vector", None)
        candidate.pop("query_vector", None)

    ranked.sort(key=lambda item: (-item["similarity"], item["chunk_id"]))

    rerank_fallback = None
    if use_rerank:
        pool = ranked[:pool_size]
        try:
            ordering = rerank(query, [item["content"] for item in pool], top_n=pool_size)
        except RerankUnavailable as exc:
            rerank_fallback = str(exc)
        else:
            for position, (index, relevance) in enumerate(ordering, start=1):
                pool[index]["rerank_score"] = round(relevance, 6)
                pool[index]["rerank_rank"] = position
            # The reranker decides the order, so sort by its score here; keeping
            # the embedding order would silently discard the rerank entirely.
            pool.sort(
                key=lambda item: (
                    -(item["rerank_score"] if item["rerank_score"] is not None else -1.0),
                    item["chunk_id"],
                )
            )
            ranked = pool

    for item in ranked:
        item.setdefault("rerank_score", None)
        item.setdefault("rerank_rank", None)
        item["rerank_applied"] = rerank_fallback is None and use_rerank
        if rerank_fallback:
            item["rerank_fallback"] = rerank_fallback

    if use_rerank and rerank_fallback is None:
        cutoff = MIN_RERANK_SCORE
        key = "rerank_score"
    else:
        cutoff = MIN_RETRIEVAL_SCORE
        key = "similarity"

    if include_below_threshold:
        matched = ranked
    else:
        matched = [
            item for item in ranked
            if item[key] is not None and item[key] >= cutoff
        ]
    return matched[: min(top_k, 20)]
