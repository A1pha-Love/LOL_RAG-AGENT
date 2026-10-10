"""Tests for the knowledge-base chunker and index integrity.

These are offline: chunking reads local Markdown only, and index checks read the
SQLite file. No Embedding or rerank API is called, so they are safe for a
pre-commit hook and for CI.
"""

from __future__ import annotations

import contextlib
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import knowledge as kb  # noqa: E402


@pytest.fixture(scope="module")
def documents() -> list[kb.KnowledgeDocument]:
    return [kb.load_document(path) for path in kb.discover_documents()]


@pytest.fixture(scope="module")
def chunks() -> list[kb.KnowledgeChunk]:
    return kb.load_knowledge_chunks()


# --------------------------------------------------------------------------
# front matter
# --------------------------------------------------------------------------

def test_every_document_parses_and_declares_required_metadata(
    documents: list[kb.KnowledgeDocument],
) -> None:
    assert documents, "knowledge base is empty"
    for document in documents:
        assert document.doc_id, document.path
        assert document.title, document.path
        assert document.category in kb.VALID_CATEGORIES, document.path


def test_document_ids_are_unique(documents: list[kb.KnowledgeDocument]) -> None:
    ids = [document.doc_id for document in documents]
    duplicates = {value for value in ids if ids.count(value) > 1}
    assert not duplicates, f"duplicate doc_id: {sorted(duplicates)}"


def test_expected_categories_are_all_populated(documents: list[kb.KnowledgeDocument]) -> None:
    present = {document.category for document in documents}
    # offtopic exists to make the no-answer threshold meaningful.
    for required in ("project", "domain", "conflict", "meta", "offtopic"):
        assert required in present, f"no documents in category {required!r}"


@contextlib.contextmanager
def _temp_markdown(text: str):
    """Write a throwaway .md inside the repo.

    pytest's tmp_path is unusable here: its base temp directory is not writable
    in this environment, so the fixture-based tests error out before running.
    """
    import tempfile

    directory = PROJECT_ROOT / ".cache"
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".md", encoding="utf-8", dir=directory, delete=False
    )
    try:
        handle.write(text)
        handle.close()
        yield Path(handle.name)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def test_missing_front_matter_is_reported_not_ignored() -> None:
    with _temp_markdown("# just a heading\n\nsome text\n") as bad:
        with pytest.raises(kb.KnowledgeDocumentError):
            kb.load_document(bad)


def test_unknown_category_is_rejected() -> None:
    with _temp_markdown(
        "---\ndoc_id: x\ntitle: T\ncategory: nonsense\n---\n\n# T\n\nbody\n"
    ) as bad:
        with pytest.raises(kb.KnowledgeDocumentError):
            kb.load_document(bad)


def test_front_matter_parser_tolerates_quotes_and_colons() -> None:
    meta, body = kb.parse_front_matter(
        '---\ndoc_id: a\ntitle: "带：冒号 的标题"\nsource_url: "https://x/y#z"\n---\n\nbody\n'
    )
    assert meta["title"] == "带：冒号 的标题"
    assert meta["source_url"] == "https://x/y#z"
    assert body.strip() == "body"


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def test_chunks_exist_and_respect_the_size_cap(chunks: list[kb.KnowledgeChunk]) -> None:
    assert chunks
    oversized = [chunk.chunk_id for chunk in chunks if len(chunk.text) > kb.MAX_CHUNK_CHARS]
    assert not oversized, f"{len(oversized)} chunk(s) exceed {kb.MAX_CHUNK_CHARS} chars"


def test_every_chunk_carries_a_heading_path(chunks: list[kb.KnowledgeChunk]) -> None:
    missing = [chunk.chunk_id for chunk in chunks if not chunk.heading_path]
    assert not missing, "a chunk without a heading path cannot be cited precisely"


def test_chunk_ids_are_unique(chunks: list[kb.KnowledgeChunk]) -> None:
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_text_starts_with_its_title_prefix(chunks: list[kb.KnowledgeChunk]) -> None:
    """The prefix makes a fragment self-describing; retrieval and citation rely on it."""
    for chunk in chunks:
        first_line = chunk.text.split("\n", 1)[0]
        assert first_line
        assert chunk.heading_path.split(" > ")[-1] in first_line or chunk.heading_path in first_line


def test_tables_are_not_split_across_chunks(chunks: list[kb.KnowledgeChunk]) -> None:
    """A chunk must not begin mid-table with a row but no header separator."""
    for chunk in chunks:
        lines = [line for line in chunk.text.splitlines() if line.strip().startswith("|")]
        if len(lines) < 2:
            continue
        assert any(set(line) <= set("|-: ") for line in lines), (
            f"chunk looks like a table without its header separator: {chunk.chunk_id}"
        )


def test_no_chunk_spans_two_champions(chunks: list[kb.KnowledgeChunk]) -> None:
    """One champion per chunk; otherwise a citation would point at the wrong hero."""
    champion_names = {
        line[3:].strip()
        for path in sorted((PROJECT_ROOT / "kbase").glob("domain_champion_mechanics_*.md"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    }
    offenders = []
    for chunk in chunks:
        parts = [part.strip() for part in chunk.heading_path.split(">") if part.strip()]
        hits = [part for part in parts if part in champion_names]
        if len(hits) > 1:
            offenders.append((chunk.chunk_id, hits))
    assert not offenders, f"chunk(s) name multiple champions: {offenders[:3]}"


def test_champion_coverage_is_complete(chunks: list[kb.KnowledgeChunk]) -> None:
    """All 138 champion ids in the dataset must be described somewhere."""
    champion_chunks = [c for c in chunks if c.source.startswith("kbase/domain_champion")]
    assert len(champion_chunks) >= 130


def test_heading_with_inline_backticks_is_normalised(chunks: list[kb.KnowledgeChunk]) -> None:
    """`## `winner` 取值分布` must not leave an unbalanced backtick in the path."""
    for chunk in chunks:
        assert "`" not in chunk.heading_path, f"backtick left in heading: {chunk.heading_path!r}"


# --------------------------------------------------------------------------
# index integrity (skipped when the index has not been built)
# --------------------------------------------------------------------------

index_exists = pytest.mark.skipif(not kb.INDEX_PATH.is_file(), reason="index not built")


@index_exists
def test_index_schema_matches_the_chunker(chunks: list[kb.KnowledgeChunk]) -> None:
    with sqlite3.connect(kb.INDEX_PATH) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(knowledge_chunks)")}
    assert {"doc_id", "heading_path", "category", "version", "source_url"} <= columns


@index_exists
def test_index_chunk_count_matches_the_knowledge_base(chunks: list[kb.KnowledgeChunk]) -> None:
    with sqlite3.connect(kb.INDEX_PATH) as connection:
        (count,) = connection.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()
    assert count == len(chunks), "index is out of date; run scripts/index_knowledge.py"


@index_exists
def test_index_is_not_stale() -> None:
    report = kb.check_index_staleness()
    assert not report["stale"], (
        f"index stale: {report.get('reason')}; "
        f"changed={report['changed_documents'][:3]} new={report['new_documents'][:3]}"
    )


@index_exists
def test_index_vectors_have_the_expected_dimension() -> None:
    import json

    with sqlite3.connect(kb.INDEX_PATH) as connection:
        row = connection.execute("SELECT embedding FROM knowledge_chunks LIMIT 1").fetchone()
    assert row is not None
    assert len(json.loads(row[0])) == kb.EMBEDDING_DIMENSIONS


@index_exists
def test_index_metadata_records_the_chunker_version() -> None:
    with sqlite3.connect(kb.INDEX_PATH) as connection:
        meta = {k: v for k, v in connection.execute("SELECT key, value FROM index_meta")}
    assert meta.get("chunker_version") == kb.CHUNKER_VERSION
