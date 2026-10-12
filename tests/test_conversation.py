"""Tests for session persistence and retrieval-query rewriting.

The store tests are pure SQLite and run offline. The rewrite tests cover only
the parts that must work without a model: the no-history shortcut, the
degradation path, and prompt/transcript formatting. The model call itself is
exercised live by scripts/measure_rewrite.py, not here, so this suite stays
safe for a pre-commit hook.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import conversation as conv  # noqa: E402


@contextlib.contextmanager
def _temp_store(temp_db):
    """A throwaway session database, supplied by the temp_db fixture."""
    store = conv.SessionStore(temp_db)
    try:
        yield store
    finally:
        store.close()


# --------------------------------------------------------------------------
# session store
# --------------------------------------------------------------------------

def test_default_store_path_is_writable(temp_db) -> None:
    """Guards a real failure mode: a default path under a read-only directory.

    The session store was briefly defaulted into reports/, which is not writable
    under the confined sandbox, so every session failed to open at runtime.
    """
    store = conv.SessionStore()
    assert store.path.name.endswith(".db")
    parent = store.path.parent
    assert parent == PROJECT_ROOT / ".cache", f"unexpected default location: {parent}"
    session_id = store.create_session()
    store.append_turn(session_id, "user", "可写性检查")
    assert store.get_session(session_id).turns[0].content == "可写性检查"
    store.delete_session(session_id)


def test_create_and_read_back_a_session(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        store.append_turn(session_id, "user", "这份数据有多少场比赛？")
        store.append_turn(session_id, "assistant", "51,053 场。")

        session = store.get_session(session_id)
        assert [turn.role for turn in session.turns] == ["user", "assistant"]
        assert session.turns[0].content == "这份数据有多少场比赛？"


def test_first_user_question_becomes_the_title(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        store.append_turn(session_id, "user", "谁被 ban 最多？")
        assert store.get_session(session_id).title == "谁被 ban 最多？"

        # A later question must not overwrite the title.
        store.append_turn(session_id, "user", "那他的胜率呢？")
        assert store.get_session(session_id).title == "谁被 ban 最多？"


def test_turn_metadata_is_persisted(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        store.append_turn(
            session_id,
            "assistant",
            "答案",
            retrieval_query="亚索的禁用率是多少",
            tool_calls=["get_ban_stats", "search_project_knowledge"],
        )
        turn = store.get_session(session_id).turns[0]
        assert turn.retrieval_query == "亚索的禁用率是多少"
        assert turn.tool_calls == ("get_ban_stats", "search_project_knowledge")


def test_turns_keep_their_order(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        for index in range(5):
            store.append_turn(session_id, "user", f"问题{index}")
        contents = [turn.content for turn in store.get_session(session_id).turns]
        assert contents == [f"问题{index}" for index in range(5)]


def test_list_sessions_orders_by_recent_activity(temp_db) -> None:
    with _temp_store(temp_db) as store:
        first = store.create_session()
        second = store.create_session()
        store.append_turn(first, "user", "第一个")
        store.append_turn(second, "user", "第二个")

        listed = store.list_sessions()
        assert {item["session_id"] for item in listed} == {first, second}
        assert listed[0]["session_id"] == second, "most recently updated should come first"
        assert listed[0]["turn_count"] == 1


def test_unknown_session_raises(temp_db) -> None:
    with _temp_store(temp_db) as store:
        store.create_session()
        with pytest.raises(conv.SessionNotFound):
            store.get_session("does-not-exist")


def test_sessions_are_isolated_from_each_other(temp_db) -> None:
    with _temp_store(temp_db) as store:
        first = store.create_session()
        second = store.create_session()
        store.append_turn(first, "user", "只属于第一个会话")
        assert store.get_session(second).turns == []


def test_delete_session_removes_its_turns(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        store.append_turn(session_id, "user", "待删除")
        store.delete_session(session_id)
        assert store.list_sessions() == []
        with pytest.raises(conv.SessionNotFound):
            store.get_session(session_id)


def test_invalid_role_is_rejected(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        with pytest.raises(ValueError, match="role"):
            store.append_turn(session_id, "system", "x")


def test_list_sessions_on_missing_database_is_empty() -> None:
    store = conv.SessionStore(PROJECT_ROOT / ".cache" / "definitely-absent.db")
    assert store.list_sessions() == []


# --------------------------------------------------------------------------
# pinning and ordering
# --------------------------------------------------------------------------

def test_pinned_sessions_sort_above_the_rest(temp_db) -> None:
    with _temp_store(temp_db) as store:
        first = store.create_session()
        second = store.create_session()
        third = store.create_session()

        store.set_pinned(first, True)

        listed = [item["session_id"] for item in store.list_sessions()]
        assert listed[0] == first, "a pinned conversation must come first"
        assert listed[1:] == [third, second], "the rest keep their recency order"


def test_sessions_default_to_unpinned(temp_db) -> None:
    with _temp_store(temp_db) as store:
        store.create_session()
        assert store.list_sessions()[0]["pinned"] == 0


def test_a_new_turn_raises_a_session_within_the_unpinned_group(temp_db) -> None:
    """Activity is what reorders the list, and a turn is what counts as activity.

    Deliberately driven through `append_turn`: that is the only thing production does
    to mark a conversation active, so exercising it directly keeps the test honest
    about how the order actually changes.
    """
    with _temp_store(temp_db) as store:
        oldest = store.create_session()
        middle = store.create_session()
        newest = store.create_session()

        assert [item["session_id"] for item in store.list_sessions()] == [newest, middle, oldest]

        store.append_turn(oldest, "user", "新问题")

        assert [item["session_id"] for item in store.list_sessions()] == [oldest, newest, middle]


def test_opening_a_session_alone_does_not_reorder(temp_db) -> None:
    """Reading a conversation is not activity.

    Reported as a bug: clicking a conversation made the sidebar jump under the cursor,
    which looked like the wrong item had been selected. `list_sessions` must therefore
    be a pure read with no side effect on the order.
    """
    with _temp_store(temp_db) as store:
        oldest = store.create_session()
        newest = store.create_session()
        before = [item["session_id"] for item in store.list_sessions()]
        assert before == [newest, oldest]

        # Reading the conversation and listing it repeatedly must change nothing.
        store.get_session(oldest)
        for _ in range(3):
            store.list_sessions()

        assert [item["session_id"] for item in store.list_sessions()] == before


def test_a_new_turn_never_lifts_a_session_above_a_pinned_one(temp_db) -> None:
    """Requirement: an active conversation sits *below* the pinned ones."""
    with _temp_store(temp_db) as store:
        pinned = store.create_session()
        other = store.create_session()
        store.set_pinned(pinned, True)

        store.append_turn(other, "user", "新问题")

        listed = [item["session_id"] for item in store.list_sessions()]
        assert listed[0] == pinned
        assert listed[1] == other


def test_unpinning_returns_a_session_to_the_activity_order(temp_db) -> None:
    with _temp_store(temp_db) as store:
        first = store.create_session()
        second = store.create_session()
        store.set_pinned(first, True)
        assert store.list_sessions()[0]["session_id"] == first

        store.set_pinned(first, False)

        # Back to recency order, where `second` is newer.
        assert [item["session_id"] for item in store.list_sessions()] == [second, first]


def test_pinning_the_same_session_twice_is_idempotent(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        store.set_pinned(session_id, True)
        store.set_pinned(session_id, True)
        assert len(store.list_sessions()) == 1
        assert store.list_sessions()[0]["pinned"] == 1


def test_pinning_an_unknown_session_raises(temp_db) -> None:
    with _temp_store(temp_db) as store:
        store.create_session()
        with pytest.raises(conv.SessionNotFound):
            store.set_pinned("does-not-exist", True)


def test_appending_a_turn_to_an_unknown_session_raises(temp_db) -> None:
    with _temp_store(temp_db) as store:
        store.create_session()
        with pytest.raises(conv.SessionNotFound):
            store.append_turn("does-not-exist", "user", "问题")


def test_new_sessions_record_last_active_at(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        assert store.list_sessions()[0]["last_active_at"], "recency must be recorded"
        store.append_turn(session_id, "user", "问题")
        assert store.list_sessions()[0]["last_active_at"]


def test_ordering_stays_stable_within_the_same_second(temp_db) -> None:
    """Timestamps carry microseconds, so rapid activity still orders.

    With second precision, several turns in quick succession would tie and the order
    would fall back to insertion — the list would look like it ignored the most recent
    activity.
    """
    with _temp_store(temp_db) as store:
        ids = [store.create_session() for _ in range(5)]
        for session_id in ids:
            store.append_turn(session_id, "user", "问题")
        assert store.list_sessions()[0]["session_id"] == ids[-1]


def test_migration_adds_columns_to_an_existing_database(temp_db) -> None:
    """An older database must be upgraded in place, keeping its conversations.

    The columns are added by whichever operation touches the database first, so an
    existing user does not have to run a migration step.
    """
    import sqlite3

    connection = sqlite3.connect(temp_db)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE turns (
            turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            retrieval_query TEXT,
            tool_calls TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("legacy", "旧会话", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    )
    connection.execute(
        """INSERT INTO turns (session_id, role, content, tool_calls, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        ("legacy", "user", "旧问题", "[]", "2026-01-02T00:00:00+00:00"),
    )
    connection.commit()
    connection.close()

    store = conv.SessionStore(temp_db)
    listed = store.list_sessions()

    assert len(listed) == 1, "the existing conversation must survive the migration"
    assert listed[0]["session_id"] == "legacy"
    assert listed[0]["title"] == "旧会话"
    assert listed[0]["pinned"] == 0
    # Backfilled from the newest turn, not left null, so ordering is meaningful.
    assert listed[0]["last_active_at"] == "2026-01-02T00:00:00+00:00"

    # The new features must work on the migrated database too.
    store.set_pinned("legacy", True)
    assert store.list_sessions()[0]["pinned"] == 1


def test_migration_is_idempotent(temp_db) -> None:
    store = conv.SessionStore(temp_db)
    store.create_session()
    # Every operation re-runs the migration; a second pass must not fail.
    for _ in range(3):
        assert len(store.list_sessions()) == 1


def test_ensure_session_is_idempotent(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        store.append_turn(session_id, "user", "原始内容")
        store.ensure_session(session_id)
        assert store.get_session(session_id).title == "原始内容"


# --------------------------------------------------------------------------
# history selection and formatting
# --------------------------------------------------------------------------

def _turn(role: str, content: str) -> conv.Turn:
    return conv.Turn(role=role, content=content, created_at="2026-01-01T00:00:00+00:00")


def test_history_returns_the_most_recent_turns(temp_db) -> None:
    with _temp_store(temp_db) as store:
        session_id = store.create_session()
        for index in range(10):
            store.append_turn(session_id, "user", f"问题{index}")
        recent = store.get_session(session_id).history(max_turns=4)
        assert [turn.content for turn in recent] == [f"问题{index}" for index in range(6, 10)]


def test_format_history_marks_the_first_turn() -> None:
    assert "第一轮" in conv.format_history([])


def test_format_history_labels_speakers_and_truncates_long_answers() -> None:
    text = conv.format_history([_turn("user", "谁被 ban 最多？"), _turn("assistant", "x" * 900)])
    assert "用户：谁被 ban 最多？" in text
    assert "助手：" in text
    assert len(text) < 900, "long assistant answers should be truncated"


def test_format_history_respects_the_char_budget() -> None:
    turns = [_turn("user", "长" * 500) for _ in range(20)]
    assert len(conv.format_history(turns, max_chars=800)) <= 800


# --------------------------------------------------------------------------
# history trimming
# --------------------------------------------------------------------------

def test_trimming_keeps_everything_that_fits() -> None:
    turns = [_turn("user", "短问题") for _ in range(5)]
    assert conv.trimmed_history(turns, max_chars=10_000) == turns


def test_trimming_keeps_the_most_recent_turns() -> None:
    """Recent context is what resolves a follow-up, so it must survive."""
    turns = [_turn("user", f"问题{index}" + "x" * 100) for index in range(10)]
    kept = conv.trimmed_history(turns, max_chars=300)
    assert kept, "must keep at least the newest turn"
    assert kept[-1] is turns[-1]
    assert turns[0] not in kept, "the oldest turns are the ones dropped"
    assert len(kept) < len(turns)


def test_trimming_never_exceeds_the_budget_when_it_can_avoid_it() -> None:
    turns = [_turn("user", "x" * 100) for _ in range(10)]
    kept = conv.trimmed_history(turns, max_chars=450)
    assert sum(len(turn.content) for turn in kept) <= 450


def test_a_single_oversized_turn_is_truncated_not_dropped() -> None:
    """Dropping the newest answer would lose the context a follow-up needs."""
    turns = [_turn("user", "问题"), _turn("assistant", "答" * 50_000)]
    kept = conv.trimmed_history(turns, max_chars=1_000, max_turn_chars=500)
    assert len(kept) == 2
    assert kept[1].content.endswith("（内容过长，已截断）")
    assert len(kept[1].content) < 600


def test_trimming_an_empty_history_is_empty() -> None:
    assert conv.trimmed_history([]) == []


def test_trimming_preserves_turn_metadata() -> None:
    turn = conv.Turn(
        role="assistant",
        content="很短",
        created_at="2026-01-01T00:00:00+00:00",
        retrieval_query="检索词",
        tool_calls=("a", "b"),
    )
    kept = conv.trimmed_history([turn], max_chars=1000)
    assert kept[0].retrieval_query == "检索词"
    assert kept[0].tool_calls == ("a", "b")


def test_history_budget_is_comfortably_above_a_normal_session() -> None:
    """Guards against the budget being lowered until it silently drops context."""
    typical_round_chars = 1_200
    assert conv.MAX_HISTORY_CHARS / typical_round_chars >= 20
    assert conv.MAX_REWRITE_CHARS < conv.MAX_HISTORY_CHARS


# --------------------------------------------------------------------------
# rewriting: the parts that must work without a model
# --------------------------------------------------------------------------

def test_first_turn_needs_no_rewrite() -> None:
    """With no history there is nothing to resolve, so no API call is made."""
    query, fallback = conv.rewrite_query("这份数据有多少场比赛？", [])
    assert query == "这份数据有多少场比赛？"
    assert fallback is None


def test_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError):
        conv.rewrite_query("   ", [])


def test_rewrite_failure_degrades_to_the_original_question(monkeypatch) -> None:
    """A rewrite problem must not fail the turn, only reduce retrieval quality."""
    import langchain_openai

    def explode(*args, **kwargs):
        raise RuntimeError("simulated rewrite outage")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", explode)
    query, fallback = conv.rewrite_query("那他的胜率呢？", [_turn("user", "谁被 ban 最多？")])
    assert query == "那他的胜率呢？"
    assert fallback is not None and "simulated rewrite outage" in fallback


def test_rewrite_prompt_carries_history_and_question() -> None:
    prompt = conv.REWRITE_PROMPT.format(history="用户：谁被 ban 最多？", question="那他的胜率呢？")
    assert "谁被 ban 最多？" in prompt
    assert "那他的胜率呢？" in prompt


def test_rewrite_prompt_forbids_explanation() -> None:
    """The model must return only the query, or parsing needs to be clever."""
    assert "只输出改写后的查询本身" in conv.REWRITE_PROMPT


def test_rewrite_without_history_never_builds_a_chat_client() -> None:
    """No history means no pronouns to resolve, so the question is already standalone.

    Pinned because the cost is invisible: without the early return, every first turn of
    every conversation would spend an extra model call, and nothing would fail loudly.
    The check installs a client stub that raises if one is ever constructed, which is
    deterministic rather than a timing assertion.
    """
    import langchain_openai

    class _ExplodingClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("rewrite must not build a chat client without history")

    question = "这份数据去重后有多少场比赛？"
    real = langchain_openai.ChatOpenAI
    langchain_openai.ChatOpenAI = _ExplodingClient
    try:
        rewritten, fallback = conv.rewrite_query(question, [])
    finally:
        langchain_openai.ChatOpenAI = real

    assert rewritten == question, "the question must pass through unchanged"
    assert fallback is None, "a no-op rewrite is not a fallback"


def test_rewrite_rejects_an_empty_question() -> None:
    with pytest.raises(ValueError):
        conv.rewrite_query("   ", [])
