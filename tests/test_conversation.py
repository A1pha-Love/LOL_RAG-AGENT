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
