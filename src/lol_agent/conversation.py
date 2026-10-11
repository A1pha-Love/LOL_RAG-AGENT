"""Multi-turn conversation: session persistence plus retrieval query rewriting.

Two problems are solved here, and they are related.

**Persistence.** ``ask()`` in agent.py starts from scratch every call, so a
follow-up question has no context. ``SessionStore`` keeps turns in SQLite so a
conversation survives a process restart and can later be listed by a UI.

**Query rewriting.** Retrieval is computed from the *current question text*
only. A follow-up like "他的胜率呢？" ("what about his win rate?") contains no
searchable subject, so embedding it retrieves nothing useful. ``rewrite_query``
condenses (history, question) into one standalone retrieval query — the standard
condensation step — and falls back to the original question on any failure so a
rewrite problem never breaks a turn.

Traceability is deliberate: the rewritten query is persisted, because "what did
the retriever actually search for" is the single most useful debugging signal in
a multi-turn RAG system.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Session state is derived, machine-local state, so it lives with the other
# caches. Not under reports/: that directory is not writable when the file
# sandbox is confined, and a session store must be writable at runtime.
SESSION_DB_PATH = Path(
    os.getenv("SESSION_DB_PATH") or (PROJECT_ROOT / ".cache" / "sessions.db")
)

MAX_REWRITE_TURNS = 4

# Character budget for the transcript handed to the model, and the cap applied to
# a single turn within it.
#
# Measured: a stored turn averages ~400 characters, and one Q&A round is ~1,200.
# 60,000 characters is roughly 47% of the model's 128K context, leaving room for
# the system prompt, tool results and the answer, and holds ~48 rounds — far more
# than the 2.5-round average session. Beyond that the oldest rounds are dropped.
#
# Deliberately a deterministic trim rather than an LLM summary: a summary would
# cost an extra model call on every long session to preserve context that a
# retrieval-augmented answer can reconstruct from the knowledge base anyway.
MAX_HISTORY_CHARS = 60_000
MAX_HISTORY_TURN_CHARS = 6_000
MAX_REWRITE_CHARS = 2000

REWRITE_PROMPT = """你在为一个中文问答系统改写检索查询。

给定「对话历史」和「用户当前问题」，请输出一个**可以独立理解**的检索查询，
用于在项目文档库中做语义检索。

要求：
1. 把当前问题里的指代（他/它/这个英雄/上面那个/那呢）替换成历史中明确提到的实体或话题。
2. 补全当前问题省略的主语与对象，使它脱离对话历史也能被理解。
3. 保留问题里的关键术语、数字、英文标识符（如英雄 ID、patch 版本号）。
4. 只输出改写后的查询本身，不要解释、不要加引号、不要输出多行。
5. 如果当前问题本身已经完整、无需依赖历史，则原样输出它。

对话历史：
{history}

用户当前问题：{question}

检索查询："""


class SessionNotFound(KeyError):
    """The requested session id does not exist."""


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str
    created_at: str
    retrieval_query: str | None = None
    tool_calls: tuple[str, ...] = ()


@dataclass
class Session:
    session_id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    turns: list[Turn] = field(default_factory=list)

    def history(self, max_turns: int = MAX_REWRITE_TURNS) -> list[Turn]:
        """The most recent turns, used as condensing context."""
        return self.turns[-max_turns:] if max_turns > 0 else list(self.turns)

    def user_questions(self) -> list[str]:
        return [turn.content for turn in self.turns if turn.role == "user"]


def _now() -> str:
    """Current UTC time as an ISO string, with microseconds.

    Microseconds matter for ordering: opening two conversations, or pinning two, can
    easily happen within the same second, and second precision would leave the order
    to an arbitrary tie-break. ISO strings compare lexicographically and the shorter
    (second-precision) values written by older versions still sort correctly against
    the longer ones.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class SessionStore:
    """SQLite-backed conversation storage.

    A plain table is used rather than a checkpointer so the schema is explicit
    and a future UI can list, rename and search conversations directly.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or SESSION_DB_PATH)

    def close(self) -> None:
        """Present for symmetry; this store holds no long-lived connection.

        Each operation opens and closes its own connection so the database file
        is never left locked (which on Windows breaks unlink/rename).
        """

    # -- schema ----------------------------------------------------------

    @contextlib.contextmanager
    def _connect(self):
        """Yield a connection and always close it.

        ``with sqlite3.connect(...)`` only wraps a transaction — it does *not*
        close the connection, which on Windows leaves the database file locked
        until garbage collection and breaks any later unlink/rename.

        The schema is created and migrated here rather than in an explicit setup
        step, so an existing database is upgraded by whichever operation touches it
        first. Sessions predating pinning therefore keep working without the user
        running anything.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id     TEXT NOT NULL UNIQUE,
                    title          TEXT NOT NULL DEFAULT '',
                    created_at     TEXT NOT NULL,
                    updated_at     TEXT NOT NULL,
                    pinned         INTEGER NOT NULL DEFAULT 0,
                    pinned_at      TEXT,
                    last_active_at TEXT
                );
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    retrieval_query TEXT,
                    tool_calls      TEXT NOT NULL DEFAULT '[]',
                    created_at      TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_id);
                """
            )
            self._migrate(connection)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Add columns introduced after the first release, and backfill them.

        ``ALTER TABLE ... ADD COLUMN`` is idempotent only by checking the existing
        columns first; SQLite has no ``ADD COLUMN IF NOT EXISTS``.
        """
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
        if "pinned" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        if "pinned_at" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN pinned_at TEXT")
        if "last_active_at" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN last_active_at TEXT")
            # Backfill from the newest turn so existing conversations sort by their
            # real recency rather than all landing on one value.
            connection.execute(
                """
                UPDATE sessions SET last_active_at = COALESCE(
                    (SELECT MAX(t.created_at) FROM turns t WHERE t.session_id = sessions.session_id),
                    updated_at
                )
                WHERE last_active_at IS NULL
                """
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_order "
            "ON sessions(pinned DESC, pinned_at DESC, last_active_at DESC, id DESC)"
        )

    # -- writes ----------------------------------------------------------

    def create_session(self, title: str = "") -> str:
        session_id = uuid.uuid4().hex[:12]
        stamp = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO sessions (session_id, title, created_at, updated_at, last_active_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, title, stamp, stamp, stamp),
            )
        return session_id

    def ensure_session(self, session_id: str, title: str = "") -> str:
        stamp = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at, last_active_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, title, stamp, stamp, stamp),
            )
        return session_id

    def set_pinned(self, session_id: str, pinned: bool) -> None:
        """Pin or unpin a conversation; raises if it does not exist.

        Pinning records its own timestamp so the pinned group keeps a stable order
        (most recently pinned first) instead of reshuffling with activity.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET pinned = ?, pinned_at = ? WHERE session_id = ?",
                (1 if pinned else 0, _now() if pinned else None, session_id),
            )
            if cursor.rowcount == 0:
                raise SessionNotFound(session_id)

    def touch(self, session_id: str) -> None:
        """Record that a conversation was just opened.

        Called by the UI so opening a conversation moves it up within the
        non-pinned group. Raises if the session does not exist.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
                (_now(), session_id),
            )
            if cursor.rowcount == 0:
                raise SessionNotFound(session_id)

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        retrieval_query: str | None = None,
        tool_calls: list[str] | None = None,
        title_from_first_question: bool = True,
    ) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        stamp = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO turns (session_id, role, content, retrieval_query, tool_calls, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, role, content, retrieval_query, json.dumps(tool_calls or [], ensure_ascii=False), stamp),
            )
            # The first user question becomes the session title, so a UI list is
            # readable without extra metadata.
            if role == "user" and title_from_first_question:
                row = connection.execute(
                    "SELECT title FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if row is not None and not row["title"]:
                    connection.execute(
                        "UPDATE sessions SET title = ? WHERE session_id = ?",
                        (content.strip()[:60], session_id),
                    )
            connection.execute(
                "UPDATE sessions SET updated_at = ?, last_active_at = ? WHERE session_id = ?",
                (stamp, stamp, session_id),
            )

    def set_title(self, session_id: str, title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title.strip()[:60], _now(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    # -- reads -----------------------------------------------------------

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        # Order: pinned conversations first (most recently pinned at the top), then
        # the rest by most recent activity. `touch()` bumps last_active_at when a
        # conversation is opened, so opening one moves it to the head of the
        # non-pinned group while the pinned group stays above it.
        #
        # Timestamps are secondary to `id`: _now() has second precision, so two
        # sessions touched in the same second must still have a stable order.
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT s.session_id, s.title, s.created_at, s.updated_at,
                          s.pinned, s.last_active_at,
                          (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.session_id) AS turn_count
                   FROM sessions s
                   ORDER BY s.pinned DESC, s.pinned_at DESC, s.last_active_at DESC, s.id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Session:
        if not self.path.is_file():
            raise SessionNotFound(session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise SessionNotFound(session_id)
            turn_rows = connection.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id", (session_id,)
            ).fetchall()
        return Session(
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            turns=[
                Turn(
                    role=turn["role"],
                    content=turn["content"],
                    created_at=turn["created_at"],
                    retrieval_query=turn["retrieval_query"],
                    tool_calls=tuple(json.loads(turn["tool_calls"] or "[]")),
                )
                for turn in turn_rows
            ],
        )


# --------------------------------------------------------------------------
# retrieval query rewriting
# --------------------------------------------------------------------------

def trimmed_history(
    turns: list[Turn],
    max_chars: int = MAX_HISTORY_CHARS,
    max_turn_chars: int = MAX_HISTORY_TURN_CHARS,
) -> list[Turn]:
    """The most recent turns that fit the character budget, oldest dropped first.

    Recent context is what resolves follow-up references, so the budget is spent
    from the newest turn backwards. Individual oversized turns are truncated
    rather than dropped, because discarding the newest answer would lose exactly
    the context a follow-up depends on.
    """
    if not turns:
        return []

    def clamp(turn: Turn) -> Turn:
        if len(turn.content) <= max_turn_chars:
            return turn
        return Turn(
            role=turn.role,
            content=turn.content[:max_turn_chars] + "…（内容过长，已截断）",
            created_at=turn.created_at,
            retrieval_query=turn.retrieval_query,
            tool_calls=turn.tool_calls,
        )

    kept: list[Turn] = []
    used = 0
    for turn in reversed(turns):
        candidate = clamp(turn)
        size = len(candidate.content)
        if kept and used + size > max_chars:
            break
        kept.append(candidate)
        used += size
    kept.reverse()
    return kept


def format_history(turns: list[Turn], max_chars: int = MAX_REWRITE_CHARS) -> str:
    """Render turns as a compact transcript for the rewrite prompt."""
    if not turns:
        return "（无历史，这是第一轮）"
    lines = []
    for turn in turns:
        speaker = "用户" if turn.role == "user" else "助手"
        # Assistant answers are long; the subject usually appears early, and the
        # prompt only needs enough to resolve references.
        content = turn.content.strip().replace("\n", " ")
        if turn.role == "assistant" and len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{speaker}：{content}")
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
    return transcript


def rewrite_query(question: str, history: list[Turn]) -> tuple[str, str | None]:
    """Condense (history, question) into one standalone retrieval query.

    Returns ``(query_to_search, fallback_reason)``. On any failure the original
    question is returned together with the reason, so a rewrite problem degrades
    retrieval quality instead of failing the turn.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if not history:
        return question, None

    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        from .config import chat_base_url, require_api_key

        model = ChatOpenAI(
            model=os.getenv("QWEN_CHAT_MODEL", "qwen3.7-flash").strip(),
            api_key=require_api_key(),
            base_url=chat_base_url(),
        )
        prompt = REWRITE_PROMPT.format(history=format_history(history), question=question)
        response = model.invoke([HumanMessage(content=prompt)])
        rewritten = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:  # noqa: BLE001 - any failure must degrade, not raise
        return question, f"{type(exc).__name__}: {exc}"

    # Models sometimes wrap the answer in quotes or add a trailing colon line.
    cleaned = rewritten.strip().strip('"').strip("“”").strip("`")
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    if not cleaned:
        return question, "rewrite returned empty text"
    if len(cleaned) > 400:
        return question, "rewrite returned an implausibly long query"
    return cleaned, None
