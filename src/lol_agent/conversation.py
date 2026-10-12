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

# ============================================================================
# 中文说明
#
# 职责：解决多轮会话的两个问题——会话持久化与检索查询改写。两者是关联的。
#
#   * 持久化：agent.ask() 每次都从零开始，追问因此没有任何上下文。SessionStore 把每一轮
#     问答存进 SQLite，于是会话能跨进程重启续接，也能被前端列出、重命名、置顶、删除。
#     这里用手写表结构而不是 LangGraph 的 checkpointer：schema 是显式的，UI 可以直接查询。
#   * 改写：检索只基于「当前问题文本」。「他的胜率呢？」这种追问不含任何可检索的主语，
#     直接拿去做 embedding 什么也召不回。rewrite_query 把（历史, 当前问题）压成一个可独立
#     理解的检索查询——即标准的 query condensation——并在任何失败时回落为原问题，保证改写
#     出问题永远不会毁掉一整轮问答。
#
# 可追溯性是刻意设计的：改写后的查询随会话一起持久化，因为「检索器究竟查了什么」是多轮 RAG
# 里最有用的调试信号。
#
# 本文件几个最容易踩的点（详见各处注释）：
#   1. 会话排序键为什么是 pinned DESC, pinned_at DESC, last_active_at DESC, id DESC；
#   2. _now() 为什么必须带微秒；
#   3. append_turn 更新 last_active_at 为什么是「产生对话才上移」的实现基础；
#   4. sqlite3 的 `with connect(...)` 只包事务、不关连接，在 Windows 上会一直锁着文件；
#   5. SQLite 默认不启用外键约束，所以 ON DELETE CASCADE 不会自动生效。
# ============================================================================

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
# 中文说明（对应项目里唯一的「可写目录」约定）：会话库是运行期必须可写的派生状态，所以和
# 其它缓存一起放在 .cache/；reports/ 在受限文件沙箱下不可写（那里放的是构建产物，如知识
# 索引）。环境变量 SESSION_DB_PATH 可覆盖，测试用临时路径、部署时指向可写目录都靠它。
SESSION_DB_PATH = Path(
    # 中文说明：`os.getenv(...) or 默认路径` 里的 or 是必需的——环境变量被设为空串时也要回落到
    # 默认路径，而不是把空串当成路径（否则会得到当前目录，行为极难排查）。
    os.getenv("SESSION_DB_PATH") or (PROJECT_ROOT / ".cache" / "sessions.db")
)

# 中文说明：改写检索查询时最多带上几轮历史。指代线索通常就在最近一两轮里；给太多既增加改写
# 那次模型调用的输入长度，也会稀释重点，所以 4 是经验取值（不是硬性上限，见 Session.history）。
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
# 中文说明（三个预算各自的职责）：
#   MAX_HISTORY_CHARS —— 送给模型的整段转录字符上限。上面英文注释记录了实测依据：一轮问答
#     约 1200 字符，6 万字符约合模型 128K 上下文的 47%，能给系统提示词、工具结果和回答留出
#     余量，并容纳约 48 轮，远高于平均 2.5 轮的会话。超出后从最旧的轮次开始丢弃。
#     刻意用确定性裁剪而不是 LLM 摘要（英文注释已说明）：摘要在每个长会话上都要多花一次模型
#     调用，而检索增强的回答本就能从知识库重建上下文。
#   MAX_HISTORY_TURN_CHARS —— 单轮上限，防止某一条超长回答吃掉整个预算。超长时**截断该轮**
#     而不是丢弃它（见 trimmed_history）。
#   MAX_REWRITE_CHARS —— 送进改写提示词的历史转录上限；超出时保留末尾（最近的部分）。
MAX_HISTORY_CHARS = 60_000
MAX_HISTORY_TURN_CHARS = 6_000
MAX_REWRITE_CHARS = 2000

# 中文说明：改写提示词本身已经是中文，这里只补充三条阅读要点：
#   1. 第 1、2 条（消解指代、补全省略主语）是这个提示词存在的全部理由；
#   2. 第 3 条要求保留数字与英文标识符（英雄 ID、patch 号）——它们往往是检索的关键词，
#      改写时最容易顺手「翻译」掉；
#   3. 第 4 条（只输出查询本身）同时也是防腐措施：模型若不守规矩，rewrite_query 里还有一层
#      清洗（去引号、只取首行、限长），双侧防护才让改写结果可以直接拿去检索。
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

    # 中文说明：继承 KeyError 而不是自定义一个全新基类——语义上就是「按 key 找不到」，
    # 于是不关心具体类型的调用方也能用 KeyError 捕获。


@dataclass(frozen=True)
class Turn:
    # 中文说明：一轮消息的不可变快照。role 只有 "user" / "assistant" 两种取值
    # （append_turn 在写入前校验，所以从库里读出来的值一定合法）。
    role: str  # "user" | "assistant"
    content: str
    created_at: str  # 写入时刻，ISO 8601 UTC（见 _now）
    # 中文说明：只有 assistant 轮才有值——它记录该轮实际检索用的查询（改写后的），
    # 这正是「多轮 RAG 里最有用调试信号」的落库处。
    retrieval_query: str | None = None
    # 中文说明：用 tuple 而不是 list。frozen dataclass 不允许可变的默认值（list 会触发
    # ValueError），而 tuple 让 Turn 真正不可变、可哈希、可安全地跨线程共享。
    tool_calls: tuple[str, ...] = ()


@dataclass
class Session:
    # 中文说明：会话的聚合对象，**可变**（非 frozen）——它由读取结果逐字段填充，且 turns 是
    # 列表。注意它不是 sessions 表的完整镜像：pinned / pinned_at / last_active_at 只服务于
    # 列表排序，由 list_sessions 直接返回给 UI，不需要随会话内容一起加载。
    session_id: str
    title: str = ""  # 默认空串；append_turn 会用第一个用户问题回填（见那里的说明）
    created_at: str = ""
    updated_at: str = ""
    turns: list[Turn] = field(default_factory=list)

    def history(self, max_turns: int = MAX_REWRITE_TURNS) -> list[Turn]:
        """The most recent turns, used as condensing context."""
        # 中文说明：取最近 max_turns 轮作为改写的上下文。
        # 注意 max_turns <= 0 时返回**全部**轮次——这里的 0 不是「零轮」而是「不限」，
        # 属于反直觉的约定，调用时别把 0 当成「不要历史」（那样读起来像是 bug 预防，
        # 实际语义是关闭限制）。
        return self.turns[-max_turns:] if max_turns > 0 else list(self.turns)

    def user_questions(self) -> list[str]:
        # 中文说明：只取用户侧的提问文本（过滤掉助手回答），按时间顺序返回。
        return [turn.content for turn in self.turns if turn.role == "user"]


def _now() -> str:
    """Current UTC time as an ISO string, with microseconds.

    Microseconds matter for ordering: opening two conversations, or pinning two, can
    easily happen within the same second, and second precision would leave the order
    to an arbitrary tie-break. ISO strings compare lexicographically and the shorter
    (second-precision) values written by older versions still sort correctly against
    the longer ones.
    """
    # 中文说明：
    #   用途：生成所有时间戳（created_at / updated_at / pinned_at / last_active_at）的唯一来源。
    #   返回：形如 2026-10-08T12:34:56.123456+00:00 的 ISO 8601 字符串（微秒精度、带 UTC 偏移）。
    #   为什么必须是微秒（上面英文注释的要点）：连续新建两个会话、或连续置顶两个会话，很容易
    #   落在同一秒内；秒精度会让它们的先后交给任意的 tie-break，表现为列表顺序看似随机。
    #   为什么可以直接用字符串排序：同一时区、定长格式下，ISO 字符串的字典序等价于时间序。
    #   旧版本写入的秒精度值（较短）也能与新值正确比较，因此不需要为兼容旧数据做迁移。
    #   用 timezone.utc 而不是 datetime.utcnow()：前者产出带 +00:00 偏移的 aware datetime，
    #   isoformat() 会把时区写进字符串，排序与跨系统展示都不会有歧义（utcnow() 是 naive，
    #   字符串里不带偏移）。
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class SessionStore:
    """SQLite-backed conversation storage.

    A plain table is used rather than a checkpointer so the schema is explicit
    and a future UI can list, rename and search conversations directly.
    """

    # 中文说明：一个方法对应一次完整操作，每次操作自己开连接、自己关连接（见 close 与
    # _connect 的说明）。这样做的好处是：不存在需要维护的长连接，也不会因为某处忘记关闭而
    # 在 Windows 上锁住数据库文件。
    def __init__(self, path: Path | None = None) -> None:
        # 中文说明：path 可注入，测试用临时库；`Path(path or SESSION_DB_PATH)` 把环境变量里
        # 可能出现的字符串路径统一转成 Path 对象，本类后续只按 Path 使用它。
        self.path = Path(path or SESSION_DB_PATH)

    def close(self) -> None:
        """Present for symmetry; this store holds no long-lived connection.

        Each operation opens and closes its own connection so the database file
        is never left locked (which on Windows breaks unlink/rename).
        """

        # 中文说明：这是一个**只含 docstring 的空实现**（函数体即文档字符串，Python 允许）。
        # 它存在的意义是让调用方用统一姿势收尾（finally: store.close()），而真正的清理早已在
        # 每次操作结束时完成。详见上面英文注释：Windows 上未关闭的连接会锁住文件，
        # 破坏之后的 unlink/rename。

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
        # 中文说明：
        #   用途：上下文管理器——产出连接、保证关闭，并顺带完成建表与迁移（因此调用方不必
        #         记得先初始化数据库）。
        #   产出：row_factory 设为 sqlite3.Row 的连接（结果行既能按下标也能按列名访问，
        #         list_sessions 末尾的 dict(row) 就依赖这一点）。
        # 首次运行时 .cache/ 可能不存在：parents=True 支持多级创建，exist_ok=True 让并发/重复
        # 调用不会因为「目录已存在」而失败。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 注意：sqlite3.connect 在文件不存在时会**新建**一个空库。因此「路径写错」不会立刻报错，
        # 而是静默生成一个空数据库（后续 get_session 查不到会抛 SessionNotFound）。
        # 排查「会话怎么都不见了」时，第一件事是确认这里解析出的绝对路径。
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            # 中文说明：sessions 表完整声明了 pinned/pinned_at/last_active_at，但老库缺列——
            # 补列由紧接着的 _migrate 负责。turns 的 (session_id, turn_id) 复合索引服务于
            # 最高频的查询「按会话取全部轮次」，避免全表扫描。
            # 注意（平台怪癖）：turns 里那句 FOREIGN KEY ... ON DELETE CASCADE **不会自动生效**，
            # 因为 SQLite 的外键约束默认关闭（PRAGMA foreign_keys 默认为 OFF）。所以
            # delete_session 必须显式先删 turns 再删 sessions——那两行不是冗余代码。
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
            # 中文说明：把连接交给调用方；正常退出时提交，抛异常时回滚并把异常继续抛出
            # （不吞异常），无论如何 finally 里关闭连接——这正是相对 `with sqlite3.connect()`
            # 的关键差别（后者不关连接）。
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
        # 中文说明：
        #   用途：为旧版本建的库补齐后加的列（pinned / pinned_at / last_active_at），并建排序
        #         索引。因为它挂在 _connect 里，**每次操作都会执行**，所以迁移是天然的幂等操作。
        #   参数：connection —— _connect 已打开的连接（处于事务中，失败会整体回滚）。
        #   返回：无。
        # PRAGMA table_info 每行描述一列，row[1] 是列名（与 knowledge.py 里同样的用法）。
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
        if "pinned" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        if "pinned_at" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN pinned_at TEXT")
        if "last_active_at" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN last_active_at TEXT")
            # Backfill from the newest turn so existing conversations sort by their
            # real recency rather than all landing on one value.
            # 中文说明：新列必须回填。取该会话**最新一轮**的时间；空会话（还没有任何轮次）取不到
            # 结果，于是 COALESCE 回落到 updated_at。这样每一行都有非空值——否则所有旧会话会挤在
            # NULL 上，排序等于失效。
            connection.execute(
                """
                UPDATE sessions SET last_active_at = COALESCE(
                    (SELECT MAX(t.created_at) FROM turns t WHERE t.session_id = sessions.session_id),
                    updated_at
                )
                WHERE last_active_at IS NULL
                """
            )
        # 中文说明：列的排列与方向都与 list_sessions 的 ORDER BY 完全一致，因此那条查询可以
        # 直接利用这条索引完成排序，而不是取出全部行再排。
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_order "
            "ON sessions(pinned DESC, pinned_at DESC, last_active_at DESC, id DESC)"
        )

    # -- writes ----------------------------------------------------------

    def create_session(self, title: str = "") -> str:
        # 中文说明：
        #   用途：新建一个空会话。
        #   参数：title —— 初始标题，默认为空串（之后由 append_turn 用第一个提问回填）。
        #   返回：新会话的对外 id（session_id）。
        # 用 uuid4 的十六进制前 12 位（48 位随机）作为对外 id。注意 sessions 表里同时存在两个
        # 标识：自增整数 id（内部主键，也用于排序的最后一级 tie-break）与 session_id（对外键，
        # 会出现在 URL 与前端里）。取 12 位是「碰撞概率足够低 + 比完整 UUID 短好读」的折中。
        session_id = uuid.uuid4().hex[:12]
        # 中文说明：同一个 stamp 写进三个时间列，语义是「刚创建」。
        stamp = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO sessions (session_id, title, created_at, updated_at, last_active_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, title, stamp, stamp, stamp),
            )
        return session_id

    def ensure_session(self, session_id: str, title: str = "") -> str:
        # 中文说明：幂等地保证某个 id 的会话存在（已存在则什么都不做），返回该 id。
        # INSERT OR IGNORE 依赖 session_id 的 UNIQUE 约束。
        # 关键细节：会话已存在时不仅不覆盖标题，**也不刷新任何时间戳**——因此这个函数不会把旧
        # 会话顶到列表最前面。若这里顺手更新了 last_active_at，「产生对话才上移」这个设计就破了：
        # 每次打开旧会话都会让它上移。
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
        # 中文说明：置顶时刻单独记在 pinned_at 里（上面英文注释说明了原因），于是置顶分组内部
        # 的顺序只由「何时被置顶」决定，不会被会话的新消息打乱。
        # 取消置顶时 pinned_at 置 NULL 而不是保留旧值：这样「是否置顶」与「置顶时间」始终一致，
        # 排序也不会读到一个已经作废的置顶时间。
        # SQLite 没有布尔类型，pinned 用 0/1 存，所以这里做一次布尔到整数的转换。
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET pinned = ?, pinned_at = ? WHERE session_id = ?",
                (1 if pinned else 0, _now() if pinned else None, session_id),
            )
            if cursor.rowcount == 0:
                # 中文说明：会话不存在就报错，而不是让调用方以为置顶成功了（实际什么都没发生）。
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
        # 中文说明：
        #   用途：追加一轮消息（用户提问或助手回答），并同时维护会话的标题与活动时间。
        #   参数：session_id —— 目标会话；role —— 只能是 "user" 或 "assistant"；
        #         content —— 消息正文；retrieval_query —— 该轮实际检索用的查询（仅回答轮有）；
        #         tool_calls —— 该轮用过的工具名；title_from_first_question —— 是否允许用第一个
        #         提问自动命名（置 False 的调用方表示「不要自动命名」）。
        #   返回：无；会话不存在时抛 SessionNotFound。
        if role not in ("user", "assistant"):
            # 中文说明：写库前校验，脏角色不进数据库（读取侧的 Turn 因此可以假定 role 合法）。
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        # 中文说明：一次取时，供 turn 与 session 共用——于是「消息时间」与「会话活动时间」严格
        # 一致，不会出现 turn 比会话的 last_active_at 还新的错位。
        stamp = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO turns (session_id, role, content, retrieval_query, tool_calls, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                # 中文说明：tool_calls 以 JSON 文本存。ensure_ascii=False 让中文不被转义成
                # \uXXXX，直接读库排查时可读性更好（数据库本身是 UTF-8）。
                (session_id, role, content, retrieval_query, json.dumps(tool_calls or [], ensure_ascii=False), stamp),
            )
            # The first user question becomes the session title, so a UI list is
            # readable without extra metadata.
            # 中文说明：只在当前标题为空时才写入——因此只有**第一个**提问会成为标题，用户之后
            # 手动改过的标题也不会被后续提问覆盖。
            if role == "user" and title_from_first_question:
                row = connection.execute(
                    "SELECT title FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if row is not None and not row["title"]:
                    connection.execute(
                        "UPDATE sessions SET title = ? WHERE session_id = ?",
                        # 中文说明：截断到 60 字符，与 set_title 用同一上限，保证列表里标题长度可控。
                        (content.strip()[:60], session_id),
                    )
            # Bumping last_active_at here is what makes a conversation rise in the
            # list: this is the only place production records activity, so the order
            # follows real participation rather than page views. Raises for an unknown
            # session instead of silently updating nothing.
            # 中文说明（「产生对话才上移」的实现基础，三条要点）：
            #   1. 这里是生产路径上**唯一**会修改 last_active_at 的地方（读取会话、改标题都不动
            #      它），所以列表顺序跟随真实对话参与，而不是「谁被点开过」；
            #   2. 这条 UPDATE 同时充当存在性检查：会话不存在时 rowcount 为 0，直接抛
            #      SessionNotFound，而不是静默写入一条孤儿 turn；
            #   3. 顺序很重要——turns 行在这之前已经插入。一旦这里抛异常，异常会穿出 with 体、
            #      触发 _connect 的 rollback，那条 turn 也随之一并回滚。因此对不存在的会话来说，
            #      append_turn 是「原子失败」，不会留下半截数据。
            cursor = connection.execute(
                "UPDATE sessions SET updated_at = ?, last_active_at = ? WHERE session_id = ?",
                (stamp, stamp, session_id),
            )
            if cursor.rowcount == 0:
                raise SessionNotFound(session_id)

    def set_title(self, session_id: str, title: str) -> None:
        # 中文说明：手动改名。与自动命名共用 60 字符上限。
        # 注意它只更新 updated_at，**不**更新 last_active_at——改名不算「产生对话」，因此不会
        # 让会话在列表中上移（这是「排序跟随真实活动」的另一处体现）。
        # 另外：会话不存在时这里不会报错（UPDATE 影响 0 行也返回成功），与 set_pinned /
        # append_turn 的严格风格不一致。此处仅如实指出，未改动代码。
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title.strip()[:60], _now(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        # 中文说明：先删 turns 再删 sessions。因为 SQLite 默认不启用外键约束，
        # ON DELETE CASCADE 不会触发，所以必须显式删；即便将来启用外键，这个顺序也依然正确。
        # 删除不存在的会话不会报错（DELETE 影响 0 行仍然成功），也就是删除是幂等的——
        # 这与 set_pinned / append_turn 的严格风格不同。
        with self._connect() as connection:
            connection.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    # -- reads -----------------------------------------------------------

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        # 中文说明：
        #   用途：给 UI 的会话列表（含轮次数与置顶标记）。
        #   参数：limit —— 最多返回多少条，默认 50。
        #   返回：字典列表（dict(row) 转换后的普通字典，便于 JSON 序列化）；没有数据库文件时
        #         返回空列表。
        if not self.path.is_file():
            # 中文说明：必须在这里提前返回，**不能**为了「顺便建库」而走 _connect——_connect 会
            # mkdir 并创建数据库文件，而列出会话是一个读操作，不该产生副作用。同理，若空库被
            # 建出来，过期检查之类的逻辑也会被误导。
            return []
        # Order: pinned conversations first (most recently pinned at the top), then
        # the rest by most recent activity. `last_active_at` is bumped by
        # `append_turn`, so a conversation rises only once it actually gains a turn —
        # merely opening one to read it does not reorder the list.
        #
        # Timestamps are secondary to `id`: microsecond precision still permits ties in
        # principle, so the id keeps the order stable when it happens.
        #
        # 中文说明（四级排序键，逐级说明「为什么是它」）：
        #   1. pinned DESC —— 置顶的排最前（1 > 0）；
        #   2. pinned_at DESC —— 置顶组内部按「最近置顶」排。因为新消息只改 last_active_at、
        #      不改 pinned_at，所以置顶组不会被聊天活动打乱（这是 set_pinned 单独记录时间戳的
        #      直接收益）；
        #   3. last_active_at DESC —— 其余会话按最近一次「产生对话」排。该列**只**由 append_turn
        #      更新，因此「只打开看一眼」不会让会话上移；老库的这一列由 _migrate 回填；
        #   4. id DESC —— 最终 tie-break。微秒精度理论上仍可能撞上（批量导入、时钟回拨），
        #      用自增主键兜底，顺序因此始终确定、可复现（测试才能稳定断言顺序）。
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
        # 中文说明：turn_count 用的是相关子查询，每行都会执行一次计数。会话数量在几十到几百的
        # 量级时完全够用；规模再大就该改成 LEFT JOIN + GROUP BY。
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Session:
        # 中文说明：
        #   用途：读取一个会话及其全部轮次。
        #   参数：session_id —— 对外会话 id。
        #   返回：Session（含按时间顺序排列的 turns）；不存在时抛 SessionNotFound。
        # 文件不存在时也抛 SessionNotFound，而不是顺手建库——读操作不产生副作用（与
        # list_sessions 的提前返回是同一个理由）。
        if not self.path.is_file():
            raise SessionNotFound(session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise SessionNotFound(session_id)
            # 中文说明：按 turn_id（自增主键）排序，而不是按 created_at——turn_id 是写入顺序的
            # 权威来源，不依赖时钟，同一微秒内的两条消息也不会错序。
            turn_rows = connection.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id", (session_id,)
            ).fetchall()
        return Session(
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            # 中文说明：只映射 Session 真正需要的列。pinned / pinned_at / last_active_at 不在这里
            # 出现——它们只服务于列表排序，由 list_sessions 直接返回给 UI。
            turns=[
                Turn(
                    role=turn["role"],
                    content=turn["content"],
                    created_at=turn["created_at"],
                    retrieval_query=turn["retrieval_query"],
                    # 中文说明：从 JSON 文本还原成 tuple（与 Turn 的不可变设计一致）；
                    # `or "[]"` 兜住历史数据里可能出现的 NULL。
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
    # 中文说明：
    #   用途：从最新一轮往回取，保留落在字符预算内的轮次（最旧的先被丢弃）。
    #   参数：turns —— 完整轮次列表（时间正序）；max_chars —— 总预算；
    #         max_turn_chars —— 单轮上限。
    #   返回：时间正序的轮次列表（至少一轮，可能为空当且仅当输入为空）。
    if not turns:
        return []

    def clamp(turn: Turn) -> Turn:
        # 中文说明：单轮超长时截断并追加省略标记，而不是整轮丢弃（英文注释的理由：丢掉最新的
        # 回答恰好丢掉追问所依赖的上下文）。注意标记是在 max_turn_chars 之后追加的，所以该轮
        # 实际会略超单轮上限几个字符。Turn 是 frozen dataclass，因此这里返回新对象而不是原地改。
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
    # 中文说明：reversed 从最新往回累加——预算优先花在最近的上下文上（英文注释的要点）。
    for turn in reversed(turns):
        candidate = clamp(turn)
        size = len(candidate.content)
        # 中文说明：`kept and` 不能省。它保证**至少保留一轮**（最新那轮），否则一个超长会话会
        # 返回空历史，模型与改写都彻底失去上下文。第一轮即使自己就超预算也会被收下（并已被
        # clamp 截断）。
        if kept and used + size > max_chars:
            break
        kept.append(candidate)
        used += size
    # 中文说明：内部是倒序累加的，返回前必须翻转回时间正序——LangChain 消息顺序与提示词里的
    # 对话顺序都依赖它。
    kept.reverse()
    return kept


def format_history(turns: list[Turn], max_chars: int = MAX_REWRITE_CHARS) -> str:
    """Render turns as a compact transcript for the rewrite prompt."""
    # 中文说明：
    #   用途：把轮次渲染成紧凑转录，直接填进 REWRITE_PROMPT 的 {history} 占位符。
    #   参数：turns —— 待渲染轮次；max_chars —— 转录总上限。
    #   返回：多行文本；空历史时返回固定的中文提示（下面那句），而不是空串——显式告诉模型
    #         这是第一轮，比留空更不容易让它去猜历史。
    if not turns:
        return "（无历史，这是第一轮）"
    lines = []
    for turn in turns:
        speaker = "用户" if turn.role == "user" else "助手"
        # Assistant answers are long; the subject usually appears early, and the
        # prompt only needs enough to resolve references.
        # 中文说明：助手回答截到 400 字；把换行压成空格，避免回答里的列表/表格在提示词里凭空
        # 造出「新的一行对话」，让模型分不清哪一行才是用户说的。
        content = turn.content.strip().replace("\n", " ")
        if turn.role == "assistant" and len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{speaker}：{content}")
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        # 中文说明：超长时保留**末尾**。转录按时间正序拼接，末尾就是最近的内容，而最近的内容才
        # 含有待消解的指代；若写成 transcript[:max_chars] 就正好把需要的部分裁掉了。
        transcript = transcript[-max_chars:]
    return transcript


def rewrite_query(question: str, history: list[Turn]) -> tuple[str, str | None]:
    """Condense (history, question) into one standalone retrieval query.

    Returns ``(query_to_search, fallback_reason)``. On any failure the original
    question is returned together with the reason, so a rewrite problem degrades
    retrieval quality instead of failing the turn.
    """
    # 中文说明：
    #   用途：把（历史, 当前问题）压成一个可独立理解的检索查询。
    #   参数：question —— 用户当前问题；history —— 最近的若干轮（调用方通常已裁剪）。
    #   返回：(用于检索的查询, 失败原因或 None)。任何失败都回落为原问题，并把原因一并返回——
    #         于是「本轮改写失败了」这件事是可见的，而不是悄悄降级。
    if not question.strip():
        # 中文说明：空问题属于调用方 bug，直接抛错而不是静默改写。
        raise ValueError("Question cannot be empty")
    if not history:
        # 中文说明：第一轮没有历史可消解，原问题本身就是完整查询——直接返回，省掉一次模型调用。
        return question, None

    try:
        # 中文说明：这三个 import 放在函数内部：模块级只依赖标准库与 dotenv，因此只使用会话存储
        # （列表、改名、删除）的代码不必因为改写功能而依赖 LangChain，模块导入也更快。
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        from .config import chat_base_url, require_api_key

        # 中文说明：改写用独立的 ChatOpenAI，没有 bind_tools、也没有设置 temperature——它只做
        # 一次单轮文本生成（与 build_agent 里那个绑定工具、可配温度的模型是两回事）。
        model = ChatOpenAI(
            model=os.getenv("QWEN_CHAT_MODEL", "qwen3.7-flash").strip(),
            api_key=require_api_key(),
            base_url=chat_base_url(),
        )
        prompt = REWRITE_PROMPT.format(history=format_history(history), question=question)
        response = model.invoke([HumanMessage(content=prompt)])
        # 中文说明：与 agent._final_text 同样的兜底——content 通常是字符串，也可能是内容块列表。
        rewritten = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:  # noqa: BLE001 - any failure must degrade, not raise
        # 中文说明：宽 except 是刻意的（noqa 注释已说明）：任何失败都必须降级为上抛原问题。
        # 原因字符串里带异常类型名，便于从日志或前端看出是配置问题、网络问题还是别的。
        return question, f"{type(exc).__name__}: {exc}"

    # Models sometimes wrap the answer in quotes or add a trailing colon line.
    # 中文说明：清洗四步，逐条对应模型的实际坏习惯——去首尾空白、去包裹的引号（英文双引号与
    # 中文引号都处理）、去反引号、只取第一行（提示词要求只输出一行，但模型仍可能补一句解释）。
    cleaned = rewritten.strip().strip('"').strip("“”").strip("`")
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    if not cleaned:
        # 中文说明：空结果视为失败，回落原问题。
        return question, "rewrite returned empty text"
    if len(cleaned) > 400:
        # 中文说明：400 字远超正常查询长度。出现这种情况说明模型开始输出解释或复述历史，此时宁可
        # 用原问题去检索，也不要把这种脏查询送进 embedding。
        return question, "rewrite returned an implausibly long query"
    return cleaned, None
