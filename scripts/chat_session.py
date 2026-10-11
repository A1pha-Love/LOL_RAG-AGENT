"""Interactive multi-turn chat over the LoL data agent.

    python scripts/chat_session.py                    # new session, interactive
    python scripts/chat_session.py --session <id>     # resume an existing session
    python scripts/chat_session.py --list             # list stored sessions
    python scripts/chat_session.py -q "问题"           # one turn of a session, then exit
    python scripts/chat_session.py --once "问题"       # stateless single question

Each session is stored in SQLite, so a conversation survives restarts. The
retrieval query actually used for each turn is printed, because that is the
fastest way to see whether follow-up questions are being rewritten usefully.

``--once`` is the stateless path: nothing is read from or written to the session
store, which is what you want for a one-off question or a script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.agent import ask, ask_in_session  # noqa: E402
from lol_agent.conversation import SessionNotFound, SessionStore  # noqa: E402

EXIT_COMMANDS = {"/exit", "/quit", ":q", "exit", "quit"}


def _redact(message: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    return message.replace(api_key, "[已隐藏]") if api_key else message


def _ask_once(question: str) -> int:
    """Answer without touching the session store."""
    try:
        print(ask(question))
    except Exception as exc:  # noqa: BLE001
        print(f"调用失败（{type(exc).__name__}）：{_redact(str(exc))}", file=sys.stderr)
        return 1
    return 0


def _show_sessions(store: SessionStore) -> int:
    sessions = store.list_sessions()
    if not sessions:
        print("（没有已保存的会话）")
        return 0
    print(f"{'session_id':14} {'更新于':20} {'轮数':>4}  标题")
    for item in sessions:
        print(f"{item['session_id']:14} {item['updated_at'][:19]:20} {item['turn_count']:>4}  {item['title']}")
    return 0


def _run_turn(store: SessionStore, session_id: str, question: str, *, show_query: bool) -> int:
    try:
        outcome = ask_in_session(question, session_id, store=store)
    except Exception as exc:  # noqa: BLE001
        print(f"调用失败（{type(exc).__name__}）：{_redact(str(exc))}", file=sys.stderr)
        return 1

    if show_query and outcome.get("retrieval_query"):
        rewritten = outcome["retrieval_query"]
        marker = "" if rewritten.strip() == question.strip() else "  (已改写)"
        print(f"  [检索查询]{marker} {rewritten}")
        if outcome.get("rewrite_fallback"):
            print(f"  [改写回退] {outcome['rewrite_fallback']}")
    if outcome.get("tool_calls"):
        print(f"  [工具] {', '.join(outcome['tool_calls'])}")
    print()
    print(outcome["answer"])
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", help="resume an existing session id")
    parser.add_argument("--list", action="store_true", help="list stored sessions and exit")
    parser.add_argument("-q", "--question", help="ask one question and exit (non-interactive)")
    parser.add_argument(
        "--once",
        metavar="QUESTION",
        help="stateless single question: no session is created or read",
    )
    parser.add_argument("--quiet-query", action="store_true", help="do not print the retrieval query")
    args = parser.parse_args()

    # The stateless path must not touch the store at all, so it is handled before
    # any session is opened.
    if args.once:
        return _ask_once(args.once.strip())

    store = SessionStore()

    if args.list:
        return _show_sessions(store)

    if args.session:
        try:
            session = store.get_session(args.session)
        except SessionNotFound:
            print(f"找不到会话：{args.session}", file=sys.stderr)
            return 1
        session_id = args.session
        print(f"会话 {session_id}（{len(session.turns)} 条历史）")
    else:
        session_id = store.create_session()
        print(f"新会话：{session_id}")

    if args.question:
        return _run_turn(store, session_id, args.question, show_query=not args.quiet_query)

    print("输入问题开始对话；输入 /exit 退出，/list 查看全部会话。")
    print()
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break
        if question == "/list":
            _show_sessions(store)
            print()
            continue
        print()
        _run_turn(store, session_id, question, show_query=not args.quiet_query)

    print(f"会话已保存：{session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
