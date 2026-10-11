"""Streaming question answering, shared by the HTTP backend and any other caller.

Kept separate from the FastAPI app so the event sequence can be unit-tested
without an HTTP client: ``stream_turn`` is an async generator yielding plain
dicts, and only ``apps/api.py`` knows about SSE framing.

Event sequence for one turn:

    {"type": "status",  "phase": "rewriting", "message": ...}  # start of the turn
    {"type": "rewrite", "query": ..., "fallback": ...}         # what retrieval will search
    {"type": "tools",   "names": [...]}                        # tools the model chose
    {"type": "retrieval", "retrieval_query": ..., "results": [...]}  # matched chunks
    {"type": "status",  "phase": "answering", "message": ...}  # model is generating
    {"type": "delta",   "text": "..."}                         # answer fragments
    {"type": "done",    "answer": ..., "seconds": ...}         # final, persisted answer
    {"type": "error",   "message": ...}                        # terminal failure

The ``status`` events exist because the first answer token can take several
seconds to arrive (the reasoning model "thinks" first). Without a signal, a
client shows nothing at all for that whole period and the answer looks frozen.

The answer is persisted only after the stream completes, so a client that
disconnects mid-answer does not leave a half-written transcript behind.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage

from .agent import _TURN, build_agent, prepare_turn, session_messages
from .conversation import SessionNotFound, SessionStore


async def stream_turn(
    question: str,
    session_id: str,
    *,
    store: SessionStore | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield the events for one conversational turn."""
    if not question.strip():
        yield {"type": "error", "message": "问题不能为空"}
        return

    store = store or SessionStore()
    try:
        session = store.get_session(session_id)
    except SessionNotFound:
        yield {"type": "error", "message": f"找不到会话：{session_id}"}
        return

    started = time.perf_counter()

    # Preparation calls the rewrite model through a synchronous client, so it is
    # offloaded to a thread; blocking the event loop here would stall every other
    # in-flight request.
    yield {"type": "status", "phase": "rewriting", "message": "正在改写检索查询…"}
    context = await asyncio.to_thread(prepare_turn, session, question)
    yield {
        "type": "rewrite",
        "query": context.retrieval_query,
        "fallback": context.rewrite_fallback,
    }

    graph = build_agent()
    payload = {"messages": [*session_messages(session), HumanMessage(content=question)]}

    token = _TURN.set(context)
    fragments: list[str] = []
    reported_tools: list[str] = []
    final_messages: list[Any] = []
    announced_answering = False
    try:
        # Two modes at once: "messages" gives per-token deltas for the live
        # answer, "updates" gives the final state so a model that streams no
        # chunks still produces an answer.
        async for mode, chunk in graph.astream(
            payload, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                message, metadata = chunk
                if (metadata or {}).get("langgraph_node") != "agent":
                    continue
                text = _chunk_text(message)
                if text:
                    if not announced_answering:
                        announced_answering = True
                        yield {"type": "status", "phase": "answering", "message": "正在组织回答…"}
                    fragments.append(text)
                    yield {"type": "delta", "text": text}
            else:
                for update in (chunk or {}).values():
                    messages = (update or {}).get("messages")
                    if isinstance(messages, list) and messages:
                        final_messages = messages
                        retrieved = _retrieval_payload(messages)
                        if retrieved is not None:
                            yield {"type": "retrieval", **retrieved}
                context_tools = list(context.tool_calls)
                if context_tools != reported_tools:
                    reported_tools = context_tools
                    yield {"type": "tools", "names": list(reported_tools)}
    except Exception as exc:  # noqa: BLE001 - the client must learn the turn failed
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return
    finally:
        _TURN.reset(token)

    answer = "".join(fragments).strip() or _last_message_text(final_messages)
    if not answer:
        yield {"type": "error", "message": "模型未返回内容"}
        return

    store.append_turn(session_id, "user", question)
    store.append_turn(
        session_id,
        "assistant",
        answer,
        retrieval_query=context.retrieval_query,
        tool_calls=list(context.tool_calls),
    )
    yield {
        "type": "done",
        "session_id": session_id,
        "answer": answer,
        "retrieval_query": context.retrieval_query,
        "tool_calls": list(context.tool_calls),
        "seconds": round(time.perf_counter() - started, 2),
    }


def _chunk_text(chunk: Any) -> str:
    """Extract plain text from a streamed model chunk, ignoring reasoning parts.

    Qwen's reasoning models emit ``reasoning_content`` alongside the answer; only
    the answer should reach the user.
    """
    if chunk is None:
        return ""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _retrieval_payload(messages: list[Any]) -> dict[str, Any] | None:
    """Extract the last knowledge-retrieval result from an update, if any.

    The retrieval panel needs the matched chunks (source, heading path, score),
    and LangGraph's ``updates`` channel carries the tool's return value. Returns
    None when the update contains no retrieval result.
    """
    for message in reversed(messages or []):
        if getattr(message, "type", None) != "tool":
            continue
        if getattr(message, "name", None) != "search_project_knowledge":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return None
        if not isinstance(content, dict) or content.get("kind") != "retrieval":
            return None
        return {
            "retrieval_query": content.get("retrieval_query"),
            "results": content.get("results") or [],
            "error": content.get("error"),
        }
    return None


def _last_message_text(messages: list[Any]) -> str:
    """Final answer text from the graph's end state.

    Used when the model produced no streamed chunks; without it a non-streaming
    response would be reported as "no content".
    """
    for message in reversed(messages or []):
        if type(message).__name__ != "AIMessage":
            continue
        text = _chunk_text(message)
        if text.strip():
            return text.strip()
    return ""
