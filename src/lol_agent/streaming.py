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

# ============================================================================
# 中文说明
#
# 本模块负责"一轮问答"的生成逻辑，被 HTTP 后端（apps/api.py）与命令行复用。
# 关键设计：它只产出普通的 dict 事件，不关心传输格式——SSE 的分帧（CRLF 空行、
# event/data 字段、ping 注释行）都由 apps/api.py 使用的 sse-starlette 库完成。
# 这样事件序列可以直接用单元测试驱动，不需要起 HTTP 客户端。
#
# 一轮问答的事件顺序（与上面英文说明一致）：
#     status(rewriting) -> rewrite -> [tools] -> [retrieval] -> status(answering)
#     -> delta* -> done          （失败则用 error 收尾并直接返回）
#
# 为什么需要 status 事件：
#   推理模型"先思考再回答"，从发起请求到第一个答案 token 之间可能安静好几秒。
#   如果这段时间一个字节都不发，客户端界面上什么都没有，用户会以为卡死了。
#   所以先发一条 status 说明"正在改写检索查询/正在组织回答"，让界面立即有反馈。
#
# 为什么答案在流结束后才落库：
#   中途断开（关页面、网络断）时不应留下半截对话记录；只有拿到完整答案才写入
#   用户与助手两条消息，保证会话历史里不会出现"只有问题没有答案"的残缺轮次。
#
# 与 agent.py 的协作：
#   检索查询改写、工具选择都在图的执行过程中完成，改写结果通过 ContextVar
#   `_TURN` 传给工具；本模块负责 set/reset 它，确保并发请求之间互不串台。
# ============================================================================

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
    """Yield the events for one conversational turn.

    生成一轮问答的全部事件（异步生成器）。

    参数:
        question: 用户本轮的问题原文；空白字符串视为非法输入。
        session_id: 会话 id；会话不存在时以 error 事件收尾。
        store: 可选的会话存储，默认新建一个 SessionStore；测试与后端注入自己的实例。

    产出:
        一系列 dict 事件，type 取值为 status / rewrite / tools / retrieval / delta /
        done / error（顺序见模块开头说明）。

    说明:
        本函数不抛异常给调用方，所有失败都以 {"type": "error"} 事件表达——调用方是
        HTTP 流式响应，一旦开始发送就无法再改状态码，错误必须走事件通道。
    """
    if not question.strip():
        yield {"type": "error", "message": "问题不能为空"}
        return

    store = store or SessionStore()
    try:
        session = store.get_session(session_id)
    except SessionNotFound:
        yield {"type": "error", "message": f"找不到会话：{session_id}"}
        return

    # 用 perf_counter 而不是 time.time：它测量的是单调递增的时间间隔，不受系统
    # 时钟调整影响，适合算"这一轮花了多久"。
    started = time.perf_counter()

    # Preparation calls the rewrite model through a synchronous client, so it is
    # offloaded to a thread; blocking the event loop here would stall every other
    # in-flight request.
    # 中文：检索查询改写用的是同步客户端，因此丢到线程里执行。若直接在事件循环里
    # 调用，这个请求的几秒等待会阻塞同一个进程内其它所有正在进行的请求。
    yield {"type": "status", "phase": "rewriting", "message": "正在改写检索查询…"}
    context = await asyncio.to_thread(prepare_turn, session, question)
    yield {
        "type": "rewrite",
        "query": context.retrieval_query,
        # 改写失败时使用的兜底查询（通常是原始问题），一并告知客户端便于排查。
        "fallback": context.rewrite_fallback,
    }

    graph = build_agent()
    payload = {"messages": [*session_messages(session), HumanMessage(content=question)]}

    # 把本轮上下文放进 ContextVar：工具执行时用它拿到改写后的检索查询。
    # 用 ContextVar 而不是模块级全局变量，是因为后端并发处理请求，共享一个槽位会
    # 让一轮的改写查询泄漏到另一轮的工具调用里（见 agent.py 的说明）。
    token = _TURN.set(context)
    fragments: list[str] = []
    reported_tools: list[str] = []
    final_messages: list[Any] = []
    announced_answering = False
    try:
        # Two modes at once: "messages" gives per-token deltas for the live
        # answer, "updates" gives the final state so a model that streams no
        # chunks still produces an answer.
        # 中文：同时订阅两种流模式。"messages" 提供逐 token 增量，用于实时显示；
        # "updates" 提供每次节点执行后的状态，用来兜底——某些模型/网络条件下不产生
        # 任何增量块，此时仍能从最终状态里取到完整答案（见 _last_message_text）。
        async for mode, chunk in graph.astream(
            payload, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                message, metadata = chunk
                # 只转发 agent 节点产出的文本：当前图里只有它调用对话模型，其余
                # 节点（工具、检索）的中间产物不应该出现在回答里。保留这个按节点名
                # 的过滤，也让以后新增 LLM 节点时不会悄悄把它的输出混进回答。
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
                # updates 模式：chunk 是 {节点名: 该节点的状态更新}。
                # 只保留最后一个带 messages 的更新（即图的当前末端状态）。
                for update in (chunk or {}).values():
                    messages = (update or {}).get("messages")
                    if isinstance(messages, list) and messages:
                        final_messages = messages
                        retrieved = _retrieval_payload(messages)
                        if retrieved is not None:
                            # 检索结果只在真的调用过知识检索工具时出现，
                            # 前端据此渲染"引用了哪些文档"面板。
                            yield {"type": "retrieval", **retrieved}
                # 工具列表来自本轮上下文，可能在一轮里被多次追加（模型多轮调用工具），
                # 因此只在发生变化时才发事件，避免重复推送同样的列表。
                context_tools = list(context.tool_calls)
                if context_tools != reported_tools:
                    reported_tools = context_tools
                    yield {"type": "tools", "names": list(reported_tools)}
    except Exception as exc:  # noqa: BLE001 - the client must learn the turn failed
        # 这里刻意捕获所有异常：流已经开始发送，无法再改 HTTP 状态码，
        # 唯一的办法是把失败作为事件报出去。异常类型名一并带上，便于定位。
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return
    finally:
        # 无论成功失败都复位 ContextVar，避免污染同一任务后续的调用（也避免
        # 线程/任务复用时残留上一轮的上下文）。
        _TURN.reset(token)

    # 优先用流式增量拼出的答案；若一个增量都没有（模型没流式输出），
    # 退回图末端状态里的最后一条 AIMessage 文本。
    answer = "".join(fragments).strip() or _last_message_text(final_messages)
    if not answer:
        yield {"type": "error", "message": "模型未返回内容"}
        return

    # 到这里才落库：用户消息与助手消息成对写入，保证历史里没有半截轮次。
    store.append_turn(session_id, "user", question)
    store.append_turn(
        session_id,
        "assistant",
        answer,
        # 把改写后的检索查询与实际用到的工具名一起记进会话，
        # 便于事后复盘"这个问题为什么检索到这些文档"。
        retrieval_query=context.retrieval_query,
        tool_calls=list(context.tool_calls),
    )
    yield {
        "type": "done",
        "session_id": session_id,
        "answer": answer,
        "retrieval_query": context.retrieval_query,
        "tool_calls": list(context.tool_calls),
        # 保留两位小数：既够用又不会暴露无意义的时间精度。
        "seconds": round(time.perf_counter() - started, 2),
    }


def _chunk_text(chunk: Any) -> str:
    """Extract plain text from a streamed model chunk, ignoring reasoning parts.

    Qwen's reasoning models emit ``reasoning_content`` alongside the answer; only
    the answer should reach the user.

    中文说明:
        从流式消息块里取出"可以给用户看的纯文本"。

        * `content` 可能是字符串，也可能是内容块列表（多模态/富内容的统一表示）。
          列表情况下只拼接 type == "text" 的块，其余块（推理、工具调用等）丢弃。
        * 推理模型会在消息里附带 `reasoning_content`（思考过程）。本函数只读
          `content`，因此思考过程不会进入回答，也不会被存进会话历史。
        * 无法识别的形状返回空字符串，调用方据此跳过该块，而不是把 repr 塞给用户。
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

    中文说明:
        从图状态的消息列表里找出**最近一次**知识检索工具的返回结果。

        识别依据：消息 type 为 "tool"、名字为 "search_project_knowledge"。工具返回值
        在消息里可能是已经解析好的 dict，也可能是 JSON 字符串，因此字符串会尝试
        json.loads（解析失败即视为"没有检索结果"）。最后校验 kind == "retrieval"，
        确认这条工具消息确实是检索结果而不是别的工具输出。

        返回:
            含 retrieval_query / results / error 的 dict；没有检索结果时返回 None
            （调用方据此决定是否发送 retrieval 事件）。
    """
    # 反向遍历取"最后一条"：多轮工具调用时会有多条检索消息，界面应展示最新一次。
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
            # 检索失败时工具会把错误信息放在这里，一并转发给界面显示。
            "error": content.get("error"),
        }
    return None


def _last_message_text(messages: list[Any]) -> str:
    """Final answer text from the graph's end state.

    Used when the model produced no streamed chunks; without it a non-streaming
    response would be reported as "no content".

    中文说明:
        从图末端状态中取最后一条有内容的 AI 消息文本，作为"没有流式增量"时的兜底。

        这里用 `type(message).__name__ != "AIMessage"` 按类名判断而不是 isinstance：
        只要类名是 AIMessage 就算命中，不需要导入具体的消息类。代码里没有写明这个
        写法是否有版本兼容方面的考虑，此处不做推测。

        返回:
            找到的非空文本（已 strip）；没有任何 AI 文本时返回空字符串，
            调用方会据此返回"模型未返回内容"错误。
    """
    for message in reversed(messages or []):
        if type(message).__name__ != "AIMessage":
            continue
        text = _chunk_text(message)
        if text.strip():
            return text.strip()
    return ""
