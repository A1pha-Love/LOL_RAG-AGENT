"""HTTP backend: sessions plus streaming question answering over SSE.

    .\\.venv\\Scripts\\python.exe -m uvicorn apps.api:app --port 8000
    curl -N -X POST localhost:8000/api/sessions/ask -H "Content-Type: application/json" -d "{\\"question\\":\\"有多少场比赛？\\"}"

Why SSE rather than WebSocket: answers are one-directional and finite, so SSE
covers the need with plain HTTP, no handshake, and automatic proxy/HTTP support.
An answer takes many seconds, so it must stream — a non-streaming endpoint would
sit silent and look hung.

The app is deliberately thin: session storage and answer generation live in
lol_agent, which the CLI and the evaluator already exercise. This layer only
translates HTTP to those calls.
"""

# ============================================================================
# 中文说明
#
# 本文件是唯一的 HTTP 后端入口（FastAPI），职责只有三件事：
#   1. 会话 CRUD（新建/列表/详情/删除/置顶）；
#   2. 把一轮问答包装成 SSE 流（真正的生成逻辑在 lol_agent.streaming）；
#   3. 前端构建产物已经生成时，顺便用同一个端口把它托管出去（SPA 兜底路由）。
# 统计口径、检索链路、Agent 图都不在这里，也不应该在这里——保持这一层"薄"，
# 才能让 CLI、评测脚本和 HTTP 三条入口共用同一套逻辑。
#
# SSE 的字节格式（本项目踩过的坑，务必留意）:
#   服务端由 sse-starlette 负责分帧，它的 `DEFAULT_SEPARATOR` 是 **"\r\n"**，
#   因此一帧的结束标记是空行 `\r\n\r\n`，而不是很多人直觉里的 `\n\n`。
#   历史事故：前端解析器按 "\n\n" 切帧，于是永远找不到分帧边界，整条流被当成
#   一个事件，界面完全没有流式效果。修复后前端对 CR/LF/CRLF 三种行尾都做了兼容，
#   并留下 scripts/verify_sse_contract.py 直接检查真实服务端产出的字节，
#   以防后端或依赖升级后悄悄改掉行尾（SSE 规范本身允许 \r\n / \n / \r 三种）。
#
# ping 保活:
#   ask 里给 EventSourceResponse 传了 ping=5（秒）。模型"思考"期间可能好几秒不产生
#   任何字节，中间的反向代理/网关会把这当成空闲连接而断开。sse-starlette 会定期发送
#   一条注释行（`: ping - <时间戳>` 加空行）——它不携带 data，客户端的 EventSource
#   会忽略它，但足以让连接保持活跃。注意 ping 只是保活，不能替代 streaming.py 里
#   的 status 事件：前者对用户不可见，后者才会在界面上显示"正在思考"。
# ============================================================================

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 这里把 src/ 手动加进 sys.path：apps/ 与 src/ 是平级目录，直接
# `python -m uvicorn apps.api:app` 时 src 不在导入路径里，lol_agent 会 import 失败。
# 先判断是否已存在，避免重复插入；下面的 lol_agent 导入因此要加 noqa: E402
# （它出现在模块级语句之后，违反 E402 的"import 必须在文件顶部"规则）。
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent.conversation import SessionNotFound, SessionStore  # noqa: E402
from lol_agent.streaming import stream_turn  # noqa: E402

app = FastAPI(
    title="LoL Ranked Data Agent",
    description="本地英雄联盟历史排位赛数据的问答 Agent（多轮 + 流式）",
    version="0.1.0",
)

# The browser UI is served from a different origin during development, so CORS is
# opened to loopback only — never to the public internet.
# 中文：开发时前端由 vite dev server 提供（不同端口 = 不同 origin），所以需要放行
# 跨域。这里用正则把允许的来源限制为本机回环地址，任何外部域名都不被放行——
# 后端持有 API key 与会话数据，绝不能对公网开放。
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 进程级单例。用模块级变量 + 懒加载而不是在导入时创建：tests/test_api.py 的 client
# fixture 会直接把 api._STORE 换成指向临时数据库的 SessionStore（用完再还原），
# 这样测试不会污染真实会话库。若改成导入时创建，就没有这个替换点。
_STORE: SessionStore | None = None


def get_store() -> SessionStore:
    """The process-wide session store (created lazily so tests can override it).

    返回进程级共享的会话存储；首次调用时才创建（懒加载是为了让测试能替换它）。
    """
    global _STORE
    if _STORE is None:
        _STORE = SessionStore()
    return _STORE


# --------------------------------------------------------------------------
# request models
# --------------------------------------------------------------------------

# 下面三个 Pydantic 模型定义了请求体契约：字段类型与默认值由 FastAPI 校验，
# 校验失败会自动返回 422，不需要在视图里手写检查。
class CreateSessionRequest(BaseModel):
    title: str = Field(default="", description="可选，留空则用首个问题作为标题")


class PinSessionRequest(BaseModel):
    pinned: bool = Field(description="true 置顶，false 取消置顶")


class AskRequest(BaseModel):
    # min_length=1 把"空问题"挡在视图之前（streaming 里还有一层 strip 后的判空）。
    question: str = Field(min_length=1, description="用户问题")
    session_id: str | None = Field(
        default=None,
        description="会话 id；留空则自动新建会话，id 通过 done 事件返回",
    )


def redact(message: str) -> str:
    """把错误信息里可能出现的 API key 替换成占位符。

    参数:
        message: 原始错误文本。

    返回:
        脱敏后的文本；环境里没有 key 时原样返回。

    说明:
        错误信息里可能夹带着上游返回的内容，其中有可能出现 API key；而这些消息会
        通过 error 事件直接发到浏览器（进而出现在客户端控制台与用户的截图里），
        所以发送前统一把 key 替换成占位符。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    return message.replace(api_key, "[已隐藏]") if api_key else message


# --------------------------------------------------------------------------
# session routes
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness plus the two things that silently break answers.

    健康检查：进程活着 + 索引是否存在/是否过期 + 会话库路径。

    为什么要把"索引"也算进健康检查:
        索引缺失或过期时接口依然能返回 200，但知识检索会退化甚至答错，属于"静默失效"。
        把这两件事显式暴露出来，出问题时一眼就能定位。索引不存在时直接给出可执行的
        补救命令（scripts/index_knowledge.py），不需要再查文档。
    """
    # 在函数内部导入：knowledge 模块会连带加载向量库/嵌入依赖，放在这里可以让
    # 只关心会话接口的调用方（与测试）不必为它付出导入成本。
    from lol_agent.knowledge import INDEX_PATH, check_index_staleness

    staleness = check_index_staleness() if INDEX_PATH.is_file() else {
        "index_exists": False,
        "stale": True,
        "reason": "索引不存在，请先运行 scripts/index_knowledge.py",
    }
    return {
        "status": "ok",
        "index_exists": staleness["index_exists"],
        "index_stale": staleness["stale"],
        "index_reason": staleness.get("reason"),
        # 会话库路径也暴露出来：多实例/多用户排查"我的会话去哪了"时最有用。
        "session_db": str(get_store().path),
    }


@app.post("/api/sessions", status_code=201)
def create_session(body: CreateSessionRequest) -> dict[str, Any]:
    """新建会话，返回其 id（标题可为空，之后可由首个问题补上）。"""
    session_id = get_store().create_session(title=body.title.strip())
    return {"session_id": session_id, "title": body.title.strip()}


@app.get("/api/sessions")
def list_sessions(limit: int = 50) -> dict[str, Any]:
    """列出最近的会话。

    limit 被夹在 1..200：避免客户端传入极端值（例如 ?limit=100000）时一次性把
    整个会话库读出来变成超大响应。上限 200 的取值来源代码里没有说明，此处不推测。
    """
    return {"sessions": get_store().list_sessions(limit=max(1, min(limit, 200)))}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """取单个会话的元数据与全部轮次。

    会话不存在时返回 404 而不是空会话：两者的界面行为完全不同（前者提示"找不到"，
    后者会被当成一个真的空会话继续对话）。
    """
    try:
        session = get_store().get_session(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=f"找不到会话：{session_id}") from exc
    return {
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        # 手工组装响应体（而不是直接把 ORM/dataclass 对象丢出去）：只暴露想暴露的
        # 字段，也避免把内部结构变成事实上的对外契约。
        "turns": [
            {
                "role": turn.role,
                "content": turn.content,
                "created_at": turn.created_at,
                # 保留每一轮的检索查询与工具调用，界面可据此展示"这轮做了什么"。
                "retrieval_query": turn.retrieval_query,
                "tool_calls": list(turn.tool_calls),
            }
            for turn in session.turns
        ],
    }


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    """删除会话；不存在时返回 404。

    先 get 再 delete：删除接口对不存在的 id 直接返回成功会掩盖客户端传错 id 的问题，
    而"删了不存在的东西"和"真的删掉了"是两件不同的事。
    """
    try:
        get_store().get_session(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=f"找不到会话：{session_id}") from exc
    get_store().delete_session(session_id)


@app.put("/api/sessions/{session_id}/pin")
def pin_session(session_id: str, body: PinSessionRequest) -> dict[str, Any]:
    """Pin or unpin a conversation. Pinned ones sort above the rest.

    置顶/取消置顶会话；置顶的会话排在列表前面（排序规则在会话存储层实现）。
    """
    try:
        get_store().set_pinned(session_id, body.pinned)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=f"找不到会话：{session_id}") from exc
    return {"session_id": session_id, "pinned": body.pinned}


# --------------------------------------------------------------------------
# streaming answer route
# --------------------------------------------------------------------------

@app.post("/api/sessions/ask")
async def ask(body: AskRequest):
    """Answer a question, streaming the turn as Server-Sent Events.

    Each SSE message carries a JSON payload with a ``type`` field; see
    ``lol_agent.streaming`` for the sequence. An auto-created session id arrives
    in the ``rewrite`` and ``done`` events.

    中文说明:
        回答问题，并以 SSE 流式返回这一轮的全部事件。

        请求里的 session_id 为空时会**先创建会话**再开始流：这样客户端在 done 事件
        里总能拿到一个可继续使用的 id（多轮对话的前提）。注意创建动作发生在返回
        响应之前，因此它仍能影响 HTTP 状态码——流一旦开始就只能用 error 事件报错。

        SSE 的 data 是 JSON（ensure_ascii=False 保留中文原样，避免被转义成反斜杠加
        u 开头的码点序列、体积放大近一倍），event 名直接用事件的 type，便于前端按
        类型分派处理。
    """
    store = get_store()
    session_id = body.session_id or store.create_session()

    async def event_source():
        # 逐条把 streaming 的事件翻译成 sse-starlette 需要的 (event, data) 结构。
        async for event in stream_turn(body.question, session_id, store=store):
            if event.get("type") == "error":
                # 错误信息是最可能夹带上游请求细节的一类文本，发送前统一脱敏。
                event = {**event, "message": redact(str(event.get("message", "")))}
            # data 必须是字符串；这里自己 json.dumps 而不是交给 sse-starlette 处理，
            # 是为了确保中文不被转义（ensure_ascii=False）。
            yield {"event": event.get("type", "message"), "data": json.dumps(event, ensure_ascii=False)}

    # ping=5 keeps bytes flowing through proxies during the model's silent
    # "thinking" phase, which is otherwise several seconds of nothing.
    # 中文：每 5 秒发一条 SSE 注释行做保活，具体原因见文件开头"ping 保活"一节。
    return EventSourceResponse(event_source(), ping=5)


# --------------------------------------------------------------------------
# static frontend (optional)
# --------------------------------------------------------------------------

# When the frontend has been built, serve it from this same origin. That removes
# the dev proxy from the picture and means the whole app runs on one port. In
# development `vite dev` serves the UI and proxies /api here instead.
#
# 中文：前端已经构建（frontend/dist 存在）时，由同一个端口托管静态文件：没有构建
# 产物时这些路由**根本不会注册**，因此生产/开发两种形态可以共用同一份后端代码。
_DIST = PROJECT_ROOT / "frontend" / "dist"

if _DIST.is_dir():
    # 只挂载构建产物里的 assets 目录（HTML/CSS/JS）；其它文件走下面的兜底路由。
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """首页：返回前端入口 HTML。"""
        return FileResponse(_DIST / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str) -> FileResponse:
        """Serve built files, falling back to index.html for client-side routes.

        An unmatched `/api/...` path is answered with JSON 404 instead of index.html:
        returning HTML for an API URL hides typos and stale clients behind a 200, and
        makes a missing endpoint look like it succeeded.

        中文说明:
            单页应用（SPA）兜底路由。前端用的是客户端路由，用户直接访问 /sessions/abc
            这类地址时服务端并没有对应文件，必须回退到 index.html 让前端路由接手。

            为什么未匹配的 /api/... 必须返回 JSON 404 而不是 index.html:
                如果 API 路径也回退到 HTML，客户端会收到 200 + 一段 HTML，于是拼错
                的接口名、已经下线的旧接口都会"看起来成功了"，错误反而在解析 JSON 时
                才以更难懂的形式暴露出来。API 路径必须保持 JSON 语义，才能让调用方
                一眼看出"这个接口不存在"。

            路径穿越防护:
                `_DIST in candidate.parents` 保证被返回的文件确实位于构建目录内，
                形如 /../../etc/passwd 的路径解析后不在 _DIST 之下，会落到 index.html
                而不是把任意文件读出去。
        """
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"未知接口：/{path}")
        candidate = (_DIST / path).resolve()
        if _DIST in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        # 其余路径一律交给前端路由处理（404 由前端自己在页面内展示）。
        return FileResponse(_DIST / "index.html")
