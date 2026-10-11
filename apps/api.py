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
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STORE: SessionStore | None = None


def get_store() -> SessionStore:
    """The process-wide session store (created lazily so tests can override it)."""
    global _STORE
    if _STORE is None:
        _STORE = SessionStore()
    return _STORE


# --------------------------------------------------------------------------
# request models
# --------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: str = Field(default="", description="可选，留空则用首个问题作为标题")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    session_id: str | None = Field(
        default=None,
        description="会话 id；留空则自动新建会话，id 通过 done 事件返回",
    )


def redact(message: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    return message.replace(api_key, "[已隐藏]") if api_key else message


# --------------------------------------------------------------------------
# session routes
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness plus the two things that silently break answers."""
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
        "session_db": str(get_store().path),
    }


@app.post("/api/sessions", status_code=201)
def create_session(body: CreateSessionRequest) -> dict[str, Any]:
    session_id = get_store().create_session(title=body.title.strip())
    return {"session_id": session_id, "title": body.title.strip()}


@app.get("/api/sessions")
def list_sessions(limit: int = 50) -> dict[str, Any]:
    return {"sessions": get_store().list_sessions(limit=max(1, min(limit, 200)))}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    try:
        session = get_store().get_session(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=f"找不到会话：{session_id}") from exc
    return {
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "turns": [
            {
                "role": turn.role,
                "content": turn.content,
                "created_at": turn.created_at,
                "retrieval_query": turn.retrieval_query,
                "tool_calls": list(turn.tool_calls),
            }
            for turn in session.turns
        ],
    }


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    try:
        get_store().get_session(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=f"找不到会话：{session_id}") from exc
    get_store().delete_session(session_id)


# --------------------------------------------------------------------------
# streaming answer route
# --------------------------------------------------------------------------

@app.post("/api/sessions/ask")
async def ask(body: AskRequest):
    """Answer a question, streaming the turn as Server-Sent Events.

    Each SSE message carries a JSON payload with a ``type`` field; see
    ``lol_agent.streaming`` for the sequence. An auto-created session id arrives
    in the ``rewrite`` and ``done`` events.
    """
    store = get_store()
    session_id = body.session_id or store.create_session()

    async def event_source():
        async for event in stream_turn(body.question, session_id, store=store):
            if event.get("type") == "error":
                event = {**event, "message": redact(str(event.get("message", "")))}
            yield {"event": event.get("type", "message"), "data": json.dumps(event, ensure_ascii=False)}

    # ping=5 keeps bytes flowing through proxies during the model's silent
    # "thinking" phase, which is otherwise several seconds of nothing.
    return EventSourceResponse(event_source(), ping=5)


# --------------------------------------------------------------------------
# static frontend (optional)
# --------------------------------------------------------------------------

# When the frontend has been built, serve it from this same origin. That removes
# the dev proxy from the picture and means the whole app runs on one port. In
# development `vite dev` serves the UI and proxies /api here instead.
_DIST = PROJECT_ROOT / "frontend" / "dist"

if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_DIST / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str) -> FileResponse:
        """Serve built files, falling back to index.html for client-side routes."""
        candidate = (_DIST / path).resolve()
        if _DIST in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
