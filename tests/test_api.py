"""Tests for the HTTP layer.

Offline and model-free: the session routes are exercised through the app, and
the answer route is checked only for its error path and content type. Generating
a real answer needs the Chat API, so that is verified live by
scripts/check_api_stream.py instead of here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

fastapi = pytest.importorskip("fastapi", reason="backend dependencies not installed")
from fastapi.testclient import TestClient  # noqa: E402

from lol_agent.conversation import SessionStore  # noqa: E402
from apps import api  # noqa: E402


@pytest.fixture()
def client(temp_db):
    """A client bound to a throwaway session database."""
    store = SessionStore(temp_db)
    previous = api._STORE
    api._STORE = store
    try:
        with TestClient(api.app) as test_client:
            yield test_client
    finally:
        api._STORE = previous


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------

def test_health_reports_index_state(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # The index must be reported honestly: a stale index silently degrades answers.
    assert isinstance(body["index_exists"], bool)
    assert isinstance(body["index_stale"], bool)
    assert body["index_reason"]


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

def test_create_session_returns_an_id(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"title": ""})
    assert response.status_code == 201
    assert response.json()["session_id"]


def test_created_session_appears_in_the_list(client: TestClient) -> None:
    created = client.post("/api/sessions", json={}).json()["session_id"]
    listed = client.get("/api/sessions").json()["sessions"]
    assert [item["session_id"] for item in listed] == [created]


def test_get_session_returns_its_turns(client: TestClient) -> None:
    session_id = client.post("/api/sessions", json={}).json()["session_id"]
    api.get_store().append_turn(session_id, "user", "有多少场比赛？")
    api.get_store().append_turn(
        session_id, "assistant", "51,053 场。", retrieval_query="去重后的总对局数", tool_calls=["get_match_count"]
    )

    body = client.get(f"/api/sessions/{session_id}").json()
    assert [turn["role"] for turn in body["turns"]] == ["user", "assistant"]
    assert body["turns"][1]["retrieval_query"] == "去重后的总对局数"
    assert body["turns"][1]["tool_calls"] == ["get_match_count"]
    assert body["title"] == "有多少场比赛？"


def test_unknown_session_is_404(client: TestClient) -> None:
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.delete("/api/sessions/nope").status_code == 404


def test_delete_removes_the_session(client: TestClient) -> None:
    session_id = client.post("/api/sessions", json={}).json()["session_id"]
    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get("/api/sessions").json()["sessions"] == []


def test_session_list_limit_is_clamped(client: TestClient) -> None:
    for _ in range(3):
        client.post("/api/sessions", json={})
    assert len(client.get("/api/sessions?limit=2").json()["sessions"]) == 2
    # An absurd limit must not be passed through to the query.
    assert client.get("/api/sessions?limit=99999").status_code == 200


# --------------------------------------------------------------------------
# ask route: validation and stream framing (no model call)
# --------------------------------------------------------------------------

def test_ask_rejects_an_empty_question(client: TestClient) -> None:
    response = client.post("/api/sessions/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_on_unknown_session_streams_an_error_event(client: TestClient) -> None:
    with client.stream(
        "POST", "/api/sessions/ask", json={"question": "有多少场比赛？", "session_id": "missing"}
    ) as response:
        assert response.status_code == 200, "an SSE stream starts before the error is known"
        assert response.headers["content-type"].startswith("text/event-stream")
        payload = "".join(response.iter_text())

    assert "event: error" in payload
    assert "找不到会话" in payload


def test_ask_creates_a_session_when_none_is_given(client: TestClient) -> None:
    """The stream must begin with a valid event and create the session.

    A client that sends no session id must still learn one; the session is
    created before streaming starts, so this is checkable from the first event
    without paying for a full completion.
    """
    before = client.get("/api/sessions").json()["sessions"]
    with client.stream("POST", "/api/sessions/ask", json={"question": "有多少场比赛？"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            first = json.loads(line[5:].strip())
            # The turn opens with a status event (so the UI can show progress)
            # and the rewrite event carrying what retrieval will search.
            assert first["type"] in {"status", "rewrite", "error"}, first
            break

    after = client.get("/api/sessions").json()["sessions"]
    assert len(after) == len(before) + 1, "an implicit session should have been created"
