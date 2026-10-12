"""Run one evaluation case through the *production* turn path.

Both evaluation scripts used to call `build_agent().invoke(...)` directly and so never
established a `TurnContext`. That silently changed the code path under test:
`current_retrieval_query` falls back to the model's own `query` argument when no context
exists, whereas the CLI and the HTTP server always install one via `prepare_turn`, so
retrieval in production searches the *rewritten* query and ignores the model's phrasing.

The effect was that retrieval assertions were graded against a path users never hit, and
a case could be recorded as a failure (or a pass) for behaviour that does not occur in
the product. Sharing this helper keeps the two scripts consistent with each other and
with production.

Keep this in step with `prepare_turn` usage in `lol_agent.streaming` and
`lol_agent.agent.answer_question`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from lol_agent.agent import _TURN, build_agent, prepare_turn, session_messages


@dataclass
class EvalSession:
    """The minimal session shape `prepare_turn` and `session_messages` need.

    Each case runs in its own empty session: evaluation cases are independent
    questions, and carrying history between them would let one case's wording leak
    into another's rewrite.
    """

    session_id: str = "eval"
    turns: list[Any] = field(default_factory=list)


def invoke_case(question: str, *, temperature: float | None = None) -> list[Any]:
    """Run one question through the production turn path and return the messages.

    ``temperature`` is passed through to `build_agent`; leaving it None keeps the
    configured/default sampling, so callers that do not care are unaffected.
    """
    session = EvalSession()
    context = prepare_turn(session, question)
    token = _TURN.set(context)
    try:
        result = build_agent(temperature=temperature).invoke(
            {"messages": [*session_messages(session), HumanMessage(content=question)]}
        )
    finally:
        _TURN.reset(token)
    return result["messages"]


def invoke_case_tolerant(
    question: str, *, temperature: float | None = None
) -> tuple[list[Any], str]:
    """`invoke_case`, but a tool raising an error is reported instead of aborting.

    A tool error inside the graph propagates out of `invoke`, which would abort the
    whole evaluation run on the first occurrence. Whether the model passes valid
    arguments is itself something the evaluation should measure, so the failure is
    returned as a reason and scored as a failed case rather than killing the run.
    """
    try:
        return invoke_case(question, temperature=temperature), ""
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def tool_results(messages: list[Any], name: str) -> list[Any]:
    """Parsed payloads returned by one tool, in call order."""
    import json

    payloads: list[Any] = []
    for message in messages:
        if getattr(message, "type", None) != "tool" or getattr(message, "name", None) != name:
            continue
        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue
        payloads.append(content)
    return payloads


def retrieval_hits(messages: list[Any]) -> list[dict[str, Any]]:
    """All chunks returned by search_project_knowledge across the turn."""
    hits: list[dict[str, Any]] = []
    for payload in tool_results(messages, "search_project_knowledge"):
        results = payload if isinstance(payload, list) else payload.get("results", [])
        hits.extend(chunk for chunk in results or [] if isinstance(chunk, dict))
    return hits


def retrieval_queries_used(messages: list[Any]) -> list[str]:
    """Every query the retrieval tool actually searched, in order.

    Recorded alongside the hits because a retrieval miss cannot be diagnosed without
    knowing which query produced it. Diagnosing this by guesswork previously produced a
    wrong conclusion about why cases were failing, so the query is captured rather than
    inferred.
    """
    queries: list[str] = []
    for payload in tool_results(messages, "search_project_knowledge"):
        if not isinstance(payload, dict):
            continue
        query = payload.get("retrieval_query")
        if query and query not in queries:
            queries.append(query)
    return queries
