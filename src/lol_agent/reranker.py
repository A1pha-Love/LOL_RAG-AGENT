"""Cross-encoder reranking for retrieved chunks, via DashScope `qwen3-rerank`.

Retrieval with embeddings answers "which chunks look similar"; a reranker scores
each candidate *against the query jointly*, which is what separates a broad
document-level chunk from the focused sub-section that actually answers the
question.

The endpoint is workspace-scoped and differs from the OpenAI-compatible path
used for chat and embeddings:

    POST https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-api/v1/reranks

Note the docs' warning: `relevance_score` is a **relative** score for one
request, so it orders candidates within a query but is not comparable across
queries. It must not be used as a global no-answer cutoff.

Reranking is best-effort: any failure raises `RerankUnavailable`, and callers
fall back to the embedding order rather than failing the request.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from .config import rerank_url, require_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_RERANK_MODEL = "qwen3-rerank"
# The reranker reads the question, so the instruction steers what "relevant"
# means. This is the documented default for question-answering retrieval.
QA_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."
REQUEST_TIMEOUT_SECONDS = 60


class RerankUnavailable(RuntimeError):
    """The rerank endpoint could not be used (missing config, network, or API error)."""


def rerank_model() -> str:
    return os.getenv("QWEN_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip() or DEFAULT_RERANK_MODEL


def rerank_enabled() -> bool:
    """Reranking is on unless explicitly disabled."""
    return os.getenv("RERANK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _endpoint() -> str:
    return rerank_url()


def rerank(
    query: str,
    documents: list[str],
    *,
    top_n: int | None = None,
    instruct: str = QA_INSTRUCT,
) -> list[tuple[int, float]]:
    """Return [(original_index, relevance_score)] ordered best-first.

    Raises `RerankUnavailable` on any failure so the caller can fall back.
    """
    if not documents:
        return []
    try:
        api_key = require_api_key()
    except RuntimeError as exc:
        raise RerankUnavailable(str(exc)) from exc
    payload: dict[str, object] = {
        "model": rerank_model(),
        "query": query,
        "documents": documents,
        "instruct": instruct,
    }
    if top_n is not None:
        payload["top_n"] = min(top_n, len(documents))

    request = urllib.request.Request(
        _endpoint(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise RerankUnavailable(f"HTTP {exc.code}: {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RerankUnavailable(f"{type(exc).__name__}: {exc}") from exc

    # qwen3-rerank puts `results` at the top level; gte-rerank-v2 nests it.
    results = body.get("results")
    if results is None and isinstance(body.get("output"), dict):
        results = body["output"].get("results")
    if results is None:
        raise RerankUnavailable(f"unexpected response shape: {sorted(body)[:6]}")

    ranked: list[tuple[int, float]] = []
    for item in results:
        try:
            ranked.append((int(item["index"]), float(item["relevance_score"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankUnavailable(f"malformed result entry: {item}") from exc
    ranked.sort(key=lambda pair: -pair[1])
    return ranked
