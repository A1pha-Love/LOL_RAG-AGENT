"""Resolve `expected_retrieval` entries into concrete (source, heading) targets.

Shared by `eval_agent.py` and `verify_retrieval_offline.py`. Both grade the same
assertions against the same cases, so a rule that lives in only one of them would let
the two disagree — one reporting a miss the other considers satisfied.

An entry may be:

    {"source": "kbase/a.md", "heading": "节"}          one required section
    {"alternatives": [{...}, {...}]}                    any one satisfies it
    [{...}, {...}]                                      shorthand for alternatives

Alternatives exist because a question about a fact is often answered equally well by
more than one section. Requiring one particular section grades retrieval on an
arbitrary choice instead of on whether the answer was available.
"""

from __future__ import annotations

from typing import Any


def retrieval_targets(entry: Any) -> list[dict[str, str]]:
    """Normalize one expected_retrieval entry into the acceptable targets."""
    if isinstance(entry, list):
        return [target for target in entry if isinstance(target, dict)]
    alternatives = entry.get("alternatives") if isinstance(entry, dict) else None
    if alternatives:
        return [target for target in alternatives if isinstance(target, dict)]
    return [entry] if isinstance(entry, dict) else []


def describe_expectation(entry: Any) -> str:
    """Human-readable label for an expectation, including its alternatives."""
    labels = [
        f"{target.get('source')} :: {target.get('heading')}" for target in retrieval_targets(entry)
    ]
    return " 或 ".join(labels) if len(labels) > 1 else (labels[0] if labels else "<无目标>")


def is_satisfied(entry: Any, retrieved: list[dict[str, Any]]) -> bool:
    """Whether any acceptable target for this expectation appears in `retrieved`."""
    targets = retrieval_targets(entry)
    return any(
        chunk.get("source") == target.get("source")
        and target.get("heading", "") in (chunk.get("heading_path") or "")
        for target in targets
        for chunk in retrieved
    )


def best_rank(entry: Any, ranked: list[dict[str, Any]]) -> int | None:
    """1-based rank of the best-matching acceptable target, or None if absent."""
    for rank, chunk in enumerate(ranked, start=1):
        if any(
            chunk.get("source") == target.get("source")
            and target.get("heading", "") in (chunk.get("heading_path") or "")
            for target in retrieval_targets(entry)
        ):
            return rank
    return None
