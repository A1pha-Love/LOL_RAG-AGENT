"""Tests for the Agent-behaviour gate's scoring and regression logic.

The gate is only useful if it fails for the right reasons. These tests pin its
decisions on synthetic payloads, so no model call or dataset is needed:

* every gate is enforced (a case cannot pass by satisfying only some of them);
* a case recorded as failing in the baseline does not block, so a check can be wired
  up while known gaps are being worked on;
* a case that previously passed and now fails is reported as a regression — the whole
  point of keeping a baseline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_module("_eval_agent_for_tests", "scripts/eval_agent.py")


def _payload(*, tools=("get_match_count",), answer="", retrieved=(), tool_results=None):
    return {
        "called_tools": list(tools),
        "tool_results": tool_results or {},
        "retrieved": [{"source": s, "heading_path": h} for s, h in retrieved],
        "answer": answer,
    }


BASE_CASE = {
    "id": "synthetic",
    "question": "问题",
    "expected_tools": ["get_match_count"],
    # A distinctive heading: a single character like 「节」 matches many paths by
    # coincidence, which makes the assertion meaningless rather than strict.
    "expected_retrieval": [{"source": "kbase/a.md", "heading": "版本边界"}],
}

MATCHING_PATH = "映射与版本边界 > 版本边界（必须说明）"


def test_a_fully_compliant_case_passes(gate) -> None:
    helpers = gate._load_scoring_helpers()
    outcome = gate.score_case(
        BASE_CASE,
        _payload(retrieved=[("kbase/a.md", MATCHING_PATH)]),
        helpers,
    )
    assert outcome.passed, outcome.gates


def test_missing_expected_tool_fails(gate) -> None:
    helpers = gate._load_scoring_helpers()
    outcome = gate.score_case(BASE_CASE, _payload(tools=("get_ban_stats",)), helpers)
    assert not outcome.passed
    assert outcome.gates["tools"] is False


def test_calling_a_forbidden_tool_fails(gate) -> None:
    helpers = gate._load_scoring_helpers()
    case = {**BASE_CASE, "forbidden_tools": ["get_ban_stats"]}
    outcome = gate.score_case(case, _payload(tools=("get_match_count", "get_ban_stats")), helpers)
    assert not outcome.passed
    assert outcome.gates["no_forbidden_tools"] is False


def test_retrieval_must_hit_the_expected_heading(gate) -> None:
    helpers = gate._load_scoring_helpers()
    outcome = gate.score_case(
        BASE_CASE, _payload(retrieved=[("kbase/a.md", "文档 > 完全不相关的小节")]), helpers
    )
    assert not outcome.passed
    assert outcome.gates["retrieval"] is False
    assert outcome.missing_retrieval


def test_retrieval_matching_is_substring_based_by_design(gate) -> None:
    """A heading is part of a path, so matching is `in`, not equality.

    This also documents the flip side: a very short expected heading can match by
    coincidence, so expectations should name a distinctive section rather than a
    single character. Pinning the `in` behaviour prevents "tightening" it to equality,
    which would break every real assertion (the index stores "文档 > 节 > 小节").
    """
    helpers = gate._load_scoring_helpers()
    outcome = gate.score_case(
        BASE_CASE, _payload(retrieved=[("kbase/a.md", "文档 > 小节 > 具体条目")]), helpers
    )
    assert outcome.gates["retrieval"] is False, "单字期望值不匹配更长的路径"

    distinctive = {
        **BASE_CASE,
        "expected_retrieval": [{"source": "kbase/a.md", "heading": "版本边界"}],
    }
    nested = gate.score_case(
        distinctive,
        _payload(retrieved=[("kbase/a.md", "映射与版本边界 > 版本边界（必须说明）")]),
        helpers,
    )
    assert nested.gates["retrieval"] is True, "路径中的子串必须能匹配"


def test_retrieval_must_come_from_the_expected_source(gate) -> None:
    helpers = gate._load_scoring_helpers()
    outcome = gate.score_case(
        BASE_CASE, _payload(retrieved=[("kbase/other.md", MATCHING_PATH)]), helpers
    )
    assert outcome.gates["retrieval"] is False


def test_forbidden_claim_fails_even_when_everything_else_passes(gate) -> None:
    """The gate must not be satisfiable by tool use alone."""
    helpers = gate._load_scoring_helpers()
    case = {**BASE_CASE, "answer_must_not_claim": ["强正向相关"]}
    outcome = gate.score_case(
        case,
        _payload(answer="两者呈强正向相关。", retrieved=[("kbase/a.md", MATCHING_PATH)]),
        helpers,
    )
    assert not outcome.passed
    assert outcome.gates["no_forbidden_claims"] is False


def test_numeric_assertion_is_enforced(gate) -> None:
    helpers = gate._load_scoring_helpers()
    case = {
        **BASE_CASE,
        "numeric_assertions": [
            {"tool": "get_match_count", "numerator": 51053, "denominator": 1, "value": 51053}
        ],
    }
    good = gate.score_case(
        case,
        _payload(
            retrieved=[("kbase/a.md", MATCHING_PATH)],
            tool_results={"get_match_count": [{"numerator": 51053, "denominator": 1, "value": 51053}]},
        ),
        helpers,
    )
    assert good.passed, good.gates

    bad = gate.score_case(
        case,
        _payload(
            retrieved=[("kbase/a.md", MATCHING_PATH)],
            tool_results={"get_match_count": [{"numerator": 1, "denominator": 1, "value": 1}]},
        ),
        helpers,
    )
    assert bad.gates["numeric"] is False


def test_cached_payloads_survive_a_json_round_trip(gate) -> None:
    """Responses are cached as JSON, so scoring must work on the replayed form.

    Tool results arrive as LangChain message content and are stored as plain JSON; if
    scoring depended on the live objects, a cached run would score differently from a
    live one and the gate would be inconsistent.
    """
    import json

    helpers = gate._load_scoring_helpers()
    payload = _payload(
        retrieved=[("kbase/a.md", MATCHING_PATH)],
        tool_results={"get_match_count": [{"numerator": 51053, "denominator": 1, "value": 51053}]},
    )
    replayed = json.loads(json.dumps(payload))
    case = {
        **BASE_CASE,
        "numeric_assertions": [
            {"tool": "get_match_count", "numerator": 51053, "denominator": 1, "value": 51053}
        ],
    }
    live = gate.score_case(case, payload, helpers)
    cached = gate.score_case(case, replayed, helpers)
    assert live.gates == cached.gates
    assert live.passed and cached.passed


def test_question_fingerprint_changes_when_the_question_changes(gate) -> None:
    """Editing a case must invalidate only its own cached response."""
    first = gate.question_fingerprint({"id": "a", "question": "原问题"})
    same = gate.question_fingerprint({"id": "a", "question": "原问题"})
    edited = gate.question_fingerprint({"id": "a", "question": "改过的问题"})
    other = gate.question_fingerprint({"id": "b", "question": "原问题"})
    assert first == same
    assert first != edited
    assert first != other
