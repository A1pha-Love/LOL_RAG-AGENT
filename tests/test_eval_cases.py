"""Execute the deterministic reference cases in evals/cases.jsonl.

Why this exists: those cases hold the project's reference answers, but nothing
executed them. `scripts/evaluate_rag.py` reads only `rag_cases.jsonl`, so the ten
statistical assertions were never verified against the code — `evals/README.md`
requires exactly that ("参考值应在评测实现前独立复核，不能把 Agent 自己的回答当成
标准答案"), and a drift between a metric and its reference value would otherwise go
unnoticed until someone spent API quota re-running the Agent.

These cases need no model call: each one names a tool and its arguments, so the
whole file runs offline and belongs in the pre-commit suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import tools  # noqa: E402

CASES_PATH = PROJECT_ROOT / "evals" / "cases.jsonl"
CSV_PATH = PROJECT_ROOT / "ShuJuYuan" / "games.csv"

# Percentage and duration values are derived, so an exact float compare is wrong.
TOLERANCE = 1e-6

needs_dataset = pytest.mark.skipif(
    not CSV_PATH.is_file(), reason=f"dataset not present: {CSV_PATH}"
)


def _load_cases() -> list[dict[str, Any]]:
    if not CASES_PATH.is_file():
        pytest.skip(f"eval cases not present: {CASES_PATH}")
    return [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


CASES = _load_cases()
CASE_IDS = [case["id"] for case in CASES]


def _run_case(case: dict[str, Any]) -> Any:
    tool = getattr(tools, case["tool"], None)
    assert tool is not None, f"case {case['id']} names a tool that does not exist: {case['tool']}"
    return tool(**case["arguments"])


def _assert_scalar(actual: Any, expected: Any, path: str) -> None:
    """Compare one expected value, honouring the derived-value formulas."""
    if isinstance(expected, str):
        if expected in {"numerator/denominator*100", "numerator/denominator"}:
            # A formula over the block's own numerator/denominator, not a literal.
            return
        # Anything else (unit names such as "percent"/"seconds"/"matches") is a
        # literal that must match exactly.
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
        return
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=TOLERANCE, abs=TOLERANCE), path
    else:
        assert actual == expected, path


def _check_expected_block(expected: dict[str, Any], actual: dict[str, Any], case_id: str) -> None:
    """Verify every key the reference declares.

    Reference fields do not always match the tool's own nesting: a list tool wraps
    each row in ``results``, and the occurrence cases name ``occurred`` /
    ``not_occurred`` where the tool reports ``numerator`` / ``denominator``. Both
    are the same numbers under different names, so the mapping is made explicit here
    rather than left to a coincidence.
    """
    # A single-champion query returns one row: compare the reference against it.
    rows = actual.get("results")
    if isinstance(rows, list) and len(rows) == 1 and "results" not in expected:
        actual = rows[0]

    def occurrence_parts(block: dict[str, Any]) -> tuple[Any, Any]:
        """`occurred` / `not_occurred` from a numerator/denominator block."""
        numerator = block.get("numerator")
        denominator = block.get("denominator")
        if numerator is None or denominator is None:
            return None, None
        return numerator, denominator - numerator

    # Reference field name -> the name the tool actually uses. Kept explicit so a
    # genuine rename in the tool shows up as a failure rather than being absorbed.
    aliases = {"appearance_rate": "appearances"}

    for key, value in expected.items():
        actual_key = aliases.get(key, key)
        if key == "occurred":
            got, _ = occurrence_parts(actual)
            assert got == value, f"{case_id}.occurred (got {got})"
            continue
        if key == "not_occurred":
            _, got_absent = occurrence_parts(actual)
            assert got_absent == value, f"{case_id}.not_occurred (got {got_absent})"
            continue
        if key == "ordered_ids":
            # Champion and spell tools name their identifier column differently.
            ids = [
                str(row.get("champion_id", row.get("spell_id")))
                for row in actual["results"]
            ]
            assert ids == [str(item) for item in value], f"{case_id}.ordered_ids"
            continue
        if key == "appearances" and isinstance(value, list):
            got = [row["appearances"]["numerator"] for row in actual["results"]]
            assert got == value, f"{case_id}.appearances"
            continue
        if key == "banned_matches" and isinstance(value, list):
            got = [row["banned_matches"] for row in actual["results"]]
            assert got == value, f"{case_id}.banned_matches"
            continue
        if key == "picked_teams" and isinstance(value, list):
            got = [row["picked_teams"] for row in actual["results"]]
            assert got == value, f"{case_id}.picked_teams"
            continue
        if key == "picked_teams" and isinstance(value, int):
            assert actual.get("picked_teams") == value, f"{case_id}.picked_teams"
            continue
        if key == "appearances" and isinstance(value, int):
            assert actual["appearances"]["numerator"] == value, f"{case_id}.appearances"
            continue
        if key == "wins" and isinstance(value, int):
            assert actual.get("wins") == value, f"{case_id}.wins"
            continue
        if isinstance(value, dict):
            nested = actual.get(actual_key)
            assert isinstance(nested, dict), (
                f"{case_id}.{key}: expected a metric block at {actual_key!r}"
            )
            _check_expected_block(value, nested, f"{case_id}.{key}")
            continue
        if actual_key in actual:
            _assert_scalar(actual[actual_key], value, f"{case_id}.{key}")
            continue
        raise AssertionError(f"{case_id}: reference field {key!r} is not produced by the tool")


# --------------------------------------------------------------------------
# the cases themselves
# --------------------------------------------------------------------------

def test_the_case_file_is_not_empty() -> None:
    assert CASES, "evals/cases.jsonl has no cases"


def test_every_case_is_well_formed() -> None:
    required = {"id", "question", "tool", "arguments", "expected"}
    seen: set[str] = set()
    for case in CASES:
        missing = required - set(case)
        assert not missing, f"{case.get('id')} is missing {sorted(missing)}"
        assert case["id"] not in seen, f"duplicate case id: {case['id']}"
        seen.add(case["id"])
        assert case["question"].strip(), case["id"]
        assert hasattr(tools, case["tool"]), f"{case['id']}: unknown tool {case['tool']}"
        assert isinstance(case["arguments"], dict), case["id"]


def test_every_tool_that_matters_has_at_least_one_case() -> None:
    """A tool with no reference case is uncovered by the CI regression."""
    covered = {case["tool"] for case in CASES}
    expected_cover = {
        "get_match_count",
        "get_dataset_overview",
        "get_champion_stats",
        "get_ban_stats",
        "get_first_event_win_rate",
        "get_first_event_occurrence_rate",
        "get_average_game_duration",
        "get_spell_stats",
        "get_spell_win_split",
    }
    assert expected_cover <= covered, f"uncovered tools: {sorted(expected_cover - covered)}"


def test_every_deterministic_tool_has_a_case() -> None:
    """Compare against the real tool registry instead of a hand-kept list.

    A hand-kept list silently rots when a tool is added; reading the registry means a
    new tool fails this test until it gets a reference case. Tools that cannot be
    pinned down offline (search_project_knowledge needs the embedding API) are
    excluded explicitly, so the exclusion is visible rather than accidental.
    """
    import inspect

    needs_api = {"search_project_knowledge"}
    registered = {
        name
        for name, value in vars(tools).items()
        if inspect.isfunction(value) and name.startswith("get_")
    }
    uncovered = registered - {case["tool"] for case in CASES} - needs_api
    assert not uncovered, (
        f"these tools have no reference case in evals/cases.jsonl: {sorted(uncovered)}. "
        "Add one, or list it in needs_api with a reason."
    )


@needs_dataset
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_reference_case_matches_the_tools(case: dict[str, Any]) -> None:
    actual = _run_case(case)
    _check_expected_block(case["expected"], actual, case["id"])


@needs_dataset
def test_derived_percentages_are_consistent_with_their_own_parts() -> None:
    """The `value` formula is a claim about the data, so verify it, not just the parts.

    A metric that reports the right numerator and denominator but computes the
    wrong ratio would pass a parts-only check.
    """
    checked = 0
    for case in CASES:
        actual = _run_case(case)
        for block in _walk_metric_blocks(actual):
            if block.get("unit") != "percent":
                continue
            expected = block["numerator"] / block["denominator"] * 100
            assert block["value"] == pytest.approx(expected, rel=TOLERANCE), case["id"]
            checked += 1
        # Duration uses a plain ratio.
        if case["expected"].get("unit") == "seconds":
            assert actual["value"] == pytest.approx(
                actual["numerator"] / actual["denominator"], rel=TOLERANCE
            )
            checked += 1
    assert checked >= 8, f"only {checked} derived values were checked"


def _walk_metric_blocks(value: Any):
    """Yield every dict that looks like a numerator/denominator/value metric."""
    if isinstance(value, dict):
        if {"numerator", "denominator", "value"} <= set(value):
            yield value
        for item in value.values():
            yield from _walk_metric_blocks(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_metric_blocks(item)
