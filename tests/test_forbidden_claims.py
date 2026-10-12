"""Tests for the forbidden-claim screen used by the Agent evaluation gate.

The screen is a heuristic, and its failure modes are asymmetric:

* **False positive** (flagging a compliant answer) makes the gate untrustworthy, which
  is how a real regression later gets waved through as "probably another false alarm".
  This actually happened: an answer that *quoted* the rule 「不允许自行评价关联强弱
  （例如说"强正向相关"）」 was flagged as making the claim.
* **False negative** (missing a real claim) defeats the point of the gate.

So both directions are pinned here, using the real wording that triggered the bug.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CASE = {"answer_must_not_claim": ["强正向相关", "首 Baron 导致胜利"]}


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location(
        "_evaluate_rag_for_tests", PROJECT_ROOT / "scripts" / "evaluate_rag.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._forbidden_claim_checks


def test_quoting_a_prohibition_is_not_a_claim(checker) -> None:
    """The exact wording that produced a false positive in the live evaluation."""
    answer = (
        "项目只允许报告统计关联，不允许声称因果，也不允许自行评价关联强弱"
        "（例如说“强正向相关”）。"
    )
    unnegated, negated = checker(CASE, answer)
    assert not unnegated, f"quoting a rule must not count as making the claim: {unnegated}"
    assert "强正向相关" in negated


def test_dictionary_style_rule_citation_is_not_a_claim(checker) -> None:
    answer = "依据指标字典：不得自行评价关联强弱（例如“强正向相关”）。"
    unnegated, _ = checker(CASE, answer)
    assert not unnegated


def test_a_plain_assertion_is_still_caught(checker) -> None:
    """The guard against weakening the screen: unwrapped claims must still fail."""
    answer = "取得首个 Baron 与最终获胜呈强正向相关，首 Baron 导致胜利。"
    unnegated, _ = checker(CASE, answer)
    assert set(unnegated) == {"强正向相关", "首 Baron 导致胜利"}


def test_quoting_while_asserting_is_still_caught(checker) -> None:
    """Quotes alone must not excuse a phrase when the sentence affirms it.

    This is why the fix requires a prohibition marker before the quote rather than
    treating any quoted phrase as innocuous.
    """
    answer = "所谓“强正向相关”是成立的，首 Baron 导致胜利。"
    unnegated, _ = checker(CASE, answer)
    assert "强正向相关" in unnegated
    assert "首 Baron 导致胜利" in unnegated


def test_explicit_denial_is_not_a_claim(checker) -> None:
    answer = "不能声称首 Baron 导致胜利，这并非强正向相关。"
    unnegated, _ = checker(CASE, answer)
    assert not unnegated


def test_prohibition_far_from_the_phrase_still_counts_as_a_quotation(checker) -> None:
    """Distance must not defeat the check, since rule citations are long sentences."""
    answer = (
        "我们的口径不允许自行评价关联强弱，这一点在指标字典中有明确规定，"
        "包括不得使用诸如“强正向相关”这样的措辞。"
    )
    unnegated, _ = checker(CASE, answer)
    assert not unnegated, f"a distant prohibition still governs the quote: {unnegated}"


def test_phrase_absent_from_the_answer_is_not_flagged(checker) -> None:
    unnegated, negated = checker(CASE, "取得一血的一方胜率为 59.2%。")
    assert not unnegated and not negated
