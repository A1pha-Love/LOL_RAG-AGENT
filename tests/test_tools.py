"""Tests for the agent-facing tool layer.

Focus: the ranking-query cap. An uncapped ranking query returned all 138
champions (~55,000 characters, roughly 13x the top-10 payload), which the model
must read before answering — a real latency cost for no benefit, since a ranking
question is answered from the top few rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import tools  # noqa: E402

CSV_PATH = PROJECT_ROOT / "ShuJuYuan" / "games.csv"

needs_dataset = pytest.mark.skipif(
    not CSV_PATH.is_file(), reason=f"dataset not present: {CSV_PATH}"
)


# --------------------------------------------------------------------------
# ranking cap
# --------------------------------------------------------------------------

@needs_dataset
def test_uncapped_ranking_query_is_capped_by_default() -> None:
    result = tools.get_champion_stats()
    assert len(result["results"]) == tools.DEFAULT_RANKING_LIMIT
    # The total is still reported, so the answer can say "of 138 champions".
    assert result["total_champions"] > tools.DEFAULT_RANKING_LIMIT


@needs_dataset
def test_default_payload_is_far_smaller_than_the_full_table() -> None:
    """The point of the cap: a ranking answer needs a few rows, not the table."""
    capped = json.dumps(tools.get_champion_stats(), ensure_ascii=False)
    uncapped = json.dumps(tools.get_champion_stats(limit=1000), ensure_ascii=False)
    assert len(capped) < len(uncapped) / 4, (
        f"capped={len(capped)} uncapped={len(uncapped)}; the cap is not reducing the payload"
    )


@needs_dataset
def test_explicit_limit_still_returns_more() -> None:
    """The cap must not remove capability: an explicit limit still wins."""
    result = tools.get_champion_stats(limit=100)
    assert len(result["results"]) == 100
    assert result["limit"] == 100


@needs_dataset
def test_ban_ranking_is_capped_too() -> None:
    result = tools.get_ban_stats()
    assert len(result["results"]) == tools.DEFAULT_RANKING_LIMIT
    assert result["total_champions"] > tools.DEFAULT_RANKING_LIMIT


@needs_dataset
def test_cap_does_not_apply_when_a_specific_champion_is_asked_for() -> None:
    """A single-champion query returns one row regardless of the ranking cap."""
    result = tools.get_champion_stats(champion_id="157")
    assert len(result["results"]) == 1
    assert result["results"][0]["champion_id"] == "157"


@needs_dataset
def test_sorting_is_unaffected_by_the_cap() -> None:
    """The cap takes the head of a correctly sorted list, not an arbitrary slice."""
    by_win_rate = tools.get_champion_stats(sort_by="win_rate")["results"]
    values = [row["win_rate"]["value"] for row in by_win_rate]
    assert values == sorted(values, reverse=True)

    by_appearances = tools.get_champion_stats(sort_by="appearances")["results"]
    appearances = [row["appearances"]["value"] for row in by_appearances]
    assert appearances == sorted(appearances, reverse=True)


@needs_dataset
def test_ban_sentinel_never_appears_in_ranked_results() -> None:
    """-1 marks an empty ban slot. It must not rank as a champion anywhere."""
    ids = [row["champion_id"] for row in tools.get_ban_stats(limit=1000)["results"]]
    assert "-1" not in ids


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------

def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit"):
        tools.get_champion_stats(limit=0)
    with pytest.raises(ValueError, match="limit"):
        tools.get_champion_stats(limit=-3)


def test_boolean_limit_is_rejected() -> None:
    """True is an int in Python; accepting it would silently mean limit=1."""
    with pytest.raises(ValueError, match="limit"):
        tools.get_champion_stats(limit=True)


def test_unknown_sort_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="sort_by"):
        tools.get_champion_stats(sort_by="kills")
    with pytest.raises(ValueError, match="sort_by"):
        tools.get_ban_stats(sort_by="kills")
