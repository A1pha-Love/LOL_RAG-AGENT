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


# --------------------------------------------------------------------------
# summoner spells
# --------------------------------------------------------------------------

@needs_dataset
def test_spell_stats_reports_every_spell_in_the_dataset() -> None:
    result = tools.get_spell_stats()
    assert result["total_spells"] == 9
    # Only nine spells exist, so the ranking cap (20) cannot be reached: the caller
    # gets all of them, and the count is clamped to what the data holds.
    assert len(result["results"]) == min(tools.DEFAULT_RANKING_LIMIT, result["total_spells"])
    assert len(result["results"]) == 9


@needs_dataset
def test_spell_can_be_addressed_by_id_or_by_chinese_name() -> None:
    by_id = tools.get_spell_stats("4")
    by_name = tools.get_spell_stats("闪现")
    assert by_id["results"] == by_name["results"]
    assert by_id["spell_id_filter"] == "4"
    assert len(by_id["results"]) == 1
    assert by_id["results"][0]["spell_name"] == "闪现"


@needs_dataset
def test_unrecognised_spell_returns_a_note_instead_of_a_silent_empty_list() -> None:
    """A silent empty list is indistinguishable from "this dataset has no data",
    so the caller is told the input was not understood."""
    result = tools.get_spell_stats("这不是技能")
    assert result["results"] == []
    assert result["spell_id_filter"] is None
    assert "未识别" in result["note"]


@needs_dataset
def test_spell_labels_come_from_the_mapping_file() -> None:
    result = tools.get_spell_stats(limit=9)
    names = {row["spell_id"]: row["spell_name"] for row in result["results"]}
    assert names["4"] == "闪现"
    assert names["11"] == "惩戒"
    assert names["21"] == "屏障"
    assert all(name for name in names.values()), "every spell must have a label"


@needs_dataset
def test_spell_ranking_is_sorted_for_every_key() -> None:
    for key in ("picked_teams", "pick_rate", "win_rate_when_picked"):
        rows = tools.get_spell_stats(sort_by=key, limit=9)["results"]
        # `picked_teams` is a plain int; the rate fields are numerator/denominator dicts.
        values = [row[key] if isinstance(row[key], int) else row[key]["value"] for row in rows]
        assert values == sorted(values, reverse=True), f"{key} is not sorted"


@needs_dataset
def test_spell_win_split_is_internally_consistent() -> None:
    result = tools.get_spell_win_split("闪现")
    with_side = result["with_spell"]
    without_side = result["without_spell"]
    # Both sides must cover every team appearance in the dataset.
    assert with_side["teams"] + without_side["teams"] == 51_053 * 2
    assert result["spell_name"] == "闪现"
    # The caveat must travel with the numbers: this is not a causal claim.
    assert "非受控实验" in result["caveat"]


@needs_dataset
def test_spell_win_split_rejects_an_unrecognised_spell() -> None:
    result = tools.get_spell_win_split("这不是技能")
    assert result["spell_id_filter"] is None
    assert "未识别" in result["note"]


def test_spell_sort_key_and_limit_validation() -> None:
    with pytest.raises(ValueError, match="sort_by"):
        tools.get_spell_stats(sort_by="kills")
    with pytest.raises(ValueError, match="limit"):
        tools.get_spell_stats(limit=0)


@needs_dataset
def test_disable_spells_from_the_mapping_are_not_invented() -> None:
    """The mapping lists retired spells (35, 36) the dataset never contains.

    Asking for one must report "not recognised" rather than returning a zero row,
    which would look like a real spell that is simply never picked.
    """
    result = tools.get_spell_stats("35")
    assert result["results"] == [], "a disabled spell must not produce a data row"


# --------------------------------------------------------------------------
# team objective averages
# --------------------------------------------------------------------------

@needs_dataset
def test_objective_averages_cover_every_objective() -> None:
    result = tools.get_team_objective_averages()
    assert result["unique_match_count"] == 51_053
    assert len(result["objectives"]) == len(tools.metrics.TEAM_OBJECTIVES)
    # Every objective must carry a human label, or the answer cannot name it.
    assert all(item["objective_label"] for item in result["objectives"])


@needs_dataset
def test_objective_can_be_addressed_by_column_or_chinese_name() -> None:
    by_column = tools.get_team_objective_averages("towerKills")
    assert by_column["objective_filter"] == "towerKills"
    assert len(by_column["objectives"]) == 1
    assert by_column["objectives"][0]["objective"] == "towerKills"

    for label, expected in (
        ("防御塔", "towerKills"),
        ("水晶", "inhibitorKills"),
        ("男爵", "baronKills"),
        ("小龙", "dragonKills"),
        ("峡谷先锋", "riftHeraldKills"),
    ):
        assert tools.get_team_objective_averages(label)["objective_filter"] == expected, label


@needs_dataset
def test_unrecognised_objective_lists_what_is_available() -> None:
    """An empty list alone is indistinguishable from "no data"; say what is valid."""
    result = tools.get_team_objective_averages("这不存在")
    assert result["objectives"] == []
    assert result["objective_filter"] is None
    assert "未识别" in result["note"]
    assert len(result["available"]) == len(tools.metrics.TEAM_OBJECTIVES)


@needs_dataset
def test_objective_averages_carry_the_cumulative_value_caveat() -> None:
    """These counts are end-of-match totals, not a timeline."""
    result = tools.get_team_objective_averages()
    assert "累计值" in result["note"]
