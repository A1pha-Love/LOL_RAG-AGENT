"""Unit tests for the deterministic metrics layer.

These run fully offline: no API key, no network. The expected values were
measured against the project dataset (51,053 unique matches) and are the same
numbers evals/cases.jsonl uses as reference answers, so a regression here means
either the data changed or a metric definition drifted.

The suite skips the data-dependent tests when ShuJuYuan/games.csv is absent
(it is gitignored, so a fresh clone needs the dataset before running them).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import metrics  # noqa: E402

CSV_PATH = PROJECT_ROOT / "ShuJuYuan" / "games.csv"

# Ground truth from the audited dataset.
RAW_ROW_COUNT = 51_490
UNIQUE_MATCH_COUNT = 51_053
DUPLICATE_ROW_COUNT = RAW_ROW_COUNT - UNIQUE_MATCH_COUNT
COLUMN_COUNT = 61
TOTAL_DURATION_SECONDS = 93_533_198


needs_dataset = pytest.mark.skipif(
    not CSV_PATH.is_file(), reason=f"dataset not present: {CSV_PATH}"
)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    if not CSV_PATH.is_file():
        pytest.skip(f"dataset not present: {CSV_PATH}")
    return metrics.load_matches(CSV_PATH)


# --------------------------------------------------------------------------
# loading and deduplication
# --------------------------------------------------------------------------

@needs_dataset
def test_dataset_shape_is_unchanged(rows: list[dict[str, str]]) -> None:
    assert len(rows) == UNIQUE_MATCH_COUNT
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        raw_rows = sum(1 for _ in reader)
    assert len(columns) == COLUMN_COUNT
    assert raw_rows == RAW_ROW_COUNT


@needs_dataset
def test_load_matches_deduplicates_by_game_id(rows: list[dict[str, str]]) -> None:
    ids = [row["gameId"] for row in rows]
    assert len(ids) == len(set(ids)), "load_matches returned duplicate gameIds"


def test_identical_duplicate_rows_collapse_to_one() -> None:
    duplicate = {"gameId": "g1", "winner": "1", "gameDuration": "1800"}
    assert len(metrics._unique_rows([dict(duplicate), dict(duplicate)])) == 1


def test_conflicting_duplicate_rows_raise_instead_of_silently_picking() -> None:
    """A conflicting duplicate means the source is broken, so it must not pass."""
    first = {"gameId": "g1", "winner": "1"}
    second = {"gameId": "g1", "winner": "2"}
    with pytest.raises(ValueError, match="Conflicting records"):
        metrics._unique_rows([first, second])


def test_missing_game_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="gameId"):
        metrics._unique_rows([{"gameId": "  ", "winner": "1"}])


# --------------------------------------------------------------------------
# result envelope
# --------------------------------------------------------------------------

def test_zero_denominator_reports_no_sample_not_zero_percent() -> None:
    """Returning 0% for an empty denominator would be a fabricated statistic."""
    result = metrics._result("some_rate", 0, 0, "percent")
    assert result["value"] is None
    assert result["numerator"] == 0
    assert result["denominator"] == 0


def test_match_unit_result_is_an_integer() -> None:
    result = metrics._result("m", 7, 1, "matches")
    assert result["value"] == 7
    assert isinstance(result["value"], int)


def test_percent_unit_result_is_scaled_by_100() -> None:
    result = metrics._result("r", 1, 4, "percent")
    assert result["value"] == pytest.approx(25.0)


# --------------------------------------------------------------------------
# match count and duration
# --------------------------------------------------------------------------

@needs_dataset
def test_match_count(rows: list[dict[str, str]]) -> None:
    result = metrics.match_count(rows)
    assert result["metric"] == "unique_match_count"
    assert result["numerator"] == UNIQUE_MATCH_COUNT
    assert result["denominator"] == 1
    assert result["value"] == UNIQUE_MATCH_COUNT
    assert result["unit"] == "matches"
    assert result["numerator"] - UNIQUE_MATCH_COUNT == 0
    assert DUPLICATE_ROW_COUNT == 437


@needs_dataset
def test_average_game_duration(rows: list[dict[str, str]]) -> None:
    result = metrics.average_game_duration(rows)
    assert result["metric"] == "average_game_duration_seconds"
    assert result["numerator"] == TOTAL_DURATION_SECONDS
    assert result["denominator"] == UNIQUE_MATCH_COUNT
    assert result["value"] == pytest.approx(1832.08, abs=0.01)
    assert result["unit"] == "seconds"


def test_average_duration_excludes_unparseable_and_non_positive() -> None:
    result = metrics.average_game_duration(
        [
            {"gameId": "a", "gameDuration": "100"},
            {"gameId": "b", "gameDuration": "0"},
            {"gameId": "c", "gameDuration": "-5"},
            {"gameId": "d", "gameDuration": "not-a-number"},
            {"gameId": "e", "gameDuration": ""},
            {"gameId": "f", "gameDuration": "300"},
        ]
    )
    assert result["numerator"] == 400
    assert result["denominator"] == 2
    assert result["value"] == pytest.approx(200.0)


# --------------------------------------------------------------------------
# champion statistics
# --------------------------------------------------------------------------

@needs_dataset
def test_champion_stats_for_a_known_champion(rows: list[dict[str, str]]) -> None:
    stats = metrics.champion_stats(rows)
    # Champion 18 is the most-picked champion in this dataset.
    champion = stats["18"]
    assert champion["appearances"]["numerator"] == 12_885
    assert champion["appearances"]["denominator"] == UNIQUE_MATCH_COUNT
    assert champion["appearances"]["unit"] == "percent"
    assert champion["wins"] == 6_662
    # Win rate divides by appearances, not by total matches.
    assert champion["win_rate"]["denominator"] == champion["appearances"]["numerator"]
    assert champion["win_rate"]["numerator"] == champion["wins"]


@needs_dataset
def test_champion_appearance_rate_matches_manual_denominator(rows: list[dict[str, str]]) -> None:
    stats = metrics.champion_stats(rows)
    champion = stats["18"]
    expected_percent = champion["appearances"]["numerator"] / UNIQUE_MATCH_COUNT * 100
    assert champion["appearances"]["value"] == pytest.approx(expected_percent)


def test_champion_counted_once_when_both_teams_pick_it() -> None:
    """The same champion on both teams is one appearance, not two."""
    row = {
        "gameId": "g1",
        "winner": "1",
        "t1_champ1id": "18",
        "t2_champ1id": "18",
    }
    stats = metrics.champion_stats([row])
    assert stats["18"]["appearances"]["numerator"] == 1
    # team1 won and holds champion 18, so the win is credited.
    assert stats["18"]["wins"] == 1


def test_win_is_not_credited_when_the_holder_lost() -> None:
    row = {"gameId": "g1", "winner": "2", "t1_champ1id": "18"}
    stats = metrics.champion_stats([row])
    assert stats["18"]["appearances"]["numerator"] == 1
    assert stats["18"]["wins"] == 0
    assert stats["18"]["win_rate"]["value"] == pytest.approx(0.0)


def test_champion_slots_holding_zero_or_empty_are_ignored() -> None:
    row = {"gameId": "g1", "winner": "1", "t1_champ1id": "0", "t1_champ2id": "", "t1_champ3id": "18"}
    stats = metrics.champion_stats([row])
    assert set(stats) == {"18"}


@needs_dataset
def test_season_id_is_constant_so_season_slicing_is_meaningless(rows: list[dict[str, str]]) -> None:
    assert {row["seasonId"] for row in rows} == {"9"}


# --------------------------------------------------------------------------
# ban statistics
# --------------------------------------------------------------------------

@needs_dataset
def test_ban_stats_for_the_most_banned_champion(rows: list[dict[str, str]]) -> None:
    stats = metrics.ban_stats(rows)
    champion = stats["157"]
    assert champion["banned_matches"] == 27_146
    assert champion["ban_rate"]["denominator"] == UNIQUE_MATCH_COUNT
    assert champion["ban_rate"]["numerator"] == champion["banned_matches"]


@needs_dataset
def test_ban_sentinel_minus_one_is_not_a_champion(rows: list[dict[str, str]]) -> None:
    """-1 marks an empty ban slot; counting it as a champion would be wrong.

    The dataset really does contain -1 in ban slots, so this asserts both that
    the sentinel is present in the source and that it never becomes a champion.
    """
    sentinel_count = sum(
        1
        for row in rows
        for team in (1, 2)
        for slot in range(1, 6)
        if row.get(f"t{team}_ban{slot}") == "-1"
    )
    assert sentinel_count > 0, "expected -1 ban slots in the dataset"
    assert "-1" not in metrics.ban_stats(rows)


def test_same_champion_banned_twice_in_one_match_counts_once() -> None:
    row = {"gameId": "g1", "winner": "1", "t1_ban1": "157", "t2_ban1": "157"}
    stats = metrics.ban_stats([row])
    assert stats["157"]["banned_matches"] == 1


def test_ban_slots_with_zero_and_empty_are_ignored() -> None:
    row = {"gameId": "g1", "winner": "1", "t1_ban1": "0", "t1_ban2": "", "t2_ban1": "157"}
    stats = metrics.ban_stats([row])
    assert set(stats) == {"157"}


# --------------------------------------------------------------------------
# first-event metrics
# --------------------------------------------------------------------------

@needs_dataset
def test_first_blood_occurrence_rate(rows: list[dict[str, str]]) -> None:
    result = metrics.first_event_occurrence_rate(rows, "firstBlood")
    assert result["numerator"] == 50_499
    assert result["denominator"] == UNIQUE_MATCH_COUNT
    assert result["value"] == pytest.approx(98.91, abs=0.01)


@needs_dataset
def test_first_blood_conditional_win_rate(rows: list[dict[str, str]]) -> None:
    result = metrics.first_event_win_rate(rows, "firstBlood")
    assert result["numerator"] == 29_870
    # The conditional denominator excludes matches where the event never happened.
    assert result["denominator"] == 50_499
    assert result["denominator"] < UNIQUE_MATCH_COUNT
    assert result["value"] == pytest.approx(59.15, abs=0.01)


@needs_dataset
def test_first_baron_metrics(rows: list[dict[str, str]]) -> None:
    win_rate = metrics.first_event_win_rate(rows, "firstBaron")
    assert win_rate["numerator"] == 24_978
    assert win_rate["denominator"] == 30_962
    occurrence = metrics.first_event_occurrence_rate(rows, "firstBaron")
    assert occurrence["numerator"] == 30_962
    assert occurrence["denominator"] == UNIQUE_MATCH_COUNT


def test_event_never_occurring_yields_no_sample() -> None:
    """occurrence with all-zero events must not report 0%."""
    rows = [{"gameId": "g1", "winner": "1", "firstBlood": "0"}]
    occurrence = metrics.first_event_occurrence_rate(rows, "firstBlood")
    assert occurrence["numerator"] == 0
    assert occurrence["denominator"] == 1
    assert occurrence["value"] == pytest.approx(0.0)
    # And the conditional win rate has a genuinely empty denominator.
    win_rate = metrics.first_event_win_rate(rows, "firstBlood")
    assert win_rate["denominator"] == 0
    assert win_rate["value"] is None


def test_unknown_first_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported first-event"):
        metrics.first_event_win_rate([{"gameId": "g1"}], "firstNexus")


def test_every_declared_first_event_is_supported() -> None:
    row = {"gameId": "g1", "winner": "1"}
    for event in metrics.FIRST_EVENTS:
        payload = dict(row, **{event: "1"})
        assert metrics.first_event_occurrence_rate([payload], event)["numerator"] == 1


# --------------------------------------------------------------------------
# cross-checks against the raw file
# --------------------------------------------------------------------------

@needs_dataset
def test_duplicate_rows_are_byte_identical_in_the_source(rows: list[dict[str, str]]) -> None:
    """The audit claims duplicates are field-for-field identical; verify it."""
    seen: dict[str, dict[str, str]] = {}
    conflicts = 0
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            normalized = {key: (value or "").strip() for key, value in row.items()}
            game_id = normalized["gameId"]
            if game_id in seen and seen[game_id] != normalized:
                conflicts += 1
            seen[game_id] = normalized
    assert conflicts == 0
    assert len(seen) == UNIQUE_MATCH_COUNT


@needs_dataset
@pytest.mark.parametrize(
    ("deduplicate", "expected"),
    [
        # Metrics work on the deduplicated view: 51,053 matches.
        (True, Counter({"1": 25_857, "2": 25_196})),
        # The audit report's table describes raw rows: 51,490 rows.
        (False, Counter({"1": 26_077, "2": 25_413})),
    ],
)
def test_winner_column_only_holds_one_or_two(deduplicate: bool, expected: Counter) -> None:
    """Both views must sum to their own total, and differ by the 437 duplicates.

    Guards the "raw rows vs deduplicated matches" distinction the metric
    dictionary warns about: reporting one total with the other's breakdown is a
    real error mode.
    """
    counter: Counter = Counter()
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            counter[row["winner"].strip()] += 1
    if deduplicate:
        seen: dict[str, str] = {}
        for row in metrics.load_matches(CSV_PATH):
            seen[row["gameId"]] = row["winner"]
        counter = Counter(seen.values())

    assert counter == expected
    assert set(counter) == {"1", "2"}
    assert sum(counter.values()) == (UNIQUE_MATCH_COUNT if deduplicate else RAW_ROW_COUNT)
