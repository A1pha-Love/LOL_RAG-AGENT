"""Deterministic metrics for the LoL ranked games CSV.

All rates follow the project metric definitions in ``kbase/`` — see
``kbase/project_metrics_core.md`` and ``kbase/project_metrics_ban_event.md``.
(The original ``docs/metric_dictionary.md`` moved into the knowledge base.)
Functions return explicit numerators and denominators so an Agent can explain
and verify each result.

Champion ids are validated as positive integers: the dataset writes ``-1`` into
unused ban slots, and counting that sentinel would invent a champion.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

Row = Mapping[str, str]

FIRST_EVENTS = (
    "firstBlood",
    "firstTower",
    "firstInhibitor",
    "firstBaron",
    "firstDragon",
    "firstRiftHerald",
)


def load_matches(csv_path: str | Path) -> list[dict[str, str]]:
    """Load CSV rows and deduplicate identical gameIds; reject conflicting IDs."""
    matches: dict[str, dict[str, str]] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            game_id = (row.get("gameId") or "").strip()
            if not game_id:
                raise ValueError(f"Missing gameId at CSV line {line_number}")
            normalized = {key: (value or "").strip() for key, value in row.items()}
            previous = matches.get(game_id)
            if previous is not None and previous != normalized:
                raise ValueError(f"Conflicting records for gameId={game_id}")
            matches[game_id] = normalized
    return list(matches.values())


def _unique_rows(rows: Iterable[Row]) -> list[Row]:
    """Deduplicate rows by gameId and guard against conflicting duplicate rows."""
    unique: dict[str, Row] = {}
    for row in rows:
        game_id = str(row.get("gameId", "")).strip()
        if not game_id:
            raise ValueError("Every match row must have a non-empty gameId")
        previous = unique.get(game_id)
        if previous is not None and dict(previous) != dict(row):
            raise ValueError(f"Conflicting records for gameId={game_id}")
        unique[game_id] = row
    return list(unique.values())


def _result(metric: str, numerator: int | float, denominator: int, unit: str) -> dict:
    value = None if denominator == 0 else numerator / denominator
    if value is not None and unit == "matches":
        value = int(value)
    if value is not None and unit == "percent":
        value *= 100
    return {
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "unit": unit,
    }


def match_count(rows: Iterable[Row]) -> dict:
    matches = _unique_rows(rows)
    return _result("unique_match_count", len(matches), 1, "matches")


def _is_champion_id(value: str) -> bool:
    """True for a real champion id.

    The dataset writes ``-1`` into unused ban slots; that is a sentinel meaning
    "no champion", not a champion. Counting it would create a phantom champion
    that ranks in ban tables, so only positive integers qualify.
    """
    text = value.strip()
    return text.isdigit() and int(text) > 0


def _champions_for_team(row: Row, team: int) -> set[str]:
    prefix = f"t{team}_champ"
    return {
        str(row.get(f"{prefix}{slot}id", "")).strip()
        for slot in range(1, 6)
        if _is_champion_id(str(row.get(f"{prefix}{slot}id", "")))
    }


def champion_stats(rows: Iterable[Row]) -> dict[str, dict]:
    """Return pick and win counts/rates for each champion ID, per unique match."""
    matches = _unique_rows(rows)
    picked_games: dict[str, int] = defaultdict(int)
    won_games: dict[str, int] = defaultdict(int)
    for row in matches:
        teams = {1: _champions_for_team(row, 1), 2: _champions_for_team(row, 2)}
        picked = teams[1] | teams[2]
        winner = str(row.get("winner", "")).strip()
        for champion_id in picked:
            picked_games[champion_id] += 1
            if winner in ("1", "2") and champion_id in teams[int(winner)]:
                won_games[champion_id] += 1
    total = len(matches)
    return {
        champion_id: {
            "champion_id": champion_id,
            "appearances": _result("champion_appearance_matches", games, total, "percent"),
            "wins": won_games[champion_id],
            "win_rate": _result("champion_win_rate", won_games[champion_id], games, "percent"),
        }
        for champion_id, games in sorted(picked_games.items(), key=lambda item: int(item[0]))
    }


def _spells_for_team(row: Row, team: int) -> set[str]:
    """The distinct summoner spells a team fielded in one match.

    Each of the 5 players has two spell slots (``_sum1`` / ``_sum2``), so a team
    contributes up to 10 slots but far fewer *distinct* spells — Flash alone
    appears in roughly 70% of slots. Deduplicating per team is what makes the
    count below a "spell was present" measure rather than a slot count; without it
    a team with Flash twice would be counted twice.

    The two slots are deliberately treated as unordered: the dataset gives no
    evidence that ``_sum1`` is the primary spell, so ranking them would invent
    information.
    """
    prefix = f"t{team}_champ"
    spells: set[str] = set()
    for slot in range(1, 6):
        for position in (1, 2):
            value = str(row.get(f"{prefix}{slot}_sum{position}", "")).strip()
            # Same positive-integer rule as champion ids: anything else (empty,
            # -1) is a missing slot, not a spell.
            if value.isdigit() and int(value) > 0:
                spells.add(value)
    return spells


def spell_stats(rows: Iterable[Row]) -> dict[str, dict]:
    """Return pick and win counts/rates for each summoner spell.

    Denominator for ``pick_rate`` is **team appearances** (2 per unique match),
    because a spell is fielded by a team, not by a match. Denominator for
    ``win_rate_when_picked`` is the number of teams that fielded that spell.

    This is observational data, not a controlled experiment: a spell's win rate
    also reflects the champions and roles that take it. Callers must not read it
    as a causal effect.
    """
    matches = _unique_rows(rows)
    picked_teams: dict[str, int] = defaultdict(int)
    won_teams: dict[str, int] = defaultdict(int)
    appeared_in_match: dict[str, int] = defaultdict(int)
    for row in matches:
        winner = str(row.get("winner", "")).strip()
        for team in (1, 2):
            spells = _spells_for_team(row, team)
            for spell in spells:
                appeared_in_match[spell] += 1
                picked_teams[spell] += 1
                if winner == str(team):
                    won_teams[spell] += 1

    team_appearances = len(matches) * 2
    return {
        spell: {
            "spell_id": spell,
            "picked_teams": picked_teams[spell],
            "matches_with_spell": appeared_in_match[spell],
            "wins": won_teams[spell],
            "pick_rate": _result("spell_pick_rate", picked_teams[spell], team_appearances, "percent"),
            "win_rate_when_picked": _result(
                "spell_win_rate_when_picked", won_teams[spell], picked_teams[spell], "percent"
            ),
        }
        for spell in sorted(picked_teams, key=lambda value: int(value))
    }


def spell_win_split(rows: Iterable[Row], spell_id: str) -> dict:
    """Compare team win rates with and without a specific summoner spell.

    Returns both sides so a difference can be stated without implying causation:
    ``with`` counts teams that fielded the spell, ``without`` counts teams that did
    not, and both are measured over the same matches.
    """
    wanted = str(spell_id).strip()
    if not (wanted.isdigit() and int(wanted) > 0):
        raise ValueError("spell_id must be a positive integer")

    matches = _unique_rows(rows)
    with_teams = with_wins = 0
    without_teams = without_wins = 0
    for row in matches:
        winner = str(row.get("winner", "")).strip()
        for team in (1, 2):
            won = winner == str(team)
            if wanted in _spells_for_team(row, team):
                with_teams += 1
                with_wins += int(won)
            else:
                without_teams += 1
                without_wins += int(won)

    return {
        "spell_id": wanted,
        "with_spell": {
            "teams": with_teams,
            "wins": with_wins,
            "win_rate": _result("team_win_rate_with_spell", with_wins, with_teams, "percent"),
        },
        "without_spell": {
            "teams": without_teams,
            "wins": without_wins,
            "win_rate": _result("team_win_rate_without_spell", without_wins, without_teams, "percent"),
        },
    }


def ban_stats(rows: Iterable[Row]) -> dict[str, dict]:
    """Return champion ban incidence: matches with >=1 ban / all matches.

    Uses the same positive-id rule as picks, so the ``-1`` sentinel that fills
    unused ban slots never appears as a champion.
    """
    matches = _unique_rows(rows)
    banned_games: dict[str, int] = defaultdict(int)
    for row in matches:
        banned = {
            str(row.get(f"t{team}_ban{slot}", "")).strip()
            for team in (1, 2)
            for slot in range(1, 6)
            if _is_champion_id(str(row.get(f"t{team}_ban{slot}", "")))
        }
        for champion_id in banned:
            banned_games[champion_id] += 1
    return {
        champion_id: {
            "champion_id": champion_id,
            "banned_matches": banned_games[champion_id],
            "ban_rate": _result("champion_ban_match_rate", banned_games[champion_id], len(matches), "percent"),
        }
        for champion_id in sorted(banned_games, key=lambda value: int(value))
    }


def first_event_win_rate(rows: Iterable[Row], event: str) -> dict:
    """Win rate of the team recorded as achieving an event first; excludes event=0."""
    if event not in FIRST_EVENTS:
        raise ValueError(f"Unsupported first-event field: {event}")
    numerator = denominator = 0
    for row in _unique_rows(rows):
        event_team = str(row.get(event, "")).strip()
        winner = str(row.get("winner", "")).strip()
        if event_team in ("1", "2") and winner in ("1", "2"):
            denominator += 1
            numerator += event_team == winner
    return _result(f"{event}_team_win_rate", numerator, denominator, "percent")


def first_event_occurrence_rate(rows: Iterable[Row], event: str) -> dict:
    """Share of valid matches where the requested first event occurred."""
    if event not in FIRST_EVENTS:
        raise ValueError(f"Unsupported first-event field: {event}")
    matches = _unique_rows(rows)
    valid = [row for row in matches if str(row.get(event, "")).strip() in ("0", "1", "2")]
    occurred = sum(str(row.get(event, "")).strip() in ("1", "2") for row in valid)
    return _result(f"{event}_occurrence_rate", occurred, len(valid), "percent")


def average_game_duration(rows: Iterable[Row]) -> dict:
    """Return mean duration in seconds over positive, parseable gameDuration values."""
    durations: list[int] = []
    for row in _unique_rows(rows):
        try:
            duration = int(row.get("gameDuration", ""))
        except (TypeError, ValueError):
            continue
        if duration > 0:
            durations.append(duration)
    return _result("average_game_duration_seconds", sum(durations), len(durations), "seconds")
