"""Agent-facing, deterministic tools over the local ranked-match CSV."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import csv
import json
from pathlib import Path
from typing import Any

from . import metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "ShuJuYuan" / "games.csv"
CHAMPION_MAPPING_PATH = PROJECT_ROOT / "docs" / "champion_mapping_7_17_1.json"
SPELL_MAPPING_PATH = PROJECT_ROOT / "docs" / "mapping_summoner_spells.json"

TOOL_DESCRIPTIONS = {
    "get_match_count": "Count distinct matches after deduplicating identical gameId rows.",
    "get_dataset_overview": "Return deterministic CSV row/column counts, unique game count, and duplicate row/group counts.",
    "get_champion_stats": "Get champion pick appearances and win rates by numeric champion ID or exact Chinese/English champion name; optionally filter one champion or return a sorted top N.",
    "get_ban_stats": "Get champion ban incidence by match; optionally filter by numeric champion ID or exact Chinese/English champion name, or return a sorted top N.",
    "get_first_event_win_rate": "Get the win rate among matches where a team achieved the requested first event.",
    "get_first_event_occurrence_rate": "Get the occurrence rate of a first event, including matches where it did not occur.",
    "get_average_game_duration": "Get the average positive, parseable game duration in seconds.",
    "get_spell_stats": "Get summoner-spell pick rates and win rates by spell ID or exact Chinese name; optionally filter one spell or return a sorted top N.",
    "get_spell_win_split": "Compare team win rates with versus without a specific summoner spell, over the same matches.",
}


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime_ns: int, size: int) -> tuple[dict[str, str], ...]:
    # File metadata is part of the cache key, so replacing the CSV refreshes the data.
    return tuple(metrics.load_matches(path))


def _rows(csv_path: str | Path | None = None) -> tuple[dict[str, str], ...]:
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Match CSV not found: {path}")
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=1)
def _champion_maps() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load the pinned 7.17.1 champion labels and exact-match name aliases."""
    data = json.loads(CHAMPION_MAPPING_PATH.read_text(encoding="utf-8"))
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, str] = {}
    for champion in data["champions"]:
        champion_id = str(champion["id"])
        by_id[champion_id] = champion
        for label in (champion["name"], champion["title"], champion["english_id"], *(champion.get("aliases") or [])):
            normalized = str(label).strip().casefold()
            existing = by_name.get(normalized)
            if existing is not None and existing != champion_id:
                raise ValueError(f"Duplicate champion name alias in mapping: {label}")
            by_name[normalized] = champion_id
    return by_id, by_name


def _resolve_champion_id(value: str | int) -> str | None:
    """Resolve a numeric ID or an exact Chinese/English champion label."""
    token = str(value).strip()
    by_id, by_name = _champion_maps()
    return token if token in by_id else by_name.get(token.casefold())


def _add_champion_labels(results: list[dict[str, Any]]) -> None:
    by_id, _ = _champion_maps()
    for item in results:
        champion = by_id.get(str(item["champion_id"]))
        item["champion_name"] = champion["name"] if champion else None
        item["champion_title"] = champion["title"] if champion else None
        item["champion_english_id"] = champion["english_id"] if champion else None


@lru_cache(maxsize=1)
def _spell_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Load the pinned spell labels, plus exact-match aliases.

    The mapping file nests the lookup under ``id_to_name`` alongside provenance
    metadata, so the ids must be read from that key rather than the top level.
    """
    data = json.loads(SPELL_MAPPING_PATH.read_text(encoding="utf-8"))
    id_to_name = data.get("id_to_name")
    if not isinstance(id_to_name, dict):
        raise ValueError(f"Mapping file has no id_to_name table: {SPELL_MAPPING_PATH}")
    by_id = {str(key): str(value) for key, value in id_to_name.items()}
    by_name = {label.strip().casefold(): spell_id for spell_id, label in by_id.items()}
    return by_id, by_name


def _resolve_spell_id(value: str | int) -> str | None:
    """Resolve a numeric spell ID or an exact Chinese spell name."""
    token = str(value).strip()
    by_id, by_name = _spell_maps()
    if token in by_id:
        return token
    resolved = by_name.get(token.casefold())
    if resolved is None:
        return None
    # The mapping also lists retired/disabled spells (35, 36) that the dataset
    # never contains. Rejecting them here keeps "not in this dataset" distinct from
    # "not a spell", which would otherwise both look like an empty result.
    return resolved


def _add_spell_labels(results: list[dict[str, Any]]) -> None:
    by_id, _ = _spell_maps()
    for item in results:
        item["spell_name"] = by_id.get(str(item["spell_id"]))


@lru_cache(maxsize=4)
def _load_dataset_overview(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    """Count raw rows, columns, unique matches, and duplicate ID groups."""
    row_count = 0
    game_id_counts: Counter[str] = Counter()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            game_id = (row.get("gameId") or "").strip()
            if not game_id:
                raise ValueError(f"Missing gameId at CSV row {row_count + 1}")
            game_id_counts[game_id] += 1

    # Reuse the canonical loader to ensure duplicate gameIds have identical rows.
    unique_match_count = len(metrics.load_matches(path))
    duplicate_groups = sum(count > 1 for count in game_id_counts.values())
    return {
        "source_file": Path(path).name,
        "raw_row_count": row_count,
        "column_count": len(columns),
        "unique_match_count": unique_match_count,
        "duplicate_row_count": row_count - unique_match_count,
        "duplicate_game_id_groups": duplicate_groups,
        "duplicate_records_identical": True,
    }


def get_dataset_overview(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Return deterministic file structure and deduplication counts."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Match CSV not found: {path}")
    stat = path.stat()
    result = _load_dataset_overview(str(path), stat.st_mtime_ns, stat.st_size)
    if path.is_relative_to(PROJECT_ROOT):
        result["source_file"] = path.relative_to(PROJECT_ROOT).as_posix()
    return result


def _validate_limit(limit: int | None) -> None:
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer")


# A ranking question is answered from the top few rows, so returning the whole
# champion table is pure waste: all 138 rows serialise to ~55,000 characters
# (~13x the top-10 payload), which the model must read before it can answer and
# which measurably delays the first token. Callers that genuinely want the full
# table can pass an explicit larger limit.
DEFAULT_RANKING_LIMIT = 20


def get_match_count(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Count unique matches."""
    return metrics.match_count(_rows(csv_path))


def get_champion_stats(
    champion_id: str | int | None = None,
    sort_by: str = "appearances",
    limit: int | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return champion appearance and win statistics, optionally filtered/sorted.

    When no champion is named, results are ranked and capped at
    ``DEFAULT_RANKING_LIMIT`` unless an explicit ``limit`` is given.
    """
    _validate_limit(limit)
    if sort_by not in {"appearances", "win_rate"}:
        raise ValueError("sort_by must be 'appearances' or 'win_rate'")
    effective_limit = limit if limit is not None else DEFAULT_RANKING_LIMIT
    all_stats = metrics.champion_stats(_rows(csv_path))
    requested_id = _resolve_champion_id(champion_id) if champion_id is not None else None
    if champion_id is not None:
        selected = [all_stats[requested_id]] if requested_id in all_stats else []
    else:
        selected = list(all_stats.values())
        metric_name = "appearances" if sort_by == "appearances" else "win_rate"
        selected.sort(
            key=lambda item: (
                -(item[metric_name]["value"] if item[metric_name]["value"] is not None else -1),
                int(item["champion_id"]),
            )
        )
        selected = selected[:effective_limit]
    _add_champion_labels(selected)
    return {
        "metric": "champion_stats",
        "sort_by": sort_by,
        "limit": effective_limit,
        "champion_query": None if champion_id is None else str(champion_id),
        "champion_id_filter": requested_id,
        "mapping_source": "Riot Data Dragon 7.17.1 (zh_CN)",
        "total_champions": len(all_stats),
        "results": selected,
    }


def get_ban_stats(
    champion_id: str | int | None = None,
    sort_by: str = "banned_matches",
    limit: int | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return ban match counts/rates, optionally filtered/sorted.

    When no champion is named, results are ranked and capped at
    ``DEFAULT_RANKING_LIMIT`` unless an explicit ``limit`` is given.
    """
    _validate_limit(limit)
    if sort_by not in {"banned_matches", "ban_rate"}:
        raise ValueError("sort_by must be 'banned_matches' or 'ban_rate'")
    effective_limit = limit if limit is not None else DEFAULT_RANKING_LIMIT
    all_stats = metrics.ban_stats(_rows(csv_path))
    requested_id = _resolve_champion_id(champion_id) if champion_id is not None else None
    if champion_id is not None:
        selected = [all_stats[requested_id]] if requested_id in all_stats else []
    else:
        metric_name = "banned_matches" if sort_by == "banned_matches" else "ban_rate"
        selected = list(all_stats.values())
        selected.sort(
            key=lambda item: (
                -(item[metric_name]["value"] if isinstance(item[metric_name], dict) else item[metric_name]),
                int(item["champion_id"]),
            )
        )
        selected = selected[:effective_limit]
    _add_champion_labels(selected)
    return {
        "metric": "ban_stats",
        "sort_by": sort_by,
        "limit": effective_limit,
        "total_champions": len(all_stats),
        "champion_query": None if champion_id is None else str(champion_id),
        "champion_id_filter": requested_id,
        "mapping_source": "Riot Data Dragon 7.17.1 (zh_CN)",
        "results": selected,
    }


def get_first_event_win_rate(event: str, csv_path: str | Path | None = None) -> dict[str, Any]:
    """Return conditional win rate for the team recorded as achieving an event first."""
    return metrics.first_event_win_rate(_rows(csv_path), event)


def get_first_event_occurrence_rate(event: str, csv_path: str | Path | None = None) -> dict[str, Any]:
    """Return event occurrence rate, with the valid-match denominator included."""
    return metrics.first_event_occurrence_rate(_rows(csv_path), event)


def get_average_game_duration(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Return mean match duration in seconds."""
    return metrics.average_game_duration(_rows(csv_path))


def get_spell_stats(
    spell: str | int | None = None,
    sort_by: str = "picked_teams",
    limit: int | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return summoner-spell pick and win statistics, optionally filtered/sorted.

    ``pick_rate`` is per **team appearance** (two per match): a spell is fielded by
    a team, not by a match. When no spell is named, results are ranked and capped at
    ``DEFAULT_RANKING_LIMIT`` unless an explicit ``limit`` is given.
    """
    _validate_limit(limit)
    if sort_by not in {"picked_teams", "pick_rate", "win_rate_when_picked"}:
        raise ValueError("sort_by must be 'picked_teams', 'pick_rate' or 'win_rate_when_picked'")
    effective_limit = limit if limit is not None else DEFAULT_RANKING_LIMIT

    all_stats = metrics.spell_stats(_rows(csv_path))
    requested_id = _resolve_spell_id(spell) if spell is not None else None

    if spell is not None and requested_id is None:
        return {
            "metric": "spell_stats",
            "spell_query": str(spell),
            "spell_id_filter": None,
            "mapping_source": "Riot Data Dragon (zh_CN) summoner spell snapshot",
            "total_spells": len(all_stats),
            "results": [],
            "note": "未识别的召唤师技能；请传入数字 ID 或精确中文名（如 闪现）",
        }

    if requested_id is not None:
        selected = [all_stats[requested_id]] if requested_id in all_stats else []
    else:
        selected = list(all_stats.values())
        metric_name = sort_by if sort_by != "picked_teams" else "picked_teams"
        selected.sort(
            key=lambda item: (
                -(item[metric_name]["value"] if isinstance(item[metric_name], dict) else item[metric_name]) or 0,
                int(item["spell_id"]),
            )
        )
        selected = selected[:effective_limit]

    _add_spell_labels(selected)
    return {
        "metric": "spell_stats",
        "sort_by": sort_by,
        "limit": effective_limit,
        "spell_query": None if spell is None else str(spell),
        "spell_id_filter": requested_id,
        "mapping_source": "Riot Data Dragon (zh_CN) summoner spell snapshot",
        "total_spells": len(all_stats),
        "results": selected,
    }


def get_spell_win_split(spell: str | int, csv_path: str | Path | None = None) -> dict[str, Any]:
    """Compare team win rate with versus without one summoner spell.

    Both sides cover the same matches, so the difference is meaningful as a
    description of this dataset. It is **not** a causal effect: which champions and
    roles take a spell also drives the outcome.
    """
    requested_id = _resolve_spell_id(spell)
    if requested_id is None:
        return {
            "metric": "spell_win_split",
            "spell_query": str(spell),
            "spell_id_filter": None,
            "note": "未识别的召唤师技能；请传入数字 ID 或精确中文名（如 闪现）",
        }

    result = metrics.spell_win_split(_rows(csv_path), requested_id)
    by_id, _ = _spell_maps()
    result["spell_name"] = by_id.get(requested_id)
    result["mapping_source"] = "Riot Data Dragon (zh_CN) summoner spell snapshot"
    result["caveat"] = "观测数据，非受控实验；胜率差异同时受英雄与位置选择影响，不可解读为因果。"
    return result
