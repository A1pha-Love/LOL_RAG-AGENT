"""Fetch dimension-expansion data for the LoL ranked dataset.

Two independent sources, both verified reachable:
  1. Kaggle `datasnaek/league-of-legends` companion JSONs (CC0: Public Domain).
     The local games.csv is byte-identical to that dataset's copy, so these
     files share the dataset's own champion/summoner-spell ID vocabulary.
  2. Riot Data Dragon 7.x static snapshots (official, zh_CN), one directory
     per patch covered by the dataset's 2017-06-08 .. 2017-09-06 UTC range.

Every download is recorded in a manifest with URL, size and sha256 so a
re-run is verifiable and idempotent. Raw payloads are never edited in place;
derived mapping files are written separately by `build_dimension_mappings`.

Usage:
    python scripts/fetch_dimension_data.py                # dataset jsons + 7.17.1
    python scripts/fetch_dimension_data.py --versions all # 7.11.1 .. 7.18.1
    python scripts/fetch_dimension_data.py --force        # re-download everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "ShuJuYuan"
DDRAGON_ROOT = RAW_ROOT / "ddragon"
MANIFEST_PATH = DDRAGON_ROOT / "_manifest.json"

USER_AGENT = "lol-ranked-data-agent/0.1 (local research; contact: project owner)"

# Patch -> date mapping is inferred from the dataset's creationTime range
# (2017-06-08 .. 2017-09-06 UTC). These are snapshots, never a per-match claim.
DATASET_PATCHES = ("7.11.1", "7.12.1", "7.13.1", "7.14.1", "7.15.1", "7.16.1", "7.17.1", "7.18.1")
PRIMARY_PATCH = "7.17.1"
DDRAGON_LOCALES = ("zh_CN",)

# Per-patch payloads. `champion.json` (list) + `championFull.json` (skills)
# + `item.json` + `summoner.json`.
DDRAGON_FILES = ("champion.json", "championFull.json", "item.json", "summoner.json")

KAGGLE_DATASET = "datasnaek/league-of-legends"
KAGGLE_FILES = ("champion_info.json", "champion_info_2.json", "summoner_spell_info.json")
KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{dataset}/{filename}"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download(url: str, *, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.is_file():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"generated_by": "scripts/fetch_dimension_data.py", "entries": {}}


def _save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["entry_count"] = len(manifest["entries"])
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fetch_to_file(
    *,
    key: str,
    url: str,
    target: Path,
    manifest: dict[str, Any],
    force: bool,
    validate: Any = None,
) -> tuple[str, str]:
    """Download `url` into `target`, verifying cache, hash and optional schema.

    Returns (status, detail) where status is one of cached | downloaded | failed.
    """
    entry = manifest["entries"].get(key)

    if target.is_file() and entry and not force:
        existing = target.read_bytes()
        if _sha256(existing) == entry.get("sha256"):
            return "cached", f"{target.stat().st_size:,} bytes (hash matches manifest)"

    try:
        payload = _download(url)
    except urllib.error.HTTPError as exc:
        return "failed", f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return "failed", f"{type(exc).__name__}: {exc}"

    if validate is not None:
        try:
            detail = validate(payload)
        except Exception as exc:  # noqa: BLE001
            return "failed", f"validation failed: {type(exc).__name__}: {exc}"
    else:
        detail = ""

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    manifest["entries"][key] = {
        "url": url,
        "path": target.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": detail,
    }
    return "downloaded", f"{len(payload):,} bytes {('| ' + detail) if detail else ''}"


def _valid_json(payload: bytes) -> str:
    parsed = json.loads(payload)
    return f"json keys={list(parsed)[:6]}" if isinstance(parsed, dict) else f"json list len={len(parsed)}"


def _make_ddragon_validator(version: str, expected_version_field: bool):
    def validate(payload: bytes) -> str:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict) or "data" not in parsed:
            raise ValueError("missing top-level 'data'")
        declared = parsed.get("version")
        if expected_version_field and declared != version:
            raise ValueError(f"declared version {declared!r} != requested {version!r}")
        return f"data entries={len(parsed['data'])}, declared version={declared}"

    return validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--versions",
        default=PRIMARY_PATCH,
        help=f"comma-separated Data Dragon patches, or 'all' for {','.join(DATASET_PATCHES)} (default: {PRIMARY_PATCH})",
    )
    parser.add_argument("--force", action="store_true", help="re-download even when the cached hash matches")
    parser.add_argument("--skip-kaggle", action="store_true", help="only fetch Data Dragon payloads")
    args = parser.parse_args()

    if args.versions.strip().lower() == "all":
        versions = list(DATASET_PATCHES)
    else:
        versions = [item.strip() for item in args.versions.split(",") if item.strip()]
    if not versions:
        parser.error("no versions requested")

    manifest = _load_manifest()
    manifest["dataset"] = {
        "kaggle": KAGGLE_DATASET,
        "license": "CC0: Public Domain",
        "local_csv": "ShuJuYuan/games.csv",
    }
    manifest["ddragon_versions_requested"] = versions

    counts = {"cached": 0, "downloaded": 0, "failed": 0}
    failures: list[str] = []

    def report(status: str, label: str, where: str, detail: str) -> None:
        counts[status] += 1
        marker = {"cached": "CACHED", "downloaded": "GET   ", "failed": "FAIL  "}[status]
        print(f"  {marker} {label:52} {detail}")
        if status == "failed":
            failures.append(f"{label} <- {where} ({detail})")

    if not args.skip_kaggle:
        print("=" * 96)
        print(f"Kaggle companion files — {KAGGLE_DATASET} (CC0)")
        print("=" * 96)
        for filename in KAGGLE_FILES:
            url = KAGGLE_DOWNLOAD_URL.format(dataset=KAGGLE_DATASET, filename=filename)
            target = RAW_ROOT / filename
            status, detail = _fetch_to_file(
                key=f"kaggle/{filename}",
                url=url,
                target=target,
                manifest=manifest,
                force=args.force,
                validate=_valid_json,
            )
            report(status, f"kaggle/{filename}", url, detail)
        print()

    print("=" * 96)
    print("Riot Data Dragon (official static data, zh_CN)")
    print("=" * 96)
    for version in versions:
        print(f"[{version}]")
        for locale in DDRAGON_LOCALES:
            for filename in DDRAGON_FILES:
                url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/{locale}/{filename}"
                target = DDRAGON_ROOT / version / locale / filename
                status, detail = _fetch_to_file(
                    key=f"ddragon/{version}/{locale}/{filename}",
                    url=url,
                    target=target,
                    manifest=manifest,
                    force=args.force,
                    validate=_make_ddragon_validator(version, filename != "championFull.json"),
                )
                report(status, f"ddragon/{version}/{locale}/{filename}", url, detail)
        print()

    _save_manifest(manifest)

    print("=" * 96)
    print(f"SUMMARY cached={counts['cached']} downloaded={counts['downloaded']} failed={counts['failed']}")
    print(f"MANIFEST {MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    if failures:
        print("FAILURES:")
        for item in failures:
            print("  -", item)
    return 1 if failures else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
