"""Guard the item-drift document: its figures must match the Data Dragon snapshots.

The drift write-up states counts, prices and stat changes for eight patches. Those
figures are easy to get subtly wrong by hand (the first draft under-counted stat
changes as 9 when the answer is 12, and mis-stated the catalogue's own scope as 228
when the catalogue contains 193). Prose numbers rot silently, so this checks the
document against the snapshots rather than trusting it.

The real check lives in `scripts/verify_item_drift.py` (also run by run_checks.py);
these tests make the claim part of the suite and fail loudly if either the document
or the snapshots drift apart.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DDRAGON = PROJECT_ROOT / "ShuJuYuan" / "ddragon"
DRIFT_DOC = PROJECT_ROOT / "kbase" / "conflict_item_catalog_drift.md"
CATALOGUE_DOC = PROJECT_ROOT / "kbase" / "domain_item_catalog_7.17.1.md"
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_item_drift.py"
VERSIONS = ["7.11.1", "7.12.1", "7.13.1", "7.14.1", "7.15.1", "7.16.1", "7.17.1", "7.18.1"]

needs_snapshots = pytest.mark.skipif(
    not all((DDRAGON / v / "zh_CN" / "item.json").is_file() for v in VERSIONS),
    reason="Data Dragon snapshots are absent",
)


def _load(version: str) -> dict[str, dict]:
    payload = json.loads((DDRAGON / version / "zh_CN" / "item.json").read_text(encoding="utf-8"))
    return {
        item_id: entry
        for item_id, entry in (payload.get("data") or {}).items()
        if (entry.get("maps") or {}).get("11")
    }


def test_document_leads_with_the_per_match_limitation() -> None:
    """The most important sentence: this data cannot answer a per-match question.

    The CSV has no item column and no per-match patch column, so a reader could
    otherwise wrongly conclude that a stated price applies to a given match.
    """
    text = DRIFT_DOC.read_text(encoding="utf-8")
    head = text.split("## 观测一")[0]
    assert "装备数据不在对局 CSV 内" in head
    assert "没有逐场 `patch` 字段" in head or "没有逐场 patch 字段" in head
    assert "不能回答" in head


@needs_snapshots
def test_documented_patch_sizes_match_the_snapshots() -> None:
    text = DRIFT_DOC.read_text(encoding="utf-8")
    for version in VERSIONS:
        items = _load(version)
        match = re.search(
            rf"\|\s*{re.escape(version)}\s*\|\s*\*{{0,2}}(\d+)\*{{0,2}}\s*\|", text
        )
        assert match, f"{version} missing from the drift table"
        assert int(match.group(1)) == len(items), version


@needs_snapshots
def test_catalogue_scope_is_stated_correctly() -> None:
    """The catalogue filters on total > 0 as well as the map, so it holds 193 items.

    Getting this wrong is easy: 228 is the map-only count. The document must name
    both figures so a reader can tell which one a given statement uses.
    """
    catalogue = CATALOGUE_DOC.read_text(encoding="utf-8")
    entries = len(re.findall(r"^### ", catalogue, re.M))
    priced = sum(
        1 for entry in _load("7.17.1").values() if (entry.get("gold") or {}).get("total", 0) > 0
    )
    assert entries == priced == 193, f"catalogue has {entries} entries, priced count is {priced}"

    text = DRIFT_DOC.read_text(encoding="utf-8")
    assert "193" in text, "the drift document must state the catalogue's actual scope"
    assert "228" in text, "the drift document must state the wider scope it uses itself"


@needs_snapshots
def test_purchasable_is_not_treated_as_a_price_floor() -> None:
    """`gold.purchasable` and `gold.total > 0` disagree on 17 entries.

    The document calls this out, because using `purchasable` as "has a price" silently
    drops 17 zero-cost entries (trinkets, turret items) while keeping 3 that cannot be
    bought despite having a price.
    """
    items = _load("7.17.1")
    zero_price = {i for i, e in items.items() if (e.get("gold") or {}).get("total", 0) <= 0}
    not_purchasable = {i for i, e in items.items() if not (e.get("gold") or {}).get("purchasable")}
    assert len(zero_price & not_purchasable) == 18
    assert len(zero_price - not_purchasable) == 17

    text = DRIFT_DOC.read_text(encoding="utf-8")
    assert "只有 18 条重叠" in text
    assert "purchasable" in text


@needs_snapshots
def test_verification_script_passes() -> None:
    """Re-run the real check, so the recorded figures stay reproducible."""
    completed = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, (
        "scripts/verify_item_drift.py failed:\n"
        + (completed.stdout or "")[-1800:]
        + (completed.stderr or "")[-800:]
    )
    assert "RESULT: PASS" in completed.stdout


def test_drift_document_declares_official_provenance() -> None:
    text = DRIFT_DOC.read_text(encoding="utf-8")
    assert "category: conflict" in text
    assert "source_url:" in text and "item.json" in text
    assert "license:" in text
