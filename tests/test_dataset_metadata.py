"""Guard the completed metadata verification from silently regressing.

`kbase/project_data_audit.md` used to carry a "待核实项（尚未完成）" section saying the
dataset's own metadata attachments had never been used to independently verify the
field definitions. That verification has now been done, and the risk is that a later
edit reintroduces the "outstanding" wording — which readers (and the Agent, since this
is indexed) would then repeat as fact.

These tests assert the recorded outcome rather than the prose, so the claim stays
checkable: the audit says the check was done, and the script that does it still
passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DOC = PROJECT_ROOT / "kbase" / "project_data_audit.md"
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_dataset_metadata.py"
DATA_DIR = PROJECT_ROOT / "ShuJuYuan"

ATTACHMENTS = (
    DATA_DIR / "champion_info.json",
    DATA_DIR / "champion_info_2.json",
    DATA_DIR / "summoner_spell_info.json",
)

needs_attachments = pytest.mark.skipif(
    not all(path.is_file() for path in ATTACHMENTS),
    reason="dataset metadata attachments are absent",
)


def test_audit_no_longer_claims_the_verification_is_outstanding() -> None:
    text = AUDIT_DOC.read_text(encoding="utf-8")
    assert "待核实项（尚未完成）" not in text, (
        "the audit still advertises an unfinished verification section; the "
        "independent metadata check has been completed"
    )
    assert "元数据附件的独立验证（已完成）" in text, "the audit must record the outcome"


def test_audit_keeps_the_two_version_qualifications() -> None:
    """The check surfaced two statements that needed narrowing; keep them narrowed.

    Both were genuine imprecisions found by the verification, so a revert would be a
    factual regression rather than a style change.
    """
    audit = AUDIT_DOC.read_text(encoding="utf-8")
    assert "champion_info_2.json" in audit, "the -1 sentinel's actual source must be named"

    ban_doc = (PROJECT_ROOT / "kbase" / "project_metrics_ban_event.md").read_text(encoding="utf-8")
    assert "champion_info_2.json" in ban_doc, (
        "the ban document must attribute the -1 placeholder to the 7.18.1 attachment, "
        "not to champion_info.json which does not contain it"
    )

    mapping_doc = (PROJECT_ROOT / "kbase" / "project_champion_name_mapping.md").read_text(
        encoding="utf-8"
    )
    assert "champion_info_2.json" in mapping_doc, (
        "the mapping document must attribute the sentinel to the correct attachment"
    )


@needs_attachments
def test_the_verification_script_still_passes() -> None:
    """Run the real check, so the recorded conclusion stays reproducible."""
    completed = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, (
        "scripts/verify_dataset_metadata.py failed:\n"
        + (completed.stdout or "")[-1500:]
        + (completed.stderr or "")[-800:]
    )
    assert "RESULT: PASS" in completed.stdout
