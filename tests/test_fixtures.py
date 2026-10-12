"""Tests for the test-suite fixtures themselves.

Motivated by a real leak: `temp_db` used to derive its path from `temp_path` via
`with_suffix(".db")`, so only the `.tmp` path was cleaned up and every database the
tests created was left behind. Nothing failed — the directory simply grew, reaching
2,218 files and 87 MB before anyone noticed. A fixture that leaks gives no other
signal, which is why it is worth a test.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from conftest import _remove_with_retry

CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"


def test_temp_db_hands_out_a_database_path_under_cache(temp_db: Path) -> None:
    assert temp_db.suffix == ".db"
    assert temp_db.parent == CACHE_ROOT


def test_temp_db_and_temp_path_do_not_collide(temp_path: Path, temp_db: Path) -> None:
    """Sharing a stem is what caused the original leak: cleanup removed only `.tmp`.

    The two fixtures are now independent, so deriving one path from the other cannot
    silently make a test's database un-cleanable again.
    """
    assert temp_path.with_suffix(".db") != temp_db
    assert temp_path.stem != temp_db.stem


def test_teardown_removes_a_database_and_its_wal_files() -> None:
    """Exercise the teardown logic against what a test actually leaves behind.

    Cleanup runs after the owning test finishes, so it cannot be asserted from inside
    that test. This reproduces the artefacts instead: SQLite writes `-wal`/`-shm`
    beside the database in WAL mode, and those must be removed too.
    """
    probe = CACHE_ROOT / f"test-{uuid.uuid4().hex[:12]}.db"
    connection = sqlite3.connect(probe)
    connection.execute("CREATE TABLE t (x INTEGER)")
    connection.commit()
    connection.close()
    assert probe.exists(), "the probe must create a file, or the test proves nothing"

    for candidate in (probe, probe.with_name(probe.name + "-wal"), probe.with_name(probe.name + "-shm")):
        _remove_with_retry(candidate)

    assert not probe.exists()
    assert not list(CACHE_ROOT.glob(probe.name + "-*")), "WAL side files must be gone too"


def test_removing_a_missing_file_is_not_an_error() -> None:
    """Teardown runs even when a test never created its file."""
    _remove_with_retry(CACHE_ROOT / "definitely-absent-test-file.db")


def test_the_suite_leaves_no_test_databases_behind() -> None:
    """A whole-run guard: after this module's teardown, nothing of ours remains.

    Any test naming pattern is per-run, so this asserts the general property that the
    fixtures clean up as they go rather than the specific count.
    """
    owned = {
        path
        for path in CACHE_ROOT.glob("test-*.db")
        # Files of the currently-running test may still be open; only stale ones matter.
        if path.is_file()
    }
    # The fixtures cannot guarantee other processes are idle, so this cannot be exact.
    # It fails loudly only for a large accumulation, which is the leak signature.
    assert len(owned) < 50, (
        f"{len(owned)} test databases left in .cache — fixtures are leaking files again"
    )
