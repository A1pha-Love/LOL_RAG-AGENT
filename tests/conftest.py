"""Shared pytest configuration.

Two environment quirks are handled here so individual tests stay ordinary.

1. **Temp directory.** pytest's default base temp directory is not writable
   under the confined file sandbox, which turns every tmp_path-based test into
   an error at fixture setup. ``basetemp`` is therefore redirected into the
   repository's gitignored ``.cache/``.
2. **Bytecode cache.** ``__pycache__`` inside the source tree is not writable
   either, and a stale ``.pyc`` silently masks edits. Python is pointed at a
   cache prefix outside the repository via ``PYTHONPYCACHEPREFIX``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"

# Make `import lol_agent...` work without an editable install.
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config: pytest.Config) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(CACHE_ROOT / "pytest-tmp")
    # A left-over directory from an interrupted run would otherwise make pytest
    # refuse to start ("basetemp exists and is not empty").
    basetemp = Path(config.option.basetemp)
    if basetemp.exists():
        shutil.rmtree(basetemp, ignore_errors=True)

    pycache = CACHE_ROOT / "pycache"
    pycache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache))
