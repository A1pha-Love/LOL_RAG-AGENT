"""Run every offline check for this project from one command.

    python scripts/run_checks.py            # all offline checks
    python scripts/run_checks.py --fast     # the seconds-long subset (pre-commit)
    python scripts/run_checks.py --list     # show what would run

Nothing here calls a model API, so it is safe to run on every commit and needs
no quota. The API-dependent checks (Agent regression, rerank measurement) are
deliberately excluded: they cost money and are not deterministic. Run those with
scripts/evaluate_rag.py and scripts/measure_rerank.py.

Each step prints one line; a failing step prints the captured output so the
failure is diagnosable without re-running with extra flags.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    argv: list[str]
    fast: bool = True
    needs_index: bool = False
    cwd: Path | None = None

    @property
    def display(self) -> str:
        return " ".join(self.argv)


def _python_files(*directories: str) -> list[str]:
    files: list[str] = []
    for directory in directories:
        root = PROJECT_ROOT / directory
        if root.is_dir():
            files.extend(str(path) for path in sorted(root.rglob("*.py")))
    return files


def _node_npm() -> str | None:
    """Path to npm.cmd, or None when the frontend toolchain is unavailable."""
    import shutil

    for name in ("npm.cmd", "npm"):
        found = shutil.which(name)
        if found:
            return found
    return None


def build_steps() -> list[Step]:
    tests = PROJECT_ROOT / "tests"
    steps = [
        Step(
            key="compile",
            title="语法编译（src / scripts / tests）",
            argv=[PYTHON, "-m", "compileall", "-q", "-x", r"__pycache__", *(
                str(PROJECT_ROOT / name) for name in ("src", "scripts", "tests")
            )],
            fast=True,
        ),
        Step(
            key="tests",
            title="单元测试（pytest）",
            argv=[PYTHON, "-m", "pytest", str(tests), "-q", "-p", "no:cacheprovider"],
            fast=True,
        ),
        Step(
            key="kbase",
            title="知识库结构与分块校验",
            argv=[PYTHON, str(PROJECT_ROOT / "scripts" / "verify_kbase.py")],
            fast=False,
        ),
        Step(
            key="recall",
            title="检索来源召回断言（离线）",
            argv=[PYTHON, str(PROJECT_ROOT / "scripts" / "verify_retrieval_offline.py")],
            fast=False,
            needs_index=True,
        ),
        Step(
            key="sse",
            title="SSE 分帧契约（需服务器）",
            argv=[PYTHON, str(PROJECT_ROOT / "scripts" / "verify_sse_contract.py")],
            fast=False,
        ),
    ]

    # The frontend tests are offline and fast, but need Node and an installed
    # node_modules, so this step is skipped rather than failed when absent —
    # otherwise the check would demand a toolchain the backend work does not need.
    npm = _node_npm()
    frontend = PROJECT_ROOT / "frontend"
    if npm and (frontend / "node_modules").is_dir():
        steps.append(
            Step(
                key="frontend",
                title="前端单测（vitest）",
                argv=[npm, "--prefix", str(frontend), "test", "--silent"],
                fast=True,
                cwd=frontend,
            )
        )
    return steps


def _server_reachable(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Whether the local API is listening, checked without spawning anything."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


def run(step: Step, *, index_present: bool) -> tuple[bool, str, float]:
    if step.needs_index and not index_present:
        return True, "skipped (index not built)", 0.0
    if step.key == "sse" and not _server_reachable():
        return True, "skipped 服务器未运行（127.0.0.1:8000）", 0.0

    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    # Keep bytecode out of the source tree: __pycache__ is not writable in the
    # confined sandbox, and stale .pyc files silently mask edits.
    environment["PYTHONPYCACHEPREFIX"] = str(Path(environment.get("TEMP", ".")) / "lolpyc")

    started = time.perf_counter()
    completed = subprocess.run(
        step.argv,
        cwd=step.cwd or PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    output = (completed.stdout or "") + (completed.stderr or "")

    # A step that cannot run must report SKIP, never PASS: passing without running
    # would fake coverage. Exit code 77 is the conventional "skipped" signal.
    if completed.returncode == 77:
        return True, f"skipped {output.strip()}", elapsed
    return completed.returncode == 0, output, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="run only the quick steps (pre-commit)")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    parser.add_argument("--verbose", action="store_true", help="always print each step's output")
    args = parser.parse_args()

    steps = [step for step in build_steps() if not args.fast or step.fast]
    if args.list:
        for step in steps:
            print(f"[{'fast' if step.fast else 'slow'}] {step.title}\n        {step.display}")
        return 0

    index_present = (PROJECT_ROOT / "reports" / "knowledge.db").is_file()
    print("=" * 88)
    print(f"离线检查（{len(steps)} 步）{'— 快速模式' if args.fast else ''}")
    print("=" * 88)

    failures: list[tuple[Step, str]] = []
    for step in steps:
        ok, output, elapsed = run(step, index_present=index_present)
        marker = "PASS" if ok else "FAIL"
        note = ""
        if output.strip().startswith("skipped"):
            marker, note = "SKIP", f"  {output.strip().removeprefix('skipped ')}"
        print(f"  [{marker}] {step.title:34} {elapsed:5.1f}s{note}", flush=True)
        if args.verbose and output.strip():
            for line in output.rstrip().splitlines()[-25:]:
                print(f"          {line}")
        if not ok:
            failures.append((step, output))

    print("=" * 88)
    if failures:
        for step, output in failures:
            print(f"--- {step.title} ({step.display}) ---")
            tail = output.rstrip().splitlines()
            for line in tail[-40:]:
                print(f"  {line}")
            print()
        print(f"RESULT: FAIL — {len(failures)}/{len(steps)} 步失败")
        return 1

    print(f"RESULT: PASS — {len(steps)}/{len(steps)} 步通过")
    if not index_present:
        print("提示：reports/knowledge.db 不存在，检索召回断言已跳过；先运行 scripts/index_knowledge.py。")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
