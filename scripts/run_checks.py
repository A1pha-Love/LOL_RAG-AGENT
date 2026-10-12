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
import json
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
        Step(
            key="concurrent",
            title="并发轮次隔离（需服务器）",
            argv=[PYTHON, str(PROJECT_ROOT / "scripts" / "check_concurrent_turns.py")],
            fast=False,
        ),
        Step(
            key="metadata",
            title="数据集元数据交叉验证",
            argv=[PYTHON, str(PROJECT_ROOT / "scripts" / "verify_dataset_metadata.py")],
            fast=False,
        ),
        Step(
            key="items",
            title="装备目录漂移核验（离线）",
            argv=[PYTHON, str(PROJECT_ROOT / "scripts" / "verify_item_drift.py")],
            fast=False,
        ),
        Step(
            key="endpoints",
            title="接口冒烟（需服务器，不调模型）",
            argv=[PYTHON, str(PROJECT_ROOT / "scripts" / "verify_api_endpoints.py")],
            fast=False,
        ),
        Step(
            key="agent_eval",
            title="Agent 行为阈值门禁（有缓存则不调模型）",
            argv=[
                PYTHON,
                str(PROJECT_ROOT / "scripts" / "eval_agent.py"),
                # Known gaps are recorded in evals/agent_baseline.json; a case that
                # used to pass and now fails is still reported as a regression.
                "--allow-known-failures",
            ],
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


# Steps that need a running server. Skipping them here (rather than letting them
# start and fail) keeps the check usable without a server, and the skip is reported
# as SKIP — never as PASS, which would fake coverage.
SERVER_STEPS = {"sse", "concurrent", "endpoints"}

# Steps that call the model API, keyed by which quota they need. Embeddings and chat
# are billed independently, so a step is skipped only when *its own* dependency is
# unavailable — otherwise a chat outage would silently drop the embedding-only checks
# and lose coverage that could still have run.
API_STEPS = {
    "kbase": "embedding",
    "recall": "embedding",
    "sse": "chat",
    "concurrent": "chat",
}


def _agent_eval_needs_model() -> bool:
    """Whether the Agent evaluation would have to call the model.

    With a populated cache it re-scores recorded answers, so the check still runs when
    the API is unavailable — it just cannot detect changes that only a fresh answer
    would reveal. An empty or missing cache means it must ask, so it becomes an API
    step and is skipped for quota reasons rather than failing.
    """
    cache = PROJECT_ROOT / ".cache" / "eval_agent_cache.json"
    try:
        return not cache.is_file() or not json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True


def _quota_reasons() -> dict[str, str]:
    """Probe each quota once: several steps share a dependency and the answer is equal."""
    import importlib.util

    scripts_dir = PROJECT_ROOT / "scripts"
    spec = importlib.util.spec_from_file_location("_checks_http", scripts_dir / "_http.py")
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reasons: dict[str, str] = {}
    for kind, probe in (
        ("embedding", module.embedding_quota_exhausted),
        ("chat", module.chat_quota_exhausted),
    ):
        reason = probe()
        if reason:
            reasons[kind] = reason
    return reasons


def run(
    step: Step,
    *,
    index_present: bool,
    quota_reasons: dict[str, str] | None = None,
) -> tuple[bool, str, float]:
    quota_reasons = quota_reasons or {}
    if step.needs_index and not index_present:
        return True, "skipped (index not built)", 0.0
    if step.key in SERVER_STEPS and not _server_reachable():
        return True, "skipped 服务器未运行（127.0.0.1:8000）", 0.0

    # The Agent evaluation only needs the model when it has nothing cached to replay.
    if step.key == "agent_eval":
        needed = "chat" if _agent_eval_needs_model() else None
    else:
        needed = API_STEPS.get(step.key)
    if needed and quota_reasons.get(needed):
        return True, f"skipped {quota_reasons[needed]}", 0.0

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
    # Probed once per dependency: several steps share one, and the answer is the same.
    quota_reasons = _quota_reasons()
    print("=" * 88)
    print(f"离线检查（{len(steps)} 步）{'— 快速模式' if args.fast else ''}")
    for kind, reason in quota_reasons.items():
        print(f"注意：{reason} — 依赖 {kind} 的步骤将标记为 SKIP")
    print("=" * 88)

    failures: list[tuple[Step, str]] = []
    for step in steps:
        ok, output, elapsed = run(step, index_present=index_present, quota_reasons=quota_reasons)
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
