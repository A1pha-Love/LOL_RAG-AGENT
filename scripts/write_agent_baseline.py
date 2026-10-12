"""Rewrite evals/agent_baseline.json from the current cached evaluation.

The baseline must reflect measured behaviour, not aspiration. Two cases proved to be
**non-deterministic** across repeated runs (see `flaky` below): they pass most of the
time and fail occasionally, so recording either outcome alone would be misleading —
"pass" would flag normal variation as a regression, "fail" would hide a real one.

They are therefore recorded as `flaky` and the gate reports them without treating them
as regressions. `scripts/eval_agent.py --repeat N` is the way to re-measure.

Run after refreshing the cache:  python scripts/write_agent_baseline.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CACHE_PATH = PROJECT_ROOT / ".cache" / "eval_agent_cache.json"
BASELINE_PATH = PROJECT_ROOT / "evals" / "agent_baseline.json"
CASES_PATH = PROJECT_ROOT / "evals" / "rag_cases.jsonl"

# Cases whose outcome genuinely varies between runs. Kept explicit so the reason is on
# the record rather than inferred.
#
# Currently empty: with the chat model pinned to temperature 0, the two cases that used
# to vary (prompt_injection_guardrail, spell_pick_rate_denominator) no longer do — a
# live refresh produced 14/15 with no case flipping. Leaving them listed as flaky would
# excuse a future real regression against them, so the list must stay accurate.
FLAKY: dict[str, str] = {}


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--clear-flaky", action="append", default=[], help="drop a case from the flaky list"
    )
    args = parser.parse_args()

    gate = _load("_eval_agent_for_baseline", "scripts/eval_agent.py")
    helpers = gate._load_scoring_helpers()

    if not CACHE_PATH.is_file():
        print(f"no cache at {CACHE_PATH}; run scripts/eval_agent.py --refresh first")
        return 1
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    flaky = {key: value for key, value in FLAKY.items() if key not in set(args.clear_flaky)}
    results: dict[str, str] = {}
    missing: list[str] = []

    for case in cases:
        payload = cache.get(gate.question_fingerprint(case))
        if payload is None:
            missing.append(case["id"])
            continue
        outcome = gate.score_case(case, payload, helpers)
        if case["id"] in flaky:
            results[case["id"]] = "flaky"
        else:
            results[case["id"]] = "pass" if outcome.passed else "fail"

    if missing:
        print(f"cache is missing {len(missing)} case(s): {', '.join(missing)}")
        print("run scripts/eval_agent.py --refresh to record them before writing a baseline")
        return 1

    passed = [k for k, v in results.items() if v == "pass"]
    failed = [k for k, v in results.items() if v == "fail"]
    flaky_cases = [k for k, v in results.items() if v == "flaky"]

    BASELINE_PATH.write_text(
        json.dumps(
            {
                "note": (
                    "Recorded from the evaluation cache by scripts/write_agent_baseline.py. "
                    "'pass' that later fails is a regression; 'flaky' is a case whose tool "
                    "selection varies between runs and is reported without failing the gate."
                ),
                # Which model these results describe. The cache key includes the model,
                # so the answers behind this baseline belong to one specific model;
                # without recording it, a baseline recorded under a different model would
                # look like it still applies after a switch.
                "chat_model": os.getenv("QWEN_CHAT_MODEL", "").strip(),
                "temperature": os.getenv("QWEN_TEMPERATURE", "").strip() or "provider-default",
                "gates": list(gate.GATES),
                "flaky_reasons": flaky,
                "cases": dict(sorted(results.items())),
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"  {len(results)} cases written to {BASELINE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  chat_model: {os.getenv('QWEN_CHAT_MODEL', '').strip() or '(unset)'}"
          f"  temperature: {os.getenv('QWEN_TEMPERATURE', '').strip() or 'provider-default'}")
    print(f"  pass  {len(passed)}: {sorted(passed)}")
    print(f"  fail  {len(failed)}: {sorted(failed) or '(none)'}")
    print(f"  flaky {len(flaky_cases)}: {sorted(flaky_cases)}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
