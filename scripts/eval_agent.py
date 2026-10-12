"""Gate Agent behaviour: run evals/rag_cases.jsonl with pass/fail thresholds.

This is the CI counterpart to `scripts/evaluate_rag.py`. That script prints a report
for a human to read and only exits non-zero for tool/retrieval/numeric/forbidden
problems; answer phrasing is left to manual review, and nothing records what "good"
looked like last time. So the Agent-behaviour half of the evaluation was never part of
the automated checks — a regression in tool choice, retrieval targeting or answer
refusal would go unnoticed.

What this adds:

* **Thresholds**, not just a report: tool selection, retrieval targeting, numeric
  assertions and forbidden claims must all pass for every case.
* **Caching**, so the check can run in CI without spending model quota on every run.
  Responses are keyed by case id plus a hash of the question, so editing a case
  invalidates its own cached response and nothing else.
* **A baseline**, so a case that used to pass and now fails is reported as a
  regression rather than merely as a failure.

Gating philosophy: `answer_must_include` relies on literal/alias substring matching,
which is a heuristic over phrasing. Gating on it is therefore opt-in (``--strict``);
by default inclusion is measured and reported but does not fail the run. The four
semantic gates are exact and always enforced.

    python scripts/eval_agent.py                      # cached, gated
    python scripts/eval_agent.py --refresh            # re-run and update the cache
    python scripts/eval_agent.py --case-id X          # one case
    python scripts/eval_agent.py --strict             # also gate answer phrasing
    python scripts/eval_agent.py --write-baseline     # record current state
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Imported for its side effect: it loads .env, so QWEN_CHAT_MODEL is set before any
# fingerprint is computed. Relying on a transitive import instead would make the cache
# key depend on import order, which is exactly the kind of thing that fails silently.
from lol_agent import config as _config  # noqa: E402,F401

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import _eval_retrieval as _retrieval_helpers  # noqa: E402

CASES_PATH = PROJECT_ROOT / "evals" / "rag_cases.jsonl"
CACHE_PATH = PROJECT_ROOT / ".cache" / "eval_agent_cache.json"
BASELINE_PATH = PROJECT_ROOT / "evals" / "agent_baseline.json"

# Semantic gates: exact checks that must hold for every case.
GATES = ("tools", "no_forbidden_tools", "retrieval", "numeric", "no_forbidden_claims")


def _load_scoring_helpers():
    """Reuse evaluate_rag.py's pure helpers instead of copying its scoring rules.

    Two copies of the same heuristics would drift apart, and a disagreement between
    the report and the gate would be worse than either alone.
    """
    spec = importlib.util.spec_from_file_location(
        "_evaluate_rag", PROJECT_ROOT / "scripts" / "evaluate_rag.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/evaluate_rag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # safe: the script guards its own main()
    return module


@dataclass
class CaseOutcome:
    case_id: str
    gates: dict[str, bool] = field(default_factory=dict)
    inclusion_passed: int = 0
    inclusion_total: int = 0
    called_tools: list[str] = field(default_factory=list)
    missing_retrieval: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    missing_inclusions: list[str] = field(default_factory=list)
    answer_chars: int = 0
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and all(self.gates.get(name, False) for name in GATES)


@dataclass
class RunResult:
    outcome: CaseOutcome
    payload: dict[str, Any]  # everything needed to re-score without the model


def question_fingerprint(case: dict[str, Any]) -> str:
    """Key a cached response to the exact question *and the model that answered it*.

    The model is part of the key because answers are not comparable across models: a
    cache recorded with one model would otherwise be replayed as if it were the current
    model's behaviour, so switching models would silently keep reporting the old
    model's pass rate. Including the name means a switch invalidates every entry, which
    is the honest outcome — the next run records fresh answers.
    """
    payload = json.dumps(
        {
            "id": case["id"],
            "question": case["question"],
            "model": os.getenv("QWEN_CHAT_MODEL", "").strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_cache() -> dict[str, Any]:
    if not CACHE_PATH.is_file():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_case_live(case: dict[str, Any], helpers) -> dict[str, Any]:
    """Ask the Agent once through the production turn path and capture the scoring data.

    Uses `_eval_turn.invoke_case` rather than calling the graph directly: without a
    `TurnContext` the retrieval query falls back to the model's own phrasing, which is
    not what production does, so grading that path would record behaviour users never
    see. See `scripts/_eval_turn.py` for the details.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _eval_turn import (
        invoke_case_tolerant,
        retrieval_hits,
        retrieval_queries_used,
        tool_results,
    )

    messages, error = invoke_case_tolerant(case["question"])
    if error:
        # Surfaced as a case failure rather than an exception: whether the model passes
        # valid tool arguments is exactly the kind of thing this evaluation measures.
        raise RuntimeError(f"turn aborted: {error}")

    called: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            if call.get("name"):
                called.append(call["name"])

    # Consume the message objects here: only plain data may reach the cache.
    tool_payloads: dict[str, list[Any]] = {}
    for name in {m.name for m in messages if getattr(m, "type", None) == "tool" and m.name}:
        tool_payloads[name] = tool_results(messages, name)

    answer = messages[-1].content
    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "called_tools": sorted(set(called)),
        "tool_results": json.loads(json.dumps(tool_payloads, ensure_ascii=False, default=str)),
        "retrieved": [
            {"source": chunk.get("source"), "heading_path": chunk.get("heading_path")}
            for chunk in retrieval_hits(messages)
        ],
        # Kept so a miss can be diagnosed later: without the query, only the symptom
        # is recorded, and the cause has to be guessed.
        "retrieval_queries": retrieval_queries_used(messages),
        "answer": answer,
    }


def explain_retrieval(case: dict[str, Any], helpers) -> None:
    """Explain a retrieval miss: absent from the index, or present but ranked low.

    Without this the two are indistinguishable in the report, and they call for
    different responses — a missing section means the knowledge base lost content,
    whereas a low rank means the query or the ranking needs work. The probe searches
    with the threshold disabled and a wide pool, so it reports position rather than
    merely "not in the results".
    """
    from lol_agent.knowledge import search_knowledge

    print(f"      检索诊断（{case['id']}）")
    for expected in case.get("expected_retrieval", []):
        try:
            hits = search_knowledge(case["question"], top_k=40, include_below_threshold=True)
        except Exception as exc:  # noqa: BLE001
            print(f"        无法诊断: {type(exc).__name__}: {exc}")
            return
        rank = None
        score = None
        for index, hit in enumerate(hits, start=1):
            if hit.get("source") == expected["source"] and expected["heading"] in (
                hit.get("heading_path") or ""
            ):
                rank, score = index, hit.get("rerank_score")
                break
        label = f"{expected['source']} :: {expected['heading']}"
        if rank is None:
            print(f"        {label}")
            print("          → 在检索结果中未出现（内容缺失或严重失配）")
        else:
            marker = "（超出默认 top_k，属排名问题）" if rank > 8 else ""
            print(f"        {label}")
            print(f"          → 排名第 {rank}，rerank={score:.4f} {marker}")


def _retrieval_targets(item: Any) -> list[dict[str, str]]:
    """Kept as a thin alias: the rules live in scripts/_eval_retrieval.py so the offline
    verifier and this gate cannot disagree about what satisfies an assertion."""
    return _retrieval_helpers.retrieval_targets(item)


def _describe_expectation(item: Any) -> str:
    return _retrieval_helpers.describe_expectation(item)


def _retrieval_satisfied(item: Any, retrieved: list[dict[str, Any]]) -> bool:
    return _retrieval_helpers.is_satisfied(item, retrieved)


def score_case(case: dict[str, Any], payload: dict[str, Any], helpers) -> CaseOutcome:
    outcome = CaseOutcome(case_id=case["id"])
    called = set(payload["called_tools"])
    outcome.called_tools = sorted(called)

    expected = set(case.get("expected_tools", []))
    forbidden = set(case.get("forbidden_tools", []))
    outcome.gates["tools"] = expected.issubset(called)
    outcome.gates["no_forbidden_tools"] = not (forbidden & called)

    # Some cases name a tool that is genuinely helpful but not required, because the
    # remaining tools already answer the question. Demanding it records a failure for a
    # correct turn — observed when a case listed search_project_knowledge alongside two
    # statistics tools, and the statistics alone produced the requested figures.
    # `optional_tools` keeps the helpful tool off the critical path while still failing
    # the case if *none* of the listed tools was used, so it cannot pass by accident.
    optional = set(case.get("optional_tools", []))
    if optional:
        outcome.gates["tools"] = outcome.gates["tools"] and bool(called & (expected | optional))

    # Retrieval assertions match the heading PATH: the knowledge base re-chunks by
    # heading, so an expected heading is a substring of "文档 > 节 > 小节".
    #
    # An expectation may list several acceptable targets ("source" + "heading" pairs as
    # a list, or `alternatives`). That is deliberate: a question about a fact is often
    # answered equally well by more than one section, and requiring one specific
    # section grades the retriever on an arbitrary choice rather than on whether the
    # answer was available. Demanding a single section previously recorded failures for
    # cases whose content had in fact been retrieved.
    #
    # When the retrieval tool was not called and is optional, there is nothing to
    # assert: grading retrieval that did not happen would fail a turn for correctly
    # using the tools that did answer the question.
    retrieval_skipped = (
        "search_project_knowledge" not in called and "search_project_knowledge" in optional
    )
    if retrieval_skipped:
        outcome.gates["retrieval"] = True
    else:
        outcome.missing_retrieval = [
            _describe_expectation(item)
            for item in case.get("expected_retrieval", [])
            if not _retrieval_satisfied(item, payload["retrieved"])
        ]
        outcome.gates["retrieval"] = not outcome.missing_retrieval

    numeric = helpers._numeric_checks(case, payload["tool_results"])
    outcome.gates["numeric"] = all(numeric)  # vacuously true when there are none

    unnegated, _negated = helpers._forbidden_claim_checks(case, payload["answer"])
    outcome.forbidden_claims = unnegated
    outcome.gates["no_forbidden_claims"] = not unnegated

    inclusion = helpers._answer_inclusion_checks(case, payload["answer"])
    outcome.inclusion_total = len(inclusion)
    outcome.inclusion_passed = sum(int(matched) for _, matched in inclusion)
    outcome.missing_inclusions = [phrase for phrase, matched in inclusion if not matched]
    outcome.answer_chars = len(payload["answer"])
    return outcome


def load_cases(only: list[str] | None) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if only:
        wanted = set(only)
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            raise SystemExit("unknown case id(s): " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case["id"] in wanted]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-id", action="append", help="run only these cases (repeatable)")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-ask the model")
    parser.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    parser.add_argument("--strict", action="store_true", help="also gate answer_must_include phrasing")
    parser.add_argument("--write-baseline", action="store_true", help="record current results as the baseline")
    parser.add_argument("--ignore-baseline", action="store_true", help="skip the regression comparison")
    parser.add_argument(
        "--allow-known-failures",
        action="store_true",
        help="do not fail on cases the baseline already records as failing (regressions still fail)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print the answer preview for failures")
    parser.add_argument(
        "--explain-retrieval",
        action="store_true",
        help="for a retrieval miss, search without the threshold to report the rank",
    )
    args = parser.parse_args()

    helpers = _load_scoring_helpers()
    cases = load_cases(args.case_id)
    cache = {} if args.no_cache else load_cache()
    results: list[RunResult] = []
    cache_dirty = False

    print("=" * 88)
    print(f"Agent 行为评测 — {len(cases)} 个用例")
    print("=" * 88)

    for index, case in enumerate(cases, start=1):
        key = question_fingerprint(case)
        payload = None if args.refresh or args.no_cache else cache.get(key)
        source = "cache"
        if payload is None:
            try:
                payload = run_case_live(case, helpers)
            except Exception as exc:  # noqa: BLE001
                outcome = CaseOutcome(case_id=case["id"], error=f"{type(exc).__name__}: {exc}")
                results.append(RunResult(outcome=outcome, payload={}))
                print(f"[{index}/{len(cases)}] {case['id']:<34} ERROR  {outcome.error[:60]}")
                continue
            source = "live"
            if not args.no_cache:
                cache[key] = payload
                cache_dirty = True

        outcome = score_case(case, payload, helpers)
        if args.strict:
            outcome.gates["inclusion"] = not outcome.missing_inclusions
        results.append(RunResult(outcome=outcome, payload=payload))

        failed = [name for name in outcome.gates if not outcome.gates[name]]
        if args.strict and not outcome.gates.get("inclusion", True):
            failed.append("inclusion")
        status = "PASS" if not failed else "FAIL"
        print(
            f"[{index}/{len(cases)}] {case['id']:<34} {status}  "
            f"({source}, {outcome.answer_chars} 字"
            f"{', 包含 ' + str(outcome.inclusion_passed) + '/' + str(outcome.inclusion_total) if outcome.inclusion_total else ''})"
        )
        if failed:
            print(f"      未通过: {', '.join(failed)}")
        if outcome.missing_retrieval:
            print(f"      未命中检索: {'; '.join(outcome.missing_retrieval)}")
            if args.explain_retrieval:
                explain_retrieval(case, helpers)
        if outcome.forbidden_claims:
            print(f"      疑似禁止声称: {'; '.join(outcome.forbidden_claims)}")
        if args.verbose and failed:
            print(f"      回答预览: {payload['answer'][:400].replace(chr(10), ' ')}")

    if cache_dirty:
        save_cache(cache)

    # ------------------------------------------------------------------ summary
    total = len(results)
    gate_totals = {
        name: sum(int(result.outcome.gates.get(name, False)) for result in results) for name in GATES
    }
    inclusion_passed = sum(result.outcome.inclusion_passed for result in results)
    inclusion_total = sum(result.outcome.inclusion_total for result in results)
    failures = [result for result in results if not result.outcome.passed]

    print()
    print("=" * 88)
    print("阈值")
    for name in GATES:
        print(f"  {name:<22} {gate_totals[name]}/{total}")
    print(f"  {'answer_must_include':<22} {inclusion_passed}/{inclusion_total}（措辞启发式，仅在 --strict 时门禁）")

    # Cases matching none of their required content deserve to be called out, because
    # that is the signature of an answer which missed the point — observed when the
    # model claimed it "cannot report" a list that had in fact been retrieved for it
    # and was being cited in the same paragraph. It is reported rather than gated
    # because a correct refusal also matches none of the required content
    # (color_side_guardrail is exactly that), so gating here would fail correct turns.
    zero_inclusion = [
        result.outcome.case_id
        for result in results
        if result.outcome.inclusion_total and result.outcome.inclusion_passed == 0
    ]
    if zero_inclusion:
        print()
        print(f"未命中任何必需内容（需人工确认是否漏答）: {', '.join(zero_inclusion)}")
        for result in results:
            if result.outcome.case_id in zero_inclusion:
                print(f"  ? {result.outcome.case_id}: 缺少 {result.outcome.missing_inclusions}")

    # ----------------------------------------------------------------- baseline
    regressions: list[str] = []
    known: list[str] = []
    flaky: list[str] = []
    if not args.ignore_baseline and BASELINE_PATH.is_file():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        baseline_cases = baseline.get("cases", {})
        reasons = baseline.get("flaky_reasons", {})
        for result in results:
            recorded = baseline_cases.get(result.outcome.case_id)
            if recorded == "pass" and not result.outcome.passed:
                regressions.append(result.outcome.case_id)
            elif recorded == "fail" and not result.outcome.passed:
                known.append(result.outcome.case_id)
            elif recorded == "flaky" and not result.outcome.passed:
                # Measured non-determinism, so neither outcome alone is a regression.
                # Reported every run so it stays visible rather than silently excused.
                flaky.append(result.outcome.case_id)
        if regressions:
            print()
            print(f"基线回归（此前通过、现在失败）: {', '.join(regressions)}")
        if known:
            print()
            print(f"基线已记录的已知失败（非本次引入）: {', '.join(known)}")
        if flaky:
            print()
            print(f"抖动用例本次未通过（工具选择随采样变化，不计为回归）: {', '.join(flaky)}")
            for case_id in flaky:
                if reasons.get(case_id):
                    print(f"  ! {case_id}: {reasons[case_id]}")

    if args.write_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if BASELINE_PATH.is_file():
            existing = json.loads(BASELINE_PATH.read_text(encoding="utf-8")).get("cases", {})
        existing.update(
            {result.outcome.case_id: ("pass" if result.outcome.passed else "fail") for result in results}
        )
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "note": "Recorded by scripts/eval_agent.py --write-baseline. A case recorded "
                            "'pass' that later fails is reported as a regression.",
                    "gates": list(GATES),
                    "cases": dict(sorted(existing.items())),
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"基线已写入 {BASELINE_PATH.relative_to(PROJECT_ROOT)}")

    print()
    print("=" * 88)
    if regressions:
        print(f"RESULT: FAIL — {len(regressions)} 个用例相对基线回归")
        return 1
    if failures:
        # Without --allow-known-failures a failure always fails, which is the right
        # default. With it, failures the baseline already records (as failing or as
        # flaky) do not block, so a check can be wired up while known gaps are being
        # worked on without turning those gaps into permanent red.
        excused = set(known) | set(flaky)
        blocking = failures if not args.allow_known_failures else [
            result for result in failures if result.outcome.case_id not in excused
        ]
        if blocking:
            print(f"RESULT: FAIL — {len(blocking)}/{total} 个用例未达阈值")
            for result in blocking:
                failed = [n for n in GATES if not result.outcome.gates.get(n)]
                print(f"  - {result.outcome.case_id}: {', '.join(failed) or result.outcome.error}")
            return 1
        print(
            f"RESULT: PASS — {total - len(excused)}/{total} 达标；"
            f"{len(known)} 个已知失败 + {len(flaky)} 个抖动（见 {BASELINE_PATH.name}）"
        )
        for case_id in known:
            print(f"  ! 已知: {case_id}")
        for case_id in flaky:
            print(f"  ~ 抖动: {case_id}")
        return 0
    print(f"RESULT: PASS — {total}/{total} 个用例达到阈值")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
