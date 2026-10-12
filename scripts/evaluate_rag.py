"""Run the RAG/Agent evaluation cases against the configured Qwen endpoint."""

from __future__ import annotations

import json
import argparse
import re
import sys
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from lol_agent.agent import build_agent  # noqa: E402


def _load_cases() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "evals" / "rag_cases.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _parse_tool_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _normalize_phrase(text: str) -> str:
    """Normalize whitespace and punctuation so equivalent formatting still matches."""
    return re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize, and record which raw index each normalized character came from.

    Quote detection cannot run on the normalized text because normalization strips the
    quote characters themselves, so positions have to be traced back to the original.
    """
    normalized: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text.casefold()):
        if re.match(r"[\W_]", character, flags=re.UNICODE):
            continue
        normalized.append(character)
        positions.append(index)
    return "".join(normalized), positions


def _requirement_options(requirement: str | list[str]) -> list[str]:
    return requirement if isinstance(requirement, list) else [requirement]


def _display_requirement(requirement: str | list[str]) -> str:
    return " / ".join(_requirement_options(requirement))


def _answer_inclusion_checks(case: dict[str, Any], answer: str) -> list[tuple[str, bool]]:
    normalized_answer = _normalize_phrase(answer)
    checks = []
    for requirement in case.get("answer_must_include", []):
        options = _requirement_options(requirement)
        matched = any(
            (phrase := _normalize_phrase(option)) and phrase in normalized_answer
            for option in options
        )
        checks.append((" / ".join(options), matched))
    return checks


_NEGATION_PREFIX = re.compile(
    r"(?:无法|不能|不得|不应|不支持|不代表|不意味着|不声称|不作|不推断|没有|不存在|并非|不是|严禁|禁止|不可)\w{0,16}$"
)
_NEGATION_SUFFIX = re.compile(r"^(?:不成立|不正确|不代表|不是|并非|不存在|无法|不能|是错误的|属于错误说法)")

# Prohibitions stated as a rule rather than a denial of a specific sentence, e.g.
# "不允许自行评价关联强弱（例如说"强正向相关"）". The distance between the marker and
# the quoted phrase can exceed the negation window, so these are matched separately.
_PROHIBITION = re.compile(
    r"(?:不允许|不得|禁止|严禁|不可|切勿|不应|不能|避免|只允许|只报告|不自行)"
)
# Straight and curly quotes, used to present a phrase as an example of what not to say.
_QUOTE_CHARS = "\"'“”‘’「」『』《》"


def _phrase_is_inside_quotes(text: str, start: int, end: int) -> bool:
    """Whether text[start:end] sits inside a quoted span."""
    return text[start - 1:start] in _QUOTE_CHARS and text[end:end + 1] in _QUOTE_CHARS


def _phrase_has_local_negation(
    normalized_answer: str,
    start: int,
    end: int,
    *,
    raw_answer: str = "",
    positions: list[int] | None = None,
) -> bool:
    before = normalized_answer[max(0, start - 24):start]
    after = normalized_answer[end:end + 16]
    if _NEGATION_PREFIX.search(before) or _NEGATION_SUFFIX.search(after):
        return True
    # A quoted phrase introduced by a prohibition is being cited as what not to say,
    # not asserted. Quoting alone is deliberately not enough: "所谓"强正向相关"确实成立"
    # quotes the phrase while still asserting it, and that must remain a failure.
    if raw_answer and positions and positions:
        raw_start = positions[start]
        raw_end = positions[end - 1] + 1
        quoted = (
            raw_answer[raw_start - 1:raw_start] in _QUOTE_CHARS
            and raw_answer[raw_end:raw_end + 1] in _QUOTE_CHARS
        )
        if quoted and _PROHIBITION.search(normalized_answer[max(0, start - 60):start]):
            return True
    return False


def _forbidden_claim_checks(case: dict[str, Any], answer: str) -> tuple[list[str], list[str]]:
    """Return likely unnegated hits and phrases seen only in a local negation."""
    normalized_answer, positions = _normalize_with_map(answer)
    unnegated: list[str] = []
    negated: list[str] = []
    for claim in case.get("answer_must_not_claim", []):
        phrase = _normalize_phrase(claim)
        if not phrase:
            continue
        starts: list[int] = []
        cursor = 0
        while (found := normalized_answer.find(phrase, cursor)) != -1:
            starts.append(found)
            cursor = found + len(phrase)
        if not starts:
            continue
        if all(
            _phrase_has_local_negation(
                normalized_answer,
                start,
                start + len(phrase),
                raw_answer=answer,
                positions=positions,
            )
            for start in starts
        ):
            negated.append(claim)
        else:
            unnegated.append(claim)
    return unnegated, negated


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _numeric_checks(case: dict[str, Any], tool_results: dict[str, list[Any]]) -> list[bool]:
    checks: list[bool] = []
    for assertion in case.get("numeric_assertions", []):
        candidates = tool_results.get(assertion["tool"], [])
        matched = False
        for candidate in candidates:
            for item in _walk_dicts(candidate):
                if all(item.get(key) == assertion.get(key) for key in ("numerator", "denominator", "unit")):
                    if "value" not in assertion or item.get("value") == assertion["value"]:
                        matched = True
                        break
            if matched:
                break
        checks.append(matched)
    return checks


def main() -> int:
    cases = _load_cases()
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", help="Run only the selected case(s); may be repeated.")
    parser.add_argument("--show-full-answer", action="store_true", help="Print the complete Agent answer instead of a 700-character preview.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail the exit code when an answer_must_include phrase is missing (heuristic, so opt-in).",
    )
    args = parser.parse_args()
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            parser.error("unknown case id(s): " + ", ".join(sorted(missing)))
    graph = build_agent()
    overall = {
        "tools": 0,
        "retrieval": 0,
        "numeric_passed": 0,
        "numeric_total": 0,
        "inclusion_passed": 0,
        "inclusion_total": 0,
        "unnegated_claim_cases": 0,
        "cases": len(cases),
    }

    for index, case in enumerate(cases, start=1):
        result = graph.invoke({"messages": [HumanMessage(content=case["question"])]})
        messages = result["messages"]
        calls = [call for message in messages for call in getattr(message, "tool_calls", [])]
        called_names = {call.get("name") for call in calls}
        tool_pass = (
            set(case.get("expected_tools", [])).issubset(called_names)
            and not (set(case.get("forbidden_tools", [])) & called_names)
        )

        tool_results: dict[str, list[Any]] = {}
        for message in messages:
            if getattr(message, "type", None) == "tool":
                tool_results.setdefault(message.name, []).append(_parse_tool_content(message.content))
        retrieved = [
            chunk
            for response in tool_results.get("search_project_knowledge", [])
            for chunk in (
                response
                if isinstance(response, list)
                else response.get("results", []) if isinstance(response, dict) else []
            )
            if isinstance(chunk, dict)
        ]
        # Retrieval assertions match against the chunk's heading PATH. The
        # knowledge base re-chunks by heading, so an expected heading is a
        # substring of the path ("文档 > 节 > 小节") rather than an exact value.
        retrieval_misses = [
            expected
            for expected in case.get("expected_retrieval", [])
            if not any(
                chunk.get("source") == expected["source"]
                and expected["heading"] in (chunk.get("heading_path") or "")
                for chunk in retrieved
            )
        ]
        retrieval_pass = not retrieval_misses
        numeric_results = _numeric_checks(case, tool_results)
        numeric_pass: bool | None = all(numeric_results) if numeric_results else None
        answer = messages[-1].content
        if not isinstance(answer, str):
            answer = str(answer)
        inclusion_checks = _answer_inclusion_checks(case, answer)
        unnegated_claims, negated_claims = _forbidden_claim_checks(case, answer)
        inclusion_passed = sum(int(matched) for _, matched in inclusion_checks)
        inclusion_total = len(inclusion_checks)

        overall["tools"] += int(tool_pass)
        overall["retrieval"] += int(retrieval_pass)
        overall["numeric_passed"] += sum(numeric_results)
        overall["numeric_total"] += len(numeric_results)
        overall["inclusion_passed"] += inclusion_passed
        overall["inclusion_total"] += inclusion_total
        overall["unnegated_claim_cases"] += int(bool(unnegated_claims))
        numeric_label = str(numeric_pass).lower() if numeric_pass is not None else "n/a"
        print(
            f"[{index}/{len(cases)}] {case['id']} | tools={tool_pass} retrieval={retrieval_pass} "
            f"numeric={numeric_label} answer_include={inclusion_passed}/{inclusion_total} "
            f"unnegated_forbidden={len(unnegated_claims)}",
            flush=True,
        )
        print("  called:", ", ".join(sorted(called_names)) or "(none)")
        print(
            "  sources:",
            "; ".join(
                f"{x.get('source')} :: {x.get('heading_path') or x.get('heading')}" for x in retrieved
            )
            or "(none)",
        )
        if retrieval_misses:
            print(
                "  missing retrieval:",
                "; ".join(f"{item['source']} :: {item['heading']}" for item in retrieval_misses),
                flush=True,
            )
        print("  answer:", (answer if args.show_full_answer else answer[:700]).replace("\n", " "))
        print(
            "  manual must include:",
            "; ".join(_display_requirement(item) for item in case.get("answer_must_include", [])),
            flush=True,
        )
        print("  manual must not claim:", "; ".join(case.get("answer_must_not_claim", [])), flush=True)
        missing_inclusions = [phrase for phrase, matched in inclusion_checks if not matched]
        if missing_inclusions:
            print("  missing literal/alias checks:", "; ".join(missing_inclusions), flush=True)
        if unnegated_claims:
            print("  possible forbidden claims:", "; ".join(unnegated_claims), flush=True)
        elif negated_claims:
            print("  forbidden phrases appear only near negation:", "; ".join(negated_claims), flush=True)

    summary_line = (
        "SUMMARY: "
        f"tools {overall['tools']}/{overall['cases']}; "
        f"retrieval {overall['retrieval']}/{overall['cases']}; "
        f"numeric assertions {overall['numeric_passed']}/{overall['numeric_total']}; "
        f"answer inclusions {overall['inclusion_passed']}/{overall['inclusion_total']}; "
        f"cases with possible unnegated forbidden claims {overall['unnegated_claim_cases']}."
    )
    print(summary_line)
    print(
        "Phrase checks are heuristic: they screen candidates and cannot judge meaning or "
        "the scope of a negation, so important answers still need manual review."
    )

    failures = []
    if overall["tools"] != overall["cases"]:
        failures.append(f"tool selection {overall['tools']}/{overall['cases']}")
    if overall["retrieval"] != overall["cases"]:
        failures.append(f"source retrieval {overall['retrieval']}/{overall['cases']}")
    if overall["numeric_passed"] != overall["numeric_total"]:
        failures.append(f"numeric assertions {overall['numeric_passed']}/{overall['numeric_total']}")
    if overall["unnegated_claim_cases"]:
        failures.append(f"unnegated forbidden claims in {overall['unnegated_claim_cases']} case(s)")
    if args.strict and overall["inclusion_passed"] != overall["inclusion_total"]:
        failures.append(
            f"answer inclusions {overall['inclusion_passed']}/{overall['inclusion_total']} (--strict)"
        )

    if failures:
        print("RESULT: FAIL -> " + "; ".join(failures))
        return 1
    if not args.strict:
        print(
            "RESULT: PASS on tool/retrieval/numeric/forbidden checks. "
            "Answer phrasing was only screened; add --strict to also gate on it."
        )
    else:
        print("RESULT: PASS (strict).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
