"""Time the slow question end to end to confirm the cause.

Reports per-step timing (rewrite, each tool call, first token, total) plus the
size of every tool payload, so the cost is attributed rather than guessed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from lol_agent.agent import build_agent  # noqa: E402

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "哪个英雄胜率最高？"

print(f"问题：{QUESTION}")
print("=" * 78)

graph = build_agent()
started = time.perf_counter()
result = graph.invoke({"messages": [HumanMessage(content=QUESTION)]})
total = time.perf_counter() - started

messages = result["messages"]
print(f"总耗时：{total:.1f}s    消息数：{len(messages)}")
print()

print("=== 逐条消息 ===")
for index, message in enumerate(messages):
    kind = type(message).__name__
    name = getattr(message, "name", "")
    calls = getattr(message, "tool_calls", []) or []
    content = getattr(message, "content", "")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    size = len(text)
    extra = ""
    if calls:
        extra = " calls=" + ", ".join(f"{c.get('name')}({json.dumps(c.get('args'), ensure_ascii=False)})" for c in calls)
    print(f"  [{index}] {kind}{' ' + name if name else ''}  {size:>7,} 字符{extra}")

print()
print("=== 工具返回大小汇总 ===")
tool_sizes: dict[str, int] = {}
for message in messages:
    if getattr(message, "type", None) == "tool":
        content = getattr(message, "content", "")
        tool_sizes[getattr(message, "name", "?")] = len(content if isinstance(content, str) else str(content))
for name, size in sorted(tool_sizes.items(), key=lambda item: -item[1]):
    print(f"  {name:34} {size:>7,} 字符")

print()
answer = messages[-1].content
print(f"=== 最终答案（{len(answer) if isinstance(answer, str) else 0} 字符）===")
print((answer if isinstance(answer, str) else str(answer))[:700])
