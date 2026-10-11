"""Measure how large a session transcript actually gets.

Decides whether history management needs an extra summarisation model call or
whether a cheaper deterministic trim is sufficient: if a long session still fits
comfortably in the model's context window, truncation is the right tool.

    python scripts/measure_history_size.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lol_agent import conversation as conv  # noqa: E402
from lol_agent.streaming import _last_message_text  # noqa: E402  (unused guard)
from lol_agent.agent import SYSTEM_PROMPT, session_messages  # noqa: E402

print("=== 单轮平均规模 ===")
store = conv.SessionStore()
sessions = store.list_sessions(limit=200)
print(f"  已存会话数: {len(sessions)}")

lengths: list[tuple[str, int, int]] = []
for item in sessions:
    try:
        session = store.get_session(item["session_id"])
    except conv.SessionNotFound:
        continue
    total = sum(len(turn.content) for turn in session.turns)
    if session.turns:
        lengths.append((item["session_id"][:6], len(session.turns), total))

if lengths:
    turns_avg = sum(n for _, n, _ in lengths) / len(lengths)
    chars_avg = sum(c for _, _, c in lengths) / len(lengths)
    chars_max = max(c for _, _, c in lengths)
    print(f"  平均轮数（消息条数）: {turns_avg:.1f}")
    print(f"  平均字符数          : {chars_avg:,.0f}")
    print(f"  最大字符数          : {chars_max:,}")
    print(f"  每条消息平均字符数   : {chars_avg / max(turns_avg, 1):,.0f}")
else:
    print("  （还没有会话记录；用下面的模型估算）")
    turns_avg, chars_avg = 4.0, 2000.0

print()
print("=== 模型估算：不同轮数下的提示词规模 ===")
print("  假设每轮 = 1 问 + 1 答，平均每题 40 字、每答 1200 字")
system_chars = len(SYSTEM_PROMPT)
print(f"  system prompt: {system_chars:,} 字符")
print()
print(f"  {'轮数':>6} {'转录字符':>12} {'约合 token':>12} {'占 128K 上下文':>14}")
for rounds in (5, 10, 20, 50, 100, 200, 400):
    per_round = 40 + 1200
    transcript = rounds * per_round
    tokens = transcript + system_chars
    print(f"  {rounds:>6} {transcript:>12,} {tokens:>12,} {tokens / 128_000 * 100:>13.1f}%")

print()
print("=== 结论 ===")
per_round = 1240
safe = 60_000
print(f"  若把转录预算设为 {safe:,} 字符（约 {safe / 128_000 * 100:.0f}% 的 128K 上下文），")
print(f"  可容纳约 {safe // per_round} 轮完整对话。")
print("  超过后按「保留最近若干轮 + 丢弃更早轮次」确定性裁剪即可，")
print("  无需摘要模型调用——只有需要跨很久之前的信息时才值得加摘要。")
