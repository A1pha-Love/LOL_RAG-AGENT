"""Run a single LoL data question from the terminal."""

import os
import sys
from pathlib import Path

# The Windows project environment currently defaults to GBK when stdout is
# captured. Emit UTF-8 so Chinese answers remain readable in PowerShell logs.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from lol_agent.agent import ask


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("请输入英雄联盟数据问题：").strip()
    if not question:
        raise SystemExit("问题不能为空")
    try:
        print(ask(question))
    except Exception as exc:
        message = str(exc)
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if api_key:
            message = message.replace(api_key, "[已隐藏]")
        print(f"Agent 调用失败（{type(exc).__name__}）：{message}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
