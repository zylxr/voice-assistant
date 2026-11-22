"""测试 llama-server 的实际生成速度，判断模型规模和线程数是否合适。

  python scripts/bench_llm.py                 # 默认测 3 轮
  python scripts/bench_llm.py --url http://127.0.0.1:8080/v1

判断标准：中文语音正常语速约 4~5 字/秒，
生成速度（字/秒）必须高于这个值，否则播报会一段一段卡顿。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

try:
    from _console import setup as _setup_console

    _setup_console()
except Exception:
    pass

PROMPTS = [
    "用两句话解释什么是光合作用。",
    "简单说说北京有哪些值得去的地方。",
    "介绍一下相对论的核心思想，简短一点。",
]


def run_once(client, model: str, prompt: str) -> tuple[float, float, int]:
    t0 = time.time()
    first = None
    text = []
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=200,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = getattr(chunk.choices[0].delta, "content", None)
        if not piece:
            continue
        if first is None:
            first = time.time() - t0
        text.append(piece)
    total = time.time() - t0
    return first or total, total, len("".join(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.url, api_key="sk-local", timeout=180.0, max_retries=0)
    try:
        names = ", ".join(m.id for m in client.models.list().data)
        print(f"服务在线，模型: {names}")
    except Exception as exc:
        print(f"连不上 {args.url}: {exc}")
        print("先启动 run_server.bat")
        return 1

    ttfts, cps = [], []
    for i in range(args.rounds):
        prompt = PROMPTS[i % len(PROMPTS)]
        ttft, total, chars = run_once(client, args.model, prompt)
        rate = chars / max(total, 1e-6)
        ttfts.append(ttft)
        cps.append(rate)
        print(f"第 {i + 1} 轮: 首字 {ttft:.2f}s | 总耗时 {total:.2f}s | {chars} 字 | {rate:.1f} 字/秒")

    avg_ttft = statistics.mean(ttfts)
    avg_cps = statistics.mean(cps)
    print(f"\n平均首字 {avg_ttft:.2f}s，平均 {avg_cps:.1f} 字/秒")

    if avg_cps >= 8:
        print("结论：很流畅，7B 完全跑得动。")
    elif avg_cps >= 5:
        print("结论：刚好够用，语音播报基本不会断。")
    elif avg_cps >= 3.5:
        print("结论：偏慢，播报会有停顿。可以把线程数调到物理核心数，或换 Qwen3-4B。")
    else:
        print("结论：太慢。检查 -t 参数是否等于物理核心数、模型是否 Q4，或换更小的模型。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
