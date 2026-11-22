"""极简日志，带毫秒时间戳，方便看延迟。"""
from __future__ import annotations

import sys
import time


def setup_console() -> None:
    """输出兜底：不改代码页，只把当前编码表示不了的字符替换掉。

    不要在这里切 chcp 65001——切了之后控制台仍用原代码页解码后续输出，
    中文反而变乱码。跟随控制台现有编码（GBK 本身能显示中文）最稳。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


setup_console()


def _ts() -> str:
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"


def info(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[{_ts()}] ! {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[{_ts()}] x {msg}", file=sys.stderr, flush=True)


def stage(name: str, seconds: float) -> str:
    """格式化成 [STAGE 123ms] 片段，便于拼接一行日志。"""
    return f"{name} {seconds * 1000:.0f}ms"
