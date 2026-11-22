"""Windows 控制台输出兜底：不改代码页，只把无法编码的字符替换掉。

为什么不直接切 65001：切了之后控制台仍按原来的代码页解码 PowerShell 的输出，
反而把中文显示成乱码。让 Python 跟随控制台现有编码（GBK 也能显示中文），
只加 errors="replace" 防止个别字符抛 UnicodeEncodeError 崩掉。
"""
from __future__ import annotations

import sys


def setup() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
