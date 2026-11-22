"""流式分句：把 LLM 的增量 token 攒成适合朗读的小句。

这是整条链路里最影响体感的一环。如果等整段回答生成完再送 TTS，
7B Q4 在 CPU 上生成 100 字要十几秒，用户会以为程序卡死。
按标点切成 10~20 字的小句，第一句一到就开念，播放速度就能盖住生成速度。
"""
from __future__ import annotations

import re

STRONG = "。！？!?；;…\n"
WEAK = "，,、：:"
# 朗读前清掉 markdown 与装饰性符号
JUNK = re.compile(r"[*#`>~\[\]{}|\\^_]")
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0000FE0F]"
)


def clean(text: str) -> str:
    text = JUNK.sub("", text)
    text = EMOJI.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ，,、：:")


class StreamSplitter:
    def __init__(self, min_chars: int = 8, max_chars: int = 24):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.buf = ""

    def reset(self) -> None:
        self.buf = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            piece = clean(self.buf[:cut])
            self.buf = self.buf[cut:]
            if piece:
                out.append(piece)
        return out

    def flush(self) -> list[str]:
        piece = clean(self.buf)
        self.buf = ""
        return [piece] if piece else []

    def _find_cut(self) -> int | None:
        for i, ch in enumerate(self.buf):
            if ch in STRONG and i >= 2:
                return i + 1
        for i, ch in enumerate(self.buf):
            if ch in WEAK and i >= self.min_chars:
                return i + 1
        if len(self.buf) >= self.max_chars:
            # 没有标点也要切，找一个尽量不劈开数字/英文词的位置
            for i in range(self.max_chars, max(self.min_chars, 1) - 1, -1):
                if i < len(self.buf) and not self.buf[i].isalnum():
                    return i + 1
            return self.max_chars
        return None
