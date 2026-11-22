"""LLM 客户端：连本地 llama.cpp 的 llama-server（OpenAI 兼容接口）。

这样把引擎和上层解耦——将来机器换成有显卡的，把 base_url 指到 vLLM / SGLang
就行，上层代码一行都不用改，这也是当初推荐 llama-server 的主要原因。
"""
from __future__ import annotations

from typing import Iterator

import time

from . import log


class LLMClient:
    def __init__(self, cfg):
        from openai import OpenAI

        self.base_url = cfg.get("llm.base_url", "http://127.0.0.1:8080/v1")
        self.model = cfg.get("llm.model", "qwen")
        self.temperature = float(cfg.get("llm.temperature", 0.6))
        self.max_tokens = int(cfg.get("llm.max_tokens", 300))
        self.history_turns = int(cfg.get("llm.history_turns", 4))
        self.system_prompt = (cfg.get("llm.system_prompt") or "").strip()
        self.client = OpenAI(base_url=self.base_url, api_key=cfg.get("llm.api_key", "sk-local"),
                             timeout=120.0, max_retries=0)
        self.history: list[dict] = []

    def available(self) -> tuple[bool, str]:
        try:
            models = self.client.models.list()
            names = ", ".join(m.id for m in models.data) or "(空)"
            return True, names
        except Exception as exc:
            return False, str(exc)

    def reset(self) -> None:
        self.history.clear()

    def _messages(self, user_text: str) -> list[dict]:
        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend(self.history)
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def stream(self, user_text: str) -> Iterator[str]:
        """流式产出增量文本。调用方负责把文本切片送 TTS。"""
        messages = self._messages(user_text)
        started = time.time()
        first = True
        collected: list[str] = []

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if not piece:
                continue
            if first:
                log.info(f"LLM 首字 {time.time() - started:.2f}s")
                first = False
            collected.append(piece)
            yield piece

        answer = "".join(collected).strip()
        if answer:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": answer})
            limit = self.history_turns * 2
            if len(self.history) > limit:
                self.history = self.history[-limit:]
        log.info(f"LLM 完成 {time.time() - started:.2f}s, {len(answer)} 字")
