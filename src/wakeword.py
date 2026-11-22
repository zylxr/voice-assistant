"""唤醒词检测（openWakeWord，ONNX 后端）。

openWakeWord 的预训练模型只支持英文关键词（hey_jarvis / alexa / hey_mycroft 等），
中文唤醒词需要自己训练。这里用英文关键词，命中后进入一句话录制的流程。

注意：openWakeWord 每次推理需要 1280 采样（80ms）的 int16 块，
而麦克风按 32ms 出块，所以内部要攒够再送。
"""
from __future__ import annotations

import numpy as np

from . import log

CHUNK = 1280  # 80ms @ 16kHz


class WakeWord:
    def __init__(self, model_path=None, model_name: str = "hey_jarvis_v0.1",
                 threshold: float = 0.55):
        self.threshold = threshold
        self._buf = np.zeros(0, dtype=np.int16)
        self._score = 0.0
        from openwakeword.model import Model  # 延迟导入，缺依赖时能给出友好报错

        targets = []
        if model_path and model_path.exists():
            targets = [str(model_path)]
        else:
            targets = [model_name]
            log.warn(f"未找到唤醒词文件 {model_path}，改用内置模型名 {model_name}")

        self.model = Model(wakeword_models=targets, inference_framework="onnx")
        log.info(f"唤醒词: {', '.join(targets)} (阈值 {threshold})")

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        self._score = 0.0
        try:
            self.model.reset()
        except Exception:
            pass

    def push(self, block: np.ndarray) -> float:
        """喂一个麦克风块，返回最近一次推理的最高分。"""
        self._buf = np.concatenate([self._buf, block.astype(np.int16, copy=False)])
        while self._buf.size >= CHUNK:
            chunk, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
            try:
                preds = self.model.predict(chunk)
                if preds:
                    self._score = max(float(v) for v in preds.values())
            except Exception as exc:
                log.err(f"唤醒词推理失败: {exc}")
                self._score = 0.0
        return self._score

    def triggered(self, block: np.ndarray) -> bool:
        return self.push(block) >= self.threshold
