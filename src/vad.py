"""语音活动检测与断句。

两个后端：
  onnx   —— silero-vad v5 的 ONNX 直推，不依赖 torch，单帧 512 采样
  energy —— 纯 RMS 能量门限，模型缺失时的兜底，够用但怕噪声

UtteranceDetector 把连续帧流变成「一句话」：检测开口 → 累积 → 静音到时长收尾。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import log


class SileroVADOnnx:
    def __init__(self, model_path: Path, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.sess.get_inputs()}
        self._use_state = "state" in self._inputs
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    def prob(self, frame: np.ndarray) -> float:
        x = frame.astype(np.float32).reshape(1, -1)
        try:
            if self._use_state:
                feeds = {"input": x, "state": self._state, "sr": self._sr}
                p, self._state = self.sess.run(None, feeds)
            else:  # v4 风格的 h/c 状态
                feeds = {"input": x, "h": self._h, "c": self._c, "sr": self._sr}
                p, self._h, self._c = self.sess.run(None, feeds)
            return float(np.asarray(p).reshape(-1)[0])
        except Exception as exc:
            log.err(f"VAD 推理失败: {exc}")
            return 0.0


class EnergyVAD:
    def __init__(self, threshold: float = 0.015):
        self.threshold = threshold

    def reset(self) -> None:
        pass

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))) + 1e-9)
        return 1.0 if rms >= self.threshold else 0.0


def build_vad(cfg):
    backend = (cfg.get("vad.backend") or "onnx").lower()
    if backend == "onnx":
        model = cfg.path(cfg.get("vad.model"))
        if model and Path(model).exists():
            try:
                vad = SileroVADOnnx(model)
                log.info(f"VAD: silero-vad onnx ({Path(model).name})")
                return vad
            except Exception as exc:
                log.warn(f"silero-vad 加载失败，降级为能量检测: {exc}")
        else:
            log.warn(f"未找到 VAD 模型 {model}，降级为能量检测")
    vad = EnergyVAD(float(cfg.get("vad.energy_threshold", 0.015)))
    log.info("VAD: energy (RMS 门限)")
    return vad


class UtteranceDetector:
    """帧进，句出。push() 返回一句话的 float32 波形，或 None。"""

    def __init__(self, vad, threshold: float = 0.5, start_frames: int = 2,
                 min_speech_ms: int = 250, end_silence_ms: int = 600,
                 max_utterance_s: float = 20.0, pre_roll_frames: int = 6,
                 frame_ms: float = 32.0):
        self.vad = vad
        self.threshold = threshold
        self.start_frames = start_frames
        self.min_speech_frames = int(min_speech_ms / frame_ms)
        self.end_silence_frames = int(end_silence_ms / frame_ms)
        self.max_frames = int(max_utterance_s * 1000 / frame_ms)
        self.pre_roll_frames = pre_roll_frames
        self.frame_ms = frame_ms
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self._pre: list[np.ndarray] = []
        self._buf: list[np.ndarray] = []
        self._hits = 0
        self._silence = 0
        self._speech_frames = 0
        try:
            self.vad.reset()
        except Exception:
            pass

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        voiced = self.vad.prob(frame) >= self.threshold

        if not self.in_speech:
            self._pre.append(frame)
            if len(self._pre) > self.pre_roll_frames:
                self._pre.pop(0)
            self._hits = self._hits + 1 if voiced else 0
            if self._hits >= self.start_frames:
                self.in_speech = True
                self._buf = list(self._pre)
                self._pre = []
                self._silence = 0
                self._speech_frames = len(self._buf)
                self._hits = 0
            return None

        self._buf.append(frame)
        if voiced:
            self._speech_frames += 1
            self._silence = 0
        else:
            self._silence += 1

        done = self._silence >= self.end_silence_frames or len(self._buf) >= self.max_frames
        if not done:
            return None

        keep = len(self._buf) - max(0, self._silence - 2)
        speech_frames = self._speech_frames
        audio = np.concatenate(self._buf[:max(1, keep)]) if self._buf else np.zeros(0, np.float32)
        self.reset()

        if speech_frames < self.min_speech_frames or audio.size == 0:
            return None
        return audio

    def flush(self) -> Optional[np.ndarray]:
        """流结束时强制收尾。

        正常的麦克风场景不需要它——人说完话后静音帧会自然到达。
        但处理文件、或录音被外部中断时，末尾没有足够静音，
        就需要它把剩下的半句话吐出来。
        """
        if not self.in_speech or not self._buf:
            self.reset()
            return None
        audio = np.concatenate(self._buf)
        speech_frames = self._speech_frames
        self.reset()
        if speech_frames < self.min_speech_frames:
            return None
        return audio
