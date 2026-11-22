"""语音合成：Piper（纯离线，CPU 实时）。

Windows 上 piper 的 Python 包偶发装不上（依赖 phonemize 的编译轮子），
所以这里做了三级降级，任何一级能用就行：
  1. piper 的 Python API（最好，无进程开销）
  2. piper.exe 子进程（最稳，多约 100ms 启动开销）
  3. pyttsx3 / Windows SAPI（零依赖，音色差，仅作保底）
"""
from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from . import log


def _read_wav(path_or_buf) -> tuple[np.ndarray, int]:
    with wave.open(str(path_or_buf) if isinstance(path_or_buf, (str, Path)) else path_or_buf, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        channels = wf.getnchannels()
        width = wf.getsampwidth()
    if width != 2:
        raise ValueError(f"仅支持 16bit wav，实际 {width * 8}bit")
    data = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]
    return (data.astype(np.float32) / 32768.0), sr


class PiperTTS:
    def __init__(self, cfg):
        self.model = cfg.path(cfg.get("tts.model"))
        self.config = cfg.path(cfg.get("tts.config"))
        self.exe = cfg.path(cfg.get("tts.exe"))
        self.timeout = float(cfg.get("tts.timeout_s", 20))
        self.backend = None
        self.sr = 22050
        self._engine = None

        if self.model and Path(self.model).exists():
            if self._init_python_api():
                return
            if self._init_exe():
                return
            log.warn("piper 的 Python API 与 exe 都不可用")
        else:
            log.warn(f"未找到 Piper 模型 {self.model}")

        self._init_sapi()

    # ---------- 后端初始化 ----------
    def _init_python_api(self) -> bool:
        try:
            import piper
            from piper import PiperVoice

            self.voice = PiperVoice.load(
                str(self.model),
                config_path=str(self.config) if self.config and Path(self.config).exists() else None,
            )
            self.sr = int(getattr(self.voice.config, "sample_rate", 22050))
            self.new_api = hasattr(piper, "SynthesisConfig")
            self.backend = "python"
            log.info(f"TTS: piper python API ({Path(self.model).name}, {self.sr}Hz)")
            return True
        except Exception as exc:
            log.warn(f"piper python API 不可用: {exc}")
            return False

    def _init_exe(self) -> bool:
        if not (self.exe and Path(self.exe).exists()):
            return False
        self.backend = "exe"
        log.info(f"TTS: piper.exe ({Path(self.exe).name})")
        return True

    def _init_sapi(self) -> None:
        import pyttsx3

        self._engine = pyttsx3.init()
        self.backend = "sapi"
        log.warn("TTS: 降级到 Windows SAPI，音色会明显机械")

    # ---------- 合成 ----------
    def synthesize(self, text: str) -> tuple[np.ndarray | None, int]:
        text = (text or "").strip()
        if not text:
            return None, self.sr
        try:
            if self.backend == "python":
                return self._synth_python(text)
            if self.backend == "exe":
                return self._synth_exe(text)
            return self._synth_sapi(text)
        except Exception as exc:
            log.err(f"TTS 失败({self.backend}): {exc}")
            return None, self.sr

    def _synth_python(self, text: str):
        if getattr(self, "new_api", False):
            # piper >= 1.3：synthesize() 按句返回 AudioChunk
            parts: list[np.ndarray] = []
            for piece in self.voice.synthesize(text):
                self.sr = int(getattr(piece, "sample_rate", self.sr))
                arr = getattr(piece, "audio_float_array", None)
                if arr is not None:
                    parts.append(np.asarray(arr, dtype=np.float32))
                    continue
                ints = getattr(piece, "audio_int16_array", None)
                if ints is None:
                    ints = np.frombuffer(piece.audio_int16_bytes, dtype=np.int16)
                parts.append(np.asarray(ints, dtype=np.int16).astype(np.float32) / 32768.0)
            if not parts:
                return None, self.sr
            return np.concatenate(parts), self.sr

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize(text, wf)
        buf.seek(0)
        return _read_wav(buf)

    def _synth_exe(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "tts.wav"
            cmd = [str(self.exe), "-m", str(self.model), "-f", str(out)]
            if self.config and Path(self.config).exists():
                cmd += ["-c", str(self.config)]
            proc = subprocess.run(
                cmd, input=text.encode("utf-8"), capture_output=True, timeout=self.timeout
            )
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(proc.stderr.decode("utf-8", "ignore")[:300])
            return _read_wav(out)

    def _synth_sapi(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "tts.wav"
            self._engine.save_to_file(text, str(out))
            self._engine.runAndWait()
            if not out.exists():
                return None, self.sr
            return _read_wav(out)
