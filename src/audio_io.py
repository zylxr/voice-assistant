"""音频输入输出。

MicStream: 固定块长采集 int16，直接丢进队列，供 VAD / 唤醒词消费。
Player:    独立线程顺序播放，支持中断（打断）与代数（generation）机制丢弃过期音频。
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import sounddevice as sd

from . import log


class MicStream:
    def __init__(self, sample_rate: int = 16000, block_ms: int = 32, device=None):
        self.sr = sample_rate
        self.blocksize = int(sample_rate * block_ms / 1000)
        self.block_ms = block_ms
        self.device = device
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.blocksize,
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # overflow 时丢块即可，不影响链路
            pass
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def read(self, timeout: float = 0.5) -> np.ndarray:
        return self._q.get(timeout=timeout)

    def try_read(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def clear(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    @property
    def frame(self) -> int:
        return self.blocksize


class Player:
    """顺序播放队列。play() 非阻塞；interrupt() 立刻静音并作废未播音频。"""

    def __init__(self, device=None):
        self.device = device
        self._q: queue.Queue = queue.Queue()
        self._gen = 0
        self._lock = threading.Lock()
        self._busy = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="player", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _current_gen(self) -> int:
        with self._lock:
            return self._gen

    def _run(self) -> None:
        while not self._closed:
            try:
                gen, audio, sr = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if gen != self._current_gen():
                    continue
                self._busy.set()
                sd.play(audio, sr, device=self.device, blocking=True)
            except Exception as exc:  # 设备占用/采样率不支持
                log.err(f"播放失败: {exc}")
            finally:
                self._busy.clear()
                self._q.task_done()

    def play(self, audio: np.ndarray, sr: int) -> None:
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        if audio.size == 0:
            return
        self._q.put((self._current_gen(), audio, int(sr)))

    def interrupt(self) -> None:
        with self._lock:
            self._gen += 1
        try:
            sd.stop()
        except Exception:
            pass
        while True:
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break

    def busy(self) -> bool:
        return self._busy.is_set() or not self._q.empty()

    def wait_done(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        while self.busy() and time.time() < deadline:
            time.sleep(0.02)

    def close(self) -> None:
        self._closed = True
        self.interrupt()


def beep(freq: int = 880, ms: int = 90, sr: int = 16000, amp: float = 0.22) -> np.ndarray:
    """开场提示音，免去打包音频文件。加淡入淡出避免爆音。"""
    n = int(sr * ms / 1000)
    t = np.arange(n) / sr
    wave = np.sin(2 * np.pi * freq * t) * amp
    fade = max(1, int(n * 0.15))
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return wave.astype(np.float32)


def list_devices() -> None:
    print(sd.query_devices())
    try:
        print("\n默认输入:", sd.query_devices(kind="input")["name"])
        print("默认输出:", sd.query_devices(kind="output")["name"])
    except Exception as exc:
        print("查询默认设备失败:", exc)
