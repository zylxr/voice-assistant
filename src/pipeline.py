"""主流程编排：唤醒 → 录音 → 识别 → 生成 → 播报。

关键设计是「生成和播报并行」：LLM 每吐出一小句就立刻合成并塞进播放队列，
而不是等整段回答生成完。CPU 推理速度慢，这是让体感可用的唯一办法。
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from . import log
from .asr import WhisperASR
from .audio_io import MicStream, Player, beep
from .config import Config
from .llm import LLMClient
from .textproc import StreamSplitter
from .tts import PiperTTS
from .vad import UtteranceDetector, build_vad
from .wakeword import WakeWord


class Assistant:
    def __init__(self, cfg: Config, use_wake: bool = True, use_mic: bool = True):
        self.cfg = cfg
        self.sr = int(cfg.get("audio.sample_rate", 16000))
        self.block_ms = int(cfg.get("audio.block_ms", 32))

        self.player = Player(cfg.get("audio.output_device"))
        self.player.start()

        self.mic: MicStream | None = None
        if use_mic:
            self.mic = MicStream(self.sr, self.block_ms, cfg.get("audio.input_device"))

        self.vad = build_vad(cfg)
        self.det = UtteranceDetector(
            self.vad,
            threshold=float(cfg.get("vad.threshold", 0.5)),
            start_frames=int(cfg.get("vad.start_frames", 2)),
            min_speech_ms=int(cfg.get("vad.min_speech_ms", 250)),
            end_silence_ms=int(cfg.get("vad.end_silence_ms", 600)),
            max_utterance_s=float(cfg.get("vad.max_utterance_s", 20)),
            pre_roll_frames=int(cfg.get("vad.pre_roll_frames", 6)),
            frame_ms=float(self.block_ms),
        )

        # 先探活大模型服务：它没起来的话，加载 ASR/TTS 是白等十几秒
        self.llm = LLMClient(cfg)
        ok, detail = self.llm.available()
        if not ok:
            raise RuntimeError(
                f"连不上大模型服务 {self.llm.base_url}\n  原因: {detail}\n"
                f"  请先在另一个窗口运行 run_server.bat"
            )
        log.info(f"LLM 服务就绪: {detail}")

        self.asr = WhisperASR(cfg)
        self.tts = PiperTTS(cfg)
        self.splitter = StreamSplitter(
            min_chars=int(cfg.get("split.min_chars", 8)),
            max_chars=int(cfg.get("split.max_chars", 24)),
        )

        self.wake: WakeWord | None = None
        self.barge_in = bool(cfg.get("playback.barge_in", False))
        self.barge_frames = int(cfg.get("playback.barge_in_frames", 10))
        if use_mic and use_wake and cfg.get("wake.enabled", True):
            try:
                self.wake = WakeWord(
                    cfg.path(cfg.get("wake.model")),
                    cfg.get("wake.name", "hey_jarvis_v0.1"),
                    float(cfg.get("wake.threshold", 0.55)),
                )
            except Exception as exc:
                log.warn(f"唤醒词不可用（{exc}），改为回车触发")

    # ---------------- 生命周期 ----------------
    def start(self) -> None:
        if self.mic:
            self.mic.start()

    def close(self) -> None:
        if self.mic:
            self.mic.stop()
        self.player.close()

    # ---------------- 主循环 ----------------
    def run(self) -> None:
        self.start()
        if self.wake:
            log.info("就绪。说 \"Hey Jarvis\" 激活，Ctrl+C 退出。")
        else:
            log.info("就绪。按回车开始说话，Ctrl+C 退出。")
        while True:
            if not self.wait_trigger():
                continue
            self.ack()
            audio = self.record()
            if audio is None:
                log.info("没听到有效语音，回到待机")
                continue
            text = self.asr.transcribe(audio)
            if not text:
                log.info("识别为空，回到待机")
                continue
            self.respond(text)

    def wait_trigger(self) -> bool:
        if not self.wake:
            try:
                input("\n[回车] 开始说话 > ")
            except EOFError:
                return False
            return True

        cooldown = float(self.cfg.get("wake.cooldown_s", 1.0))
        last = 0.0
        while True:
            try:
                block = self.mic.read(0.3)
            except queue.Empty:
                continue
            if self.wake.triggered(block) and time.time() - last > cooldown:
                last = time.time()
                log.info(">>> 唤醒")
                return True

    def ack(self) -> None:
        if not self.cfg.get("wake.ack", True):
            return
        self.player.play(beep(660, 70, self.sr), self.sr)
        self.player.play(beep(990, 90, self.sr), self.sr)

    def record(self) -> np.ndarray | None:
        if not self.mic:
            return None
        self.mic.clear()
        self.det.reset()
        started = time.time()
        while True:
            if time.time() - started > 25:
                return None
            try:
                block = self.mic.read(0.2)
            except queue.Empty:
                if not self.det.in_speech and time.time() - started > 6:
                    return None  # 唤醒后一直没开口
                continue
            frame = block.astype(np.float32) / 32768.0
            audio = self.det.push(frame)
            if audio is not None:
                log.info(f"录音 {audio.size / self.sr:.2f}s")
                return audio

    def respond(self, user_text: str) -> None:
        self.splitter.reset()
        stop = threading.Event()
        t_start = time.time()
        first_audio = {"t": None}

        def producer() -> None:
            try:
                for delta in self.llm.stream(user_text):
                    if stop.is_set():
                        return
                    for sentence in self.splitter.feed(delta):
                        if stop.is_set():
                            return
                        self._speak(sentence, t_start, first_audio)
                for sentence in self.splitter.flush():
                    if stop.is_set():
                        return
                    self._speak(sentence, t_start, first_audio)
            except Exception as exc:
                log.err(f"生成/合成失败: {exc}")
                stop.set()

        th = threading.Thread(target=producer, name="llm-tts", daemon=True)
        th.start()

        interrupted = False
        while th.is_alive() or self.player.busy():
            if self.barge_in and self.mic and not interrupted and self._barge_detected():
                log.warn("检测到打断，停止播报")
                interrupted = True
                stop.set()
                self.player.interrupt()
            time.sleep(0.03)

        self.player.wait_done(5.0)
        total = time.time() - t_start
        first = f"{first_audio['t']:.2f}s" if first_audio["t"] is not None else "无"
        log.info(f"本轮结束: 首响 {first}, 全程 {total:.2f}s")

    def _speak(self, text: str, t_start: float, first_audio: dict) -> None:
        audio, sr = self.tts.synthesize(text)
        if audio is None or audio.size == 0:
            return
        if first_audio["t"] is None:
            first_audio["t"] = time.time() - t_start
            log.info(f"首音 {first_audio['t']:.2f}s | {text}")
        else:
            log.info(f"  续播 | {text}")
        self.player.play(audio, sr)

    def _barge_detected(self) -> bool:
        hits = 0
        while True:
            block = self.mic.try_read()
            if block is None:
                break
            frame = block.astype(np.float32) / 32768.0
            if self.vad.prob(frame) > 0.6:
                hits += 1
        return hits >= self.barge_frames

    # ---------------- 纯文本模式（不开麦克风，用于逐步验证） ----------------
    def run_text(self) -> None:
        log.info("文本模式：输入问题回车，助手会朗读回答。空行退出。")
        while True:
            try:
                text = input("\n你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not text:
                return
            self.respond(text)
