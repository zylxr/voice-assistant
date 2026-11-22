"""离线自测：不需要大模型，验证 TTS 合成 + VAD 断句两条链路。

  python scripts/smoke_test.py

产物写到 out/smoke_tts.wav，可以直接双击听音色。
"""
from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np

try:
    from _console import setup as _setup_console

    _setup_console()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load  # noqa: E402
from src.tts import PiperTTS  # noqa: E402
from src.vad import UtteranceDetector, build_vad  # noqa: E402

TEXT = "今天天气不错，适合出去走走，顺便买杯咖啡。"


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x
    n = int(round(x.size * sr_out / sr_in))
    return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(audio, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main() -> int:
    cfg = load()
    print(f"项目根目录: {cfg.root}")

    print("\n=== 1. TTS ===")
    t0 = time.time()
    tts = PiperTTS(cfg)
    print(f"后端: {tts.backend}")
    t0 = time.time()
    audio, sr = tts.synthesize(TEXT)
    cost = time.time() - t0
    if audio is None or audio.size == 0:
        print("  [FAIL] 合成结果为空")
        return 1
    dur = audio.size / sr
    peak = float(np.max(np.abs(audio)))
    print(f"  [OK] {audio.size} 采样 @ {sr}Hz = {dur:.2f}s，合成耗时 {cost:.2f}s")
    print(f"       实时率 {dur / max(cost, 1e-6):.1f}x，峰值 {peak:.2f}")
    out = ROOT / "out" / "smoke_tts.wav"
    write_wav(out, audio, sr)
    print(f"       已写入 {out}")

    print("\n=== 2. VAD 断句 ===")
    t0 = time.time()
    vad = build_vad(cfg)
    print(f"后端: {type(vad).__name__}（加载 {time.time() - t0:.2f}s）")
    det = UtteranceDetector(
        vad,
        threshold=float(cfg.get("vad.threshold", 0.5)),
        start_frames=int(cfg.get("vad.start_frames", 2)),
        min_speech_ms=int(cfg.get("vad.min_speech_ms", 250)),
        end_silence_ms=int(cfg.get("vad.end_silence_ms", 600)),
        frame_ms=float(cfg.get("audio.block_ms", 32)),
    )
    audio16 = resample(audio, sr, 16000)
    # Piper 输出结尾几乎没有静音，补 1 秒静音模拟真实麦克风（人说完话后的沉默）
    audio16 = np.concatenate([audio16, np.zeros(16000, dtype=np.float32)])
    frame = 512
    frames = audio16.size // frame
    t0 = time.time()
    got = None
    for i in range(frames):
        piece = det.push(audio16[i * frame:(i + 1) * frame])
        if piece is not None and got is None:
            got = piece
    if got is None:
        got = det.flush()
    vad_cost = time.time() - t0
    if got is None:
        print(f"  [FAIL] {frames} 帧里没有检测到一句完整语音")
        print("         可能原因: 阈值过高，或 VAD 模型与代码不匹配")
        return 1
    secs = got.size / 16000
    print(f"  [OK] 断出一句 {secs:.2f}s（原始 {audio16.size / 16000:.2f}s）")
    print(f"       {frames} 帧耗时 {vad_cost:.2f}s = 每帧 {vad_cost / frames * 1000:.2f}ms")

    print("\n通过。下一步: python scripts/download_models.py --only llm,llama 然后跑 run_server.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
