"""端到端自测：不依赖麦克风，直接跑 文本→LLM→分句→TTS→播放，以及 TTS 音频→ASR。

  .venv\\Scripts\\python.exe scripts\\e2e_test.py
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from _console import setup

    setup()
except Exception:
    pass

from src.config import load  # noqa: E402
from src.pipeline import Assistant  # noqa: E402


def main() -> int:
    cfg = load(None)
    print("== 初始化（VAD / LLM 探活 / ASR / TTS）==")
    a = Assistant(cfg, use_wake=False, use_mic=False)
    try:
        print("\n== 对话链路：文本 -> LLM 流式 -> 分句 -> TTS -> 扬声器 ==")
        for q in ["你好，用一句话介绍你自己。", "上海在哪个省？"]:
            print(f"\n问：{q}")
            t0 = time.time()
            a.respond(q)
            print(f"  耗时 {time.time() - t0:.2f}s")

        print("\n== 识别链路：Piper 合成的音频 -> ASR ==")
        wav = ROOT / "out" / "asr_probe.wav"
        wav.parent.mkdir(exist_ok=True)
        text = "今天天气不错，我们出去走走吧。"
        audio, sr = a.tts.synthesize(text)
        if audio is None:
            print("  TTS 不可用，跳过")
            return 1
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        print(f"  合成 {audio.size / sr:.2f}s 音频 -> {wav.name}")

        a16 = audio if sr == 16000 else np.interp(
            np.linspace(0, audio.size - 1, int(audio.size * 16000 / sr)),
            np.arange(audio.size), audio).astype(np.float32)
        got = a.asr.transcribe(a16)
        print(f"  原句：{text}")
        print(f"  识别：{got}")
        ok = bool(got)
        print("  识别链路" + ("正常" if ok else "异常：返回空"))
        return 0 if ok else 1
    finally:
        a.close()


if __name__ == "__main__":
    sys.exit(main())
