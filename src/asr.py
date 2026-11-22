"""语音识别：faster-whisper，CPU int8。

为什么用 small：medium 在本机 6C12T 上 RTF 约 1.0~1.5，
也就是说 3 秒语音要算 3 秒以上，语音助手会明显迟钝。small 的 RTF 约 0.3。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import log

# whisper 在静音/噪声上常见的幻觉输出，直接丢弃
_NOISE = {
    "谢谢观看", "谢谢大家", "感谢观看", "字幕由", "请不吝点赞", "订阅",
    "thanks for watching", "thank you", "you", "bye", ".", "。", "",
}


def _local_snapshot(root: Path, name: str) -> Path | None:
    """在 download_root 里找已下载好的模型目录，找到就直接加载，不再联网校验。

    huggingface_hub 的缓存布局是
      whisper/models--Systran--faster-whisper-small/snapshots/<hash>/model.bin
    直接把这个 snapshots/<hash> 目录喂给 WhisperModel，等于完全离线。
    """
    bins = list(root.rglob("model.bin"))
    if not bins:
        return None
    key = name.lower()
    hit = [p for p in bins if key in str(p.parent).lower()]
    pick = max(hit or bins, key=lambda p: p.stat().st_mtime)
    return pick.parent


def _looks_like_noise(text: str, dur: float) -> bool:
    """过滤 whisper 在噪声/静音上的典型幻觉。

    - 短句里单字重复：「啊啊啊啊…」「，，，，」
    - 单位时间字数远超正常语速（中文 4~5 字/秒，超过 8 字/秒必是幻觉）
    """
    t = text.strip("，。、！？,.!? 　")
    if len(t) <= 1:
        return True
    if t.lower() in _NOISE:
        return True
    if len(t) >= 4 and len(set(t)) <= 2:
        return True
    if dur > 0.5 and len(t) / dur > 8:
        return True
    return False


class WhisperASR:
    def __init__(self, cfg):
        from faster_whisper import WhisperModel

        model_dir = cfg.path(cfg.get("asr.model_dir", "models/whisper"))
        model_dir.mkdir(parents=True, exist_ok=True)
        name = cfg.get("asr.model", "small")
        threads = int(cfg.get("asr.threads", 4))

        local = _local_snapshot(model_dir, name)
        target = str(local) if local else name
        log.info(f"加载 ASR: whisper-{name} int8 ({threads} 线程)"
                 f"{' [本地]' if local else ' [首次需下载]'}...")
        t0 = time.time()
        self.model = WhisperModel(
            target,
            device="cpu",
            compute_type=cfg.get("asr.compute_type", "int8"),
            cpu_threads=threads,
            download_root=str(model_dir),
        )
        self.language = cfg.get("asr.language", "zh")
        self.initial_prompt = cfg.get("asr.initial_prompt")
        log.info(f"ASR 就绪 ({time.time() - t0:.1f}s)")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or audio.size == 0:
            return ""
        t0 = time.time()
        dur = audio.size / 16000
        segments, _info = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            initial_prompt=self.initial_prompt,
            # 下面三个是防止「噪声被识别成几百个啊」的关键：
            # 不加 max_new_tokens 时，一段 1 秒噪声能生成 400 个 token，白等 5 秒
            repetition_penalty=1.05,
            max_new_tokens=int(dur * 6) + 8,
        )
        text = "".join(seg.text for seg in segments).strip()
        text = text.strip("，。、！？,.!? ")
        if _looks_like_noise(text, dur):
            text = ""
        log.info(f"ASR {time.time() - t0:.2f}s (音频 {dur:.2f}s) -> "
                 f"{(text[:60] + '…') if len(text) > 60 else (text or '(空)')}")
        return text
