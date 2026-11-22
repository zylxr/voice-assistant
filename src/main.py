"""入口。

  python -m src.main                 # 语音全链路（唤醒词）
  python -m src.main --no-wake       # 回车触发，跳过唤醒词
  python -m src.main --text          # 纯文本模式，只测 LLM + TTS
  python -m src.main --list-devices  # 查看音频设备编号
"""
from __future__ import annotations

import argparse
import sys

from . import log
from .audio_io import list_devices
from .config import load


def main() -> int:
    ap = argparse.ArgumentParser(description="本地语音助手")
    ap.add_argument("--config", default=None, help="配置文件路径")
    ap.add_argument("--no-wake", action="store_true", help="不加载唤醒词，改用回车触发")
    ap.add_argument("--text", action="store_true", help="纯文本模式，不打开麦克风")
    ap.add_argument("--list-devices", action="store_true", help="列出音频设备")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    cfg = load(args.config)
    try:
        from .pipeline import Assistant

        if args.text:
            assistant = Assistant(cfg, use_wake=False, use_mic=False)
            try:
                assistant.run_text()
            finally:
                assistant.close()
            return 0

        assistant = Assistant(cfg, use_wake=not args.no_wake, use_mic=True)
        try:
            assistant.run()
        finally:
            assistant.close()
            log.info("已退出")
        return 0
    except KeyboardInterrupt:
        print()
        log.info("已退出")
        return 0
    except Exception as exc:
        log.err(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
