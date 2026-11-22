"""环境自检：把「装没装好、模型齐不齐、服务通不通」一次说清楚。

  python scripts/check_env.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

try:
    from _console import setup as _setup_console

    _setup_console()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    ("numpy", "数值计算"),
    ("sounddevice", "音频 IO"),
    ("yaml", "配置解析"),
    ("faster_whisper", "语音识别"),
    ("onnxruntime", "VAD / 唤醒词推理"),
    ("openwakeword", "唤醒词"),
    ("openai", "大模型客户端"),
    ("piper", "语音合成"),
]

MODEL_FILES = [
    ("models/whisper", "Whisper 模型目录", True),
    ("models/piper/zh_CN-huayan-medium.onnx", "Piper 中文音色", True),
    ("models/vad/silero_vad.onnx", "silero-vad", False),
    ("models/wake/hey_jarvis_v0.1.onnx", "唤醒词", False),
    ("models/llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf", "Qwen2.5-7B 模型", True),
    ("models/llama/llama-server.exe", "llama.cpp 运行时", True),
]


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def bad(msg: str) -> None:
    print(f"  [MISS] {msg}")


def main() -> int:
    problems = 0
    print("=== Python ===")
    ver = sys.version.split()[0]
    major, minor = sys.version_info[:2]
    if (major, minor) in {(3, 10), (3, 11), (3, 12)}:
        ok(f"{ver}")
    else:
        bad(f"{ver} —— 需要 3.10 ~ 3.12，3.13/3.14 上部分轮子不可用")
        problems += 1

    print("\n=== 依赖 ===")
    for name, desc in REQUIRED:
        try:
            mod = __import__(name)
            ok(f"{name:16} {getattr(mod, '__version__', ''):10} {desc}")
        except Exception as exc:
            bad(f"{name:16} {desc} —— {exc}")
            problems += 1

    print("\n=== 可选依赖 ===")
    try:
        import pyttsx3  # noqa: F401
        ok("pyttsx3 (SAPI 兜底)")
    except Exception:
        print("  [--]   pyttsx3 未装，TTS 只剩 piper 两条路径")

    try:
        import piper
        from piper import PiperVoice

        sig = str(inspect.signature(PiperVoice.synthesize))
        ok(f"piper API: synthesize{sig[:60]}")
        print(f"         新式流式 API: {hasattr(piper, 'SynthesisConfig')}")
    except Exception as exc:
        bad(f"piper 不可用: {exc}")
        problems += 1

    print("\n=== 音频设备 ===")
    try:
        import sounddevice as sd

        print(f"  默认输入: {sd.query_devices(kind='input')['name']}")
        print(f"  默认输出: {sd.query_devices(kind='output')['name']}")
        ok("PortAudio 正常")
    except Exception as exc:
        bad(f"音频设备异常: {exc}")
        problems += 1

    print("\n=== 模型文件 ===")
    for rel, desc, required in MODEL_FILES:
        p = ROOT / rel
        if p.exists():
            size = (
                sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                if p.is_dir()
                else p.stat().st_size
            )
            ok(f"{desc:20} {size / 1024 / 1024:8.0f} MB  {rel}")
        elif required:
            bad(f"{desc:20} 缺失  {rel}")
            problems += 1
        else:
            print(f"  [--]   {desc:20} 缺失（有降级方案）  {rel}")

    print("\n=== 大模型服务 ===")
    try:
        from openai import OpenAI

        client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local", timeout=5)
        names = ", ".join(m.id for m in client.models.list().data)
        ok(f"llama-server 在线: {names}")
    except Exception as exc:
        bad(f"连不上 127.0.0.1:8080 —— {exc}")
        print("         （先启动 run_server.bat；只做环境检查时可忽略此项）")

    print()
    if problems == 0:
        print("全部通过，可以跑了：run_server.bat 然后 run_assistant.bat")
    else:
        print(f"有 {problems} 项需要处理，见上面的 [MISS]")
        print("模型缺失时运行: python scripts/download_models.py")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
