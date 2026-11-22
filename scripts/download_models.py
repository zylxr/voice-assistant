"""一键下载全部模型与 llama.cpp 运行时。

  python scripts/download_models.py                # 全部下载
  python scripts/download_models.py --only llm     # 只下大模型（4.36GB）
  python scripts/download_models.py --probe        # 只测各下载源速度，不下文件
  python scripts/download_models.py --skip llm     # 跳过 7B 模型

网络说明：
  huggingface.co 在国内直连不通，脚本默认把 HF_ENDPOINT 指向 hf-mirror.com。
  大文件（GGUF）优先走 ModelScope（国内 CDN，不经过 xet 中转），
  会先对多个源各探 1MB 测速，再用最快的那个下载，支持断点续传。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# 必须在 import huggingface_hub / faster_whisper 之前设置，否则镜像不生效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
UA = "voice-assistant/0.1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                                   # 让 PowerShell 里直接跑也能正常显示中文
    from _console import setup as _setup_console

    _setup_console()
except Exception:
    pass

# ---------------------------------------------------------------- 模型来源

SILERO_URLS = [
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx",
]

PIPER_REPO = "rhasspy/piper-voices"
PIPER_FILES = [
    "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
    "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json",
]

GGUF_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
GGUF_SIZE = 4_683_074_240
GGUF_SHA256 = "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423"

# 顺序不重要，脚本会实测速度后排序；留多个是为了某个源挂掉时能顶上来
GGUF_SOURCES = [
    ("ModelScope 官方镜像",
     "https://modelscope.cn/models/bartowski/Qwen2.5-7B-Instruct-GGUF/"
     "resolve/master/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("ModelScope API",
     "https://modelscope.cn/api/v1/models/bartowski/Qwen2.5-7B-Instruct-GGUF/repo"
     "?Revision=master&FilePath=Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("hf-mirror",
     "https://hf-mirror.com/bartowski/Qwen2.5-7B-Instruct-GGUF/"
     "resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("HuggingFace 官方",
     "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/"
     "resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
]

# GitHub 在国内时通时不通，按顺序尝试直连 + 已知的加速镜像
GH_MIRRORS = [
    "",
    "https://ghproxy.net/",
    "https://gh-proxy.com/",
    "https://ghfast.top/",
]


# ---------------------------------------------------------------- 工具函数

def human(n: float) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f}GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f}MB"
    return f"{n / 1024:.0f}KB"


def sha256_of(path: Path, quiet: bool = False) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    t0 = time.time()
    last = 0.0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 << 20)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if not quiet and time.time() - last > 2:
                last = time.time()
                print(f"\r    校验 {done * 100 / total:5.1f}%  "
                      f"{done / max(time.time() - t0, 1e-6) / 1e6:.0f}MB/s",
                      end="", flush=True)
    if not quiet:
        print("\r" + " " * 40 + "\r", end="")
    return h.hexdigest()


def probe(url: str, nbytes: int = 1 << 20, timeout: float = 15.0) -> float:
    """取 1MB 测速，返回 bytes/s；失败返回 0。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Range": f"bytes=0-{nbytes - 1}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            got = 0
            while got < nbytes:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                got += len(chunk)
                if time.time() - t0 > timeout:
                    break
    except Exception:
        return 0.0
    dt = max(time.time() - t0, 1e-3)
    if got < 64 << 10:          # 拿到的数据太少，不认为可用
        return 0.0
    return got / dt


def rank_sources(sources: list[tuple[str, str]], label: str = "") -> list[tuple[str, str]]:
    """对每个源探测速度，按快慢排序。全部失败时返回原顺序。"""
    print(f"  正在测试 {len(sources)} 个下载源的速度{'（' + label + '）' if label else ''} ...")
    scored = []
    for name, url in sources:
        speed = probe(url)
        if speed > 0:
            print(f"    {name:20} {speed / 1e6:6.2f} MB/s")
            scored.append((speed, name, url))
        else:
            print(f"    {name:20} 不可用")
    if not scored:
        print("    所有源测速失败，按默认顺序尝试")
        return sources
    scored.sort(reverse=True)
    return [(name, url) for _, name, url in scored]


def fetch(url: str, dest: Path, total: int | None = None, retries: int = 4) -> bool:
    """断点续传下载到 dest.part，成功后改名。返回是否拿到完整文件。"""
    part = dest.parent / (dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        if total is not None and start > total:
            part.unlink()
            start = 0
        if total is not None and start == total:
            break

        headers = {"User-Agent": UA}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                resume_ok = start > 0 and resp.status == 206
                if start and not resume_ok:
                    print("    服务器不支持续传，从头开始")
                    start = 0
                mode = "ab" if resume_ok else "wb"
                got = 0
                t0 = last = time.time()
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        now = time.time()
                        if now - last > 2:
                            last = now
                            speed = got / max(now - t0, 1e-6)
                            done = start + got
                            if total:
                                eta = (total - done) / max(speed, 1)
                                print(f"\r    {done * 100 / total:5.1f}%  "
                                      f"{human(done)} / {human(total)}  "
                                      f"{speed / 1e6:5.1f}MB/s  剩余 {int(eta // 60)}m{int(eta % 60):02d}s ",
                                      end="", flush=True)
                            else:
                                print(f"\r    {human(done)}  {speed / 1e6:5.1f}MB/s ",
                                      end="", flush=True)
                if total is None or part.stat().st_size >= total:
                    if total:
                        print("\r" + " " * 78 + "\r", end="")
                    break          # 下完了，出循环去改名
                print(f"\n    连接提前结束（{human(part.stat().st_size)}），续传重试 "
                      f"{attempt}/{retries}")
        except Exception as exc:
            print(f"\n    出错: {exc}  （重试 {attempt}/{retries}）")
            time.sleep(min(2 * attempt, 8))

    if not part.exists():
        return False
    size = part.stat().st_size
    if total is not None and size != total:
        print(f"    未下完：{human(size)} / {human(total)}，.part 已保留，下次运行会续传")
        return False
    part.replace(dest)
    return True


def download_model(dest: Path, sources: list[tuple[str, str]],
                   total: int | None = None, sha256: str | None = None,
                   probe_first: bool = True) -> bool:
    if dest.exists() and (total is None or dest.stat().st_size == total):
        print(f"  跳过（已存在 {human(dest.stat().st_size)}）")
        if sha256:
            print("  校验中 ...", end="", flush=True)
            ok = sha256_of(dest, quiet=True) == sha256
            print(" 一致" if ok else " 不一致！建议删除后重下")
            if not ok:
                return False
        return True

    ordered = rank_sources(sources) if probe_first else sources
    for name, url in ordered:
        print(f"  源: {name}")
        if fetch(url, dest, total):
            break
    if not dest.exists():
        return False

    size = dest.stat().st_size
    print(f"  完成 {dest.name} ({human(size)})")
    if total is not None and size != total:
        print(f"  大小不符（期望 {human(total)}），建议删除重下")
        return False
    if sha256:
        print("  校验 SHA256 ...", end="", flush=True)
        if sha256_of(dest, quiet=True) == sha256:
            print(" 通过")
        else:
            print(" 失败！文件可能损坏，删除后重下")
            return False
    return True


# ---------------------------------------------------------------- 各环节

def do_whisper() -> None:
    print("[1/5] faster-whisper small")
    print(f"  使用镜像: {os.environ['HF_ENDPOINT']}")
    try:
        from faster_whisper import WhisperModel

        WhisperModel("small", device="cpu", compute_type="int8",
                     download_root=str(MODELS / "whisper"))
        print("  完成")
    except Exception as exc:
        print(f"  失败: {exc}")
        print("  可手动下载 https://hf-mirror.com/Systran/faster-whisper-small 里的全部文件")


def do_piper() -> None:
    print("[2/5] Piper 中文音色 zh_CN-huayan-medium")
    flat = [MODELS / "piper" / Path(n).name for n in PIPER_FILES]
    if all(p.exists() and p.stat().st_size > 1024 for p in flat):
        print("  跳过（已存在）")
        return
    try:
        from huggingface_hub import hf_hub_download

        for name in PIPER_FILES:
            p = Path(hf_hub_download(repo_id=PIPER_REPO, filename=name,
                                     local_dir=str(MODELS / "piper")))
            # hf 会保留仓库内的子目录结构，config.yaml 用的是扁平路径，这里补一份
            dest = MODELS / "piper" / Path(name).name
            if p.resolve() != dest.resolve():
                dest.write_bytes(p.read_bytes())
            print(f"  完成 {dest.name} ({human(dest.stat().st_size)})")
    except Exception as exc:
        print(f"  失败: {exc}")
        print("  可手动从 https://hf-mirror.com/rhasspy/piper-voices 下载")


def do_vad() -> None:
    print("[3/5] silero-vad onnx")
    dest = MODELS / "vad" / "silero_vad.onnx"
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"  跳过（已存在 {human(dest.stat().st_size)}）")
        return
    for url in SILERO_URLS:
        if fetch(url, dest):
            print(f"  完成 {dest.name} ({human(dest.stat().st_size)})")
            return
    if dest.exists():
        print(f"  使用已有文件 {human(dest.stat().st_size)}")
        return
    print("  全部源失败，程序会自动降级为能量 VAD，不影响运行")


def do_wake() -> None:
    print("[4/5] openWakeWord 唤醒词 hey_jarvis")
    dest = MODELS / "wake" / "hey_jarvis_v0.1.onnx"
    if dest.exists():
        print("  跳过（已存在）")
        return
    try:
        import openwakeword
        from openwakeword.utils import download_models

        download_models(model_names=["hey_jarvis_v0.1"])
        src = Path(openwakeword.__file__).parent / "resources" / "models" / "hey_jarvis_v0.1.onnx"
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            print(f"  完成 {dest.name}")
        else:
            print(f"  已下载到 openwakeword 包内 ({src})，程序会按名称自动找到")
    except Exception as exc:
        print(f"  失败: {exc}\n  可改用 --no-wake 回车触发")


def do_llm() -> None:
    print(f"[5/5] Qwen2.5-7B-Instruct Q4_K_M ({human(GGUF_SIZE)})")
    print("  大文件，支持断点续传；中途 Ctrl+C 后重跑会接着下")
    ok = download_model(MODELS / "llm" / GGUF_FILE, GGUF_SOURCES,
                        total=GGUF_SIZE, sha256=GGUF_SHA256)
    if not ok:
        print("  下载未完成。也可以手动下载后放到 models/llm/ 下：")
        print("    https://modelscope.cn/models/bartowski/Qwen2.5-7B-Instruct-GGUF")
        print("    文件名必须是 Qwen2.5-7B-Instruct-Q4_K_M.gguf")


def _pick_llama_asset() -> tuple[str, int]:
    """从 GitHub releases 里找一个带 win-cpu-x64 压缩包的版本。"""
    api = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
    req = urllib.request.Request(api, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        releases = json.load(resp)
    for rel in releases:
        for a in rel.get("assets", []):
            name = a["name"].lower()
            if name.endswith(".zip") and "win" in name and "cpu" in name and "x64" in name:
                return a["browser_download_url"], a["size"]
    raise RuntimeError("最近 10 个 release 里都没有 win-cpu-x64 资产")


def do_llama_cpp() -> None:
    print("[+] llama.cpp Windows CPU 版")
    dest = MODELS / "llama"
    if (dest / "llama-server.exe").exists():
        print("  跳过（已存在）")
        return
    try:
        url, size = _pick_llama_asset()
        print(f"  {url.rsplit('/', 1)[-1]} ({human(size)})")
    except Exception as exc:
        print(f"  查询版本失败: {exc}")
        print("  请手动下载 https://github.com/ggml-org/llama.cpp/releases 里的")
        print("  llama-*-bin-win-cpu-x64.zip 并解压到 models/llama/")
        return

    zip_path = MODELS / "llama" / "llama-bin.zip"
    sources = [(m or "GitHub 直连", (m + url) if m else url) for m in GH_MIRRORS]
    if not download_model(zip_path, sources, total=size):
        print("  请手动下载后用压缩软件解压到 models/llama/")
        return

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink()

    # 压缩包内可能有多层目录，把 exe 和 dll 统一挪到 models/llama/ 根下
    if not (dest / "llama-server.exe").exists():
        for exe in dest.rglob("llama-server.exe"):
            for f in exe.parent.iterdir():
                if f.is_file():
                    target = dest / f.name
                    if not target.exists():
                        f.replace(target)
            break
    if (dest / "llama-server.exe").exists():
        print(f"  完成 -> {dest / 'llama-server.exe'}")
    else:
        print(f"  解压完成但没找到 llama-server.exe，请检查 {dest}")


def do_probe() -> None:
    print("== 大模型下载源测速（各取 1MB）==")
    for name, url in GGUF_SOURCES:
        t0 = time.time()
        speed = probe(url, nbytes=4 << 20, timeout=20)
        if speed:
            left = GGUF_SIZE / speed / 60
            print(f"  {name:18} {speed / 1e6:6.2f} MB/s  预计 {left:5.1f} 分钟 "
                  f"({time.time() - t0:.1f}s)")
        else:
            print(f"  {name:18} 不可用")
    print("\n== llama.cpp 下载源测速 ==")
    try:
        url, size = _pick_llama_asset()
        for m in GH_MIRRORS:
            label = m or "GitHub 直连"
            speed = probe(m + url if m else url, nbytes=1 << 20, timeout=15)
            if speed:
                print(f"  {label:18} {speed / 1e6:6.2f} MB/s  ({human(size)} 文件)")
            else:
                print(f"  {label:18} 不可用")
    except Exception as exc:
        print(f"  查询失败: {exc}")


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只做这些，逗号分隔: whisper,tts,vad,wake,llm,llama")
    ap.add_argument("--skip", default="", help="跳过这些，逗号分隔")
    ap.add_argument("--probe", action="store_true", help="只测下载源速度，不下载")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    if args.probe:
        do_probe()
        return 0

    tasks = {
        "whisper": do_whisper,
        "tts": do_piper,
        "vad": do_vad,
        "wake": do_wake,
        "llm": do_llm,
        "llama": do_llama_cpp,
    }
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    for name, fn in tasks.items():
        if only and name not in only:
            continue
        if name in skip:
            print(f"[跳过] {name}")
            continue
        fn()
    print("\n全部结束。检查 models/ 目录后再启动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
