"""本地语音助手。"""
import os

# 国内网络：huggingface.co 直连不通，必须在导入 huggingface_hub / faster_whisper
# 之前把镜像设好，否则 whisper 首次加载会卡在联网校验上（表现为 502 Bad Gateway）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

__version__ = "0.1.0"
