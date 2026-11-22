# 本地语音助手（纯 CPU · llama.cpp · Qwen2.5-7B Q4）

面向 **32G 内存 + MX150 2GB** 这类机器：显卡不参与推理，全部跑在 CPU 上，
靠「流式分句 + 边生成边播报」把体感延迟压到可接受范围。

## 一、为什么是这套方案

| 决策 | 原因 |
|---|---|
| 推理引擎用 **llama.cpp** | vLLM 硬性要求 CUDA CC≥7.0，SGLang 依赖 FlashAttention，MX150 是 Pascal CC 6.1 + 2GB，两者都装不上 |
| **不用 GPU** | 2GB 显存塞不下 7B Q4（4.36GB），能放的 8~12 层带来不到 5% 提升，还容易触发驱动超时 |
| LLM 用 **llama-server 常驻** | OpenAI 兼容接口，将来换机器/换 vLLM 只改 `llm.base_url`，上层代码不动 |
| ASR 用 **whisper small** | medium 在 CPU 上 RTF≈1.0~1.5，会明显迟钝；small 实测 RTF 0.72 |
| TTS 用 **Piper** | CPU 实时合成，首音约 200ms；CosyVoice/GPT-SoVITS 需要 4GB+ 显存，这台跑不了 |
| 线程分配 **8/12** | LLM 吃 8 线程，留 4 个给 ASR、TTS、音频回调，避免互相抢 CPU 导致爆音 |

**实测数据**（i7-9850H 6C12T / 32G，全部本地离线）：

| 指标 | 结果 |
|---|---|
| 7B Q4_K_M 生成速度 | **10.3 字/秒**（首字 0.93s），高于语音流畅线 5 字/秒 |
| TTS 合成 | Piper 约 **19 倍实时**（4.1s 音频 0.21s 合成完） |
| VAD | silero-vad 每帧 **0.18ms** |
| ASR | 2.82s 音频识别 2.04s（RTF 0.72，4 线程最优；2/6/8 线程都更慢） |
| 文本模式首响 | 文本→首音 **1.1~1.5s** |

**延迟构成**：VAD 断句 0.6s + ASR ≈2.2s（3 秒语音，RTF 0.72）+ LLM 首字 0.9s + TTS 首音 0.2s ≈ **3.9 秒首响**，
之后靠流式播报保持连续。ASR 这一段是当前最大的瓶颈（见第九节优化方向）。

## 二、目录结构

```
voice-assistant/
├── config.yaml              全部参数集中在这里
├── requirements.txt
├── setup.bat                建环境 + 装依赖（setup.bat models 顺带下模型）
├── run_server.bat           启动 llama-server（大模型）
├── run_assistant.bat        启动语音助手
├── scripts/
│   ├── download_models.py   一键下载模型与 llama.cpp（多源竞速 + 断点续传 + 校验）
│   ├── check_env.py         环境自检：依赖/设备/模型/服务
│   ├── smoke_test.py        离线自测：TTS + VAD + 唤醒词，不需要大模型
│   ├── e2e_test.py          端到端自测：对话链路 + 合成音频回灌 ASR
│   └── bench_llm.py         实测生成速度，判断参数是否合适
├── src/
│   ├── config.py            配置加载
│   ├── audio_io.py          采集 / 播放 / 打断 / 提示音
│   ├── vad.py               silero-vad + 断句状态机（带能量兜底）
│   ├── wakeword.py          openWakeWord 唤醒词（80ms 攒块）
│   ├── asr.py               faster-whisper 封装
│   ├── llm.py               流式对话 + 历史管理
│   ├── textproc.py          流式分句（延迟的关键）
│   ├── tts.py               Piper（三级降级）
│   ├── pipeline.py          主流程编排
│   └── main.py              入口
└── models/                  下载后自动生成
```

## 三、安装

### 1. 环境

需要 Python **3.10 ~ 3.12**。这台机器默认的 `python` 是 3.14，**不要用它**——
ctranslate2 / onnxruntime / piper 这组依赖在 3.14 上还没有稳定轮子。

⚠️ 这台机器的 `py -3.12` 是**坏的**：注册表指向 `D:\Programs\Python\Python312\python.exe`，
而该目录已不存在，直接调用会报 `Unable to create process using ...`。
所以 `setup.bat` 不写死版本，而是按顺序自动探测：
`uv python find 3.12` → `py -3.12` → `uv python find 3.11` → `py -3.11`。
实际会用 uv 管理的 3.12.13。

一条命令搞定：

```powershell
cd D:\dev\workbuddy\voice-assistant
.\setup.bat            # 建 venv + 装依赖
.\setup.bat models     # 顺便下载模型（约 4.9 GB）
```

> ⚠️ **绝对不要往 `.bat` 文件里写中文**。cmd.exe 按 OEM 代码页（GBK）逐字节解析批处理，
> UTF-8 中文字节会连带吃掉后面的换行和命令名，于是出现满屏
> `'local' 不是内部或外部命令` 这类错误。现在三个 `.bat` 都是**纯 ASCII + CRLF**，
> 中文提示全部由 Python 输出。要改脚本时请保持这个约束。

手动等价操作（不想用脚本时）：

```powershell
$py = uv python find 3.12
& $py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> PowerShell 下不建议 `Activate.ps1`（默认执行策略会拦），直接用
> `.\.venv\Scripts\python.exe xxx` 调会更省事。

确认：

```powershell
.\.venv\Scripts\python.exe --version    # 必须显示 3.12.x
```

`piper-tts` 现在有 Windows 轮子（`cp39-abi3-win_amd64`），能正常装上。
万一你的网络拉不到，把它从 requirements.txt 删掉即可，程序会自动降级到
`piper.exe` 或 Windows SAPI。

### 2. 下载模型

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

**国内网络不用手动设环境变量**。脚本里已经固定：

- `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 直连不通，whisper / piper 走这个）
- `HF_HUB_DISABLE_XET=1`（避免被重定向到 xet 存储，那个域名国内更不通）
- 大模型 GGUF 优先走 **ModelScope**（国内 CDN，不过 xet 中转）

会依次准备：

| 项目 | 体积 | 说明 |
|---|---|---|
| faster-whisper small | ~480MB | 识别 |
| Piper zh_CN-huayan-medium | ~60MB | 合成 |
| silero_vad.onnx | ~2MB | 断句 |
| hey_jarvis_v0.1.onnx | ~2MB | 唤醒词 |
| Qwen2.5-7B-Instruct-Q4_K_M | 4.36GB | 大模型 |
| llama.cpp win-cpu-x64 | ~24MB | 推理运行时 |

大模型那几个源各有各的速度，先测再下：

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --probe    # 只测速，不下文件
.\.venv\Scripts\python.exe scripts\download_models.py --only llm # 只下大模型
```

下载器行为：

- **多源竞速**：ModelScope 主站 / ModelScope API / hf-mirror / HF 官方，各探 1MB 后按实测速度排序
- **断点续传**：写到 `xxx.gguf.part`，中断后重跑会从断点接着下，换源也能续（内容一致）
- **SHA256 校验**：下完自动比对 `65b8fcd9...1423`，防止半截文件被当成功
- **进度显示**：百分比 / 已下载 / 实时速度 / 剩余时间

> llama.cpp 解压后如果多了一层目录，脚本会自动把 `llama-server.exe` 和同目录的
> `*.dll` 挪到 `models/llama/` 下，`run_server.bat` 才能找到。

## 四、启动

**窗口 1** —— 常驻大模型服务（首次加载约 10~30 秒）：

```bash
run_server.bat
```

看到 `server is listening on http://127.0.0.1:8080` 就成了。

**窗口 2** —— 语音助手：

```bash
run_assistant.bat
```

说 **“Hey Jarvis”** → 听到两声提示音 → 说你的问题 → 等它回答。

## 五、按顺序验证（强烈建议，别一步跳到全链路）

出问题时这样能立刻定位是哪一环坏了。

```powershell
# ① 环境自检：Python 版本、依赖、音频设备、模型是否齐、服务通不通
.\.venv\Scripts\python.exe scripts\check_env.py

# ② 离线自测：不需要大模型，验证 TTS 合成 + VAD 断句 + 唤醒词推理
.\.venv\Scripts\python.exe scripts\smoke_test.py
#    产物 out/smoke_tts.wav 可以直接双击听音色

# ③ 先确认生成速度够不够（需要 run_server.bat 已启动）
.\.venv\Scripts\python.exe scripts\bench_llm.py

# ④ 端到端：文本→LLM→TTS→扬声器，再把合成音频回灌 ASR 验证识别
.\.venv\Scripts\python.exe scripts\e2e_test.py

# ⑤ 只测 LLM + TTS，不开麦克风，自己打字
.\.venv\Scripts\python.exe -m src.main --text

# ⑥ 加麦克风但不加载唤醒词（回车开始说）
.\.venv\Scripts\python.exe -m src.main --no-wake

# ⑦ 完整链路（唤醒词）
.\.venv\Scripts\python.exe -m src.main
```

> 直接调 `.venv\Scripts\python.exe` 比先 `Activate.ps1` 省事，
> 也绕开了 PowerShell 默认执行策略的拦截。

其他常用命令：

```bash
python -m src.main --list-devices   # 查看麦克风/扬声器编号
python -m src.main --config my.yaml # 用别的配置
```

如果默认设备不对（比如笔记本默认走了耳机孔），把 `config.yaml` 里
`audio.input_device` / `audio.output_device` 改成上一步查到的编号。

## 六、参数调优

### 线程数（最影响速度）

`run_server.bat` 里的 `set T=8`：

- 6 核 12 线程 → `T=8`（超线程对推理帮助有限，物理核心数 +2 是甜点）
- 8 核 16 线程 → `T=10`
- 改完必须重启服务窗口

同时把 `config.yaml` 的 `asr.threads` 保持在 4 以内，两边总和别超过逻辑核心数。

### 如果 bench 结果不到 5 字/秒

中文正常语速约 4~5 字/秒，低于这个数播报会一顿一顿。按顺序试：

1. `T` 调到物理核心数
2. 把 `-c 4096` 降到 `-c 2048`（上下文越长预热越慢）
3. 换 **Qwen3-4B-Instruct-2507**（Q4_K_M 2.33GB，速度大概翻倍）。
   国内直连 HF 不通，用 ModelScope 的单文件地址（`curl.exe` 是 Win10+ 自带的）：
   ```powershell
   curl.exe -L -o models\llm\Qwen3-4B-Instruct-2507-Q4_K_M.gguf `
     "https://modelscope.cn/models/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/master/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
   ```
   然后改 `run_server.bat` 的 `MODEL` 指向它。

### 想试试显卡帮点忙

2GB 塞不下 7B，但可以塞一部分层：

```bat
rem 把 -ngl 0 改成 -ngl 10，观察 bench 是否有提升
```

期望值是**没有明显提升**（PCIe 搬运 + Windows WDDM 开销抵消了算力收益）。
如果出现显存不足崩溃，改回 `-ngl 0`，不要在这上面花时间。

### 唤醒词

默认 `hey_jarvis_v0.1`，阈值 0.55。误唤醒多就调到 0.65，反应不灵就降到 0.45。
想换 `alexa` / `hey_mycroft` 改 `config.yaml` 的 `wake.name`。
**中文唤醒词需要自己训练**（openWakeWord 提供训练脚本），成本不低，先不折腾。

## 七、已知限制

- **不能边说边打断**（`playback.barge_in: false`）：笔记本外放时麦克风会收到自己的声音，
  没有回声消除就会误判。戴耳机时可以在 config 里打开。
- **唤醒词是英文**：预训练模型只有英文，说“小爱同学”唤不醒。
- **回答长度受限**：`max_tokens: 300` + 系统提示要求 1~3 句，因为 7B 在 CPU 上生成长文本会很久。
- **单轮对话，不支持流式 ASR**：说完一句才识别，不支持边说边出字。要做增量识别得换
  FunASR 的流式模型，是下一步的事。

## 八、常见问题

| 现象 | 原因与处理 |
|---|---|
| 启动时报 `502 Bad Gateway` | 99% 是 whisper 模型不在本地，运行时去 huggingface.co 校验被拦（代理返回 502）。跑一次 `scripts\download_models.py --only whisper`，之后 `asr.py` 会直接加载本地快照（日志显示 `[本地]`），不再联网 |
| 下载模型时 `WinError 10060` / `HEAD https://huggingface.co` 超时 | 直连不通。`download_models.py` 已内置镜像（hf-mirror + ModelScope + 多源竞速），用脚本下，不要手动 `hf_hub_download` |
| `连不上大模型服务` | `run_server.bat` 没启动，或端口 8080 被占（改 `-port` 与 `llm.base_url`） |
| `PortAudio` 报错 / 没声音 | 设备编号选错，用 `--list-devices` 确认；或被其他程序独占 |
| ASR 一直输出空 | 麦克风增益太低，Win 设置里把输入音量调到 80% 以上；也可能是 `asr.py` 的噪声过滤把短句滤掉了，看日志里的原始识别结果 |
| 播报一顿一顿 | 生成速度不够，见“参数调优”；或把 `split.max_chars` 调大到 40 |
| 唤醒不灵敏 | `wake.threshold` 降到 0.45；确认 `models/wake/` 里模型存在 |
| 回答念出星号/井号 | `textproc.py` 的 clean 已过滤，若仍出现说明模型输出了特殊符号，可扩充 `JUNK` 正则 |
| CPU 100% 且卡顿 | LLM 线程数超过物理核心，调小 `T` |
| PowerShell 里中文变乱码 | 控制台代码页不是 UTF-8。先执行 `chcp 65001`，或用 `run_assistant.bat` 启动（脚本内部已 `chcp 65001`） |
| `.bat` 报 `'local' 不是内部或外部命令` | 文件被存成了 UTF-8 且含中文。`.bat` 必须纯 ASCII + CRLF，中文交给 Python 输出 |
| `py -3.12` 报 `Unable to create process` | 这台机器的 py 启动器注册表记录已失效，用 `uv python find 3.12` 拿真实路径，`setup.bat` 会自动处理 |

## 九、后续可升级

1. **流式 ASR**：换 FunASR 的 Paraformer-streaming，说一半就开始出字，首响能再降 0.5s
2. **回声消除**：接 `speexdsp` 做 AEC，就能开打断
3. **Function Calling**：给 `llm.py` 加工具注册（查天气、开程序、查行情），
   从“聊天”变成“助手”
4. **换硬件**：有 12GB 以上显卡时，把 `llm.base_url` 指向 vLLM，其余代码不用改
