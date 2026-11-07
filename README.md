# WhisperFlow

浏览器端实时语音转写：麦克风切片 → WebSocket 送后端 → VAD 断句 → CRNN 降噪 → faster-whisper 识别 → LLM 流式纠错 → 回显。FastAPI + Vue 3。

## 快速开始

```bash
uv venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -r backend/requirements.txt

cp backend/.env.example backend/.env      # 填 LLM_API_KEY（不填就关掉纠错）

cd frontend && npm install && npm run build && cd ..

python backend/main.py                    # 默认 http://127.0.0.1:8000
```

跑单测（只覆盖纯逻辑，不需要装 torch）：

```bash
python -m unittest discover -s tests -t .
```

## 管道

| 阶段 | 做什么 | 实现 |
| --- | --- | --- |
| 前端 | MediaRecorder 按 500ms 切片，Int16Array 二进制上送 | `frontend/src` |
| 接收 | 解码、重采样到 16k 单声道、丢进单线程执行器（VAD 有状态，不能并发） | `backend/ws_api.py` |
| Stage 1 | CRNN 预测频域掩膜（IRM）做降噪，ISTFT 重建 | `ai/denoise` |
| Stage 2 | VAD 判定语音段边界，静音超时即切片 | `ai/vad` |
| Stage 3 | faster-whisper 转录，OpenCC 繁转简 | `ai/whisper` |
| 后处理 | LLM 流式纠错 + 复读拦截 | `ai/llm`、`ai/dedup.py` |

管道把四个阶段收成 `Stages`（vad / denoise / transcribe / correct_stream），默认实现放在 `build_default_stages`，只有真正要跑模型时才 import torch。

## 配置

环境变量都在 `backend/.env`，代码里不留魔法值。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ZHIPUAI_API_KEY` | 无 | LLM 纠错用的接口密钥 |
| `WHISPERFLOW_WHISPER_MODEL` | `openai/whisper-turbo` | 转录模型，内部会映射到 `large-v3-turbo` |
| `WHISPERFLOW_LLM_MODEL` | `glm-4-flash-250414` | 纠错模型 |
| `WHISPERFLOW_USE_LLM` | `true` | 关掉就只做转录 |
| `WHISPERFLOW_DENOISE_WEIGHTS` | `ai/denoise/weights/denoise_crnn.pt` | 降噪权重，相对路径按项目根解析 |
| `WHISPERFLOW_VAD_THRESHOLD` | `0.15` | VAD 灵敏度 |
| `WHISPERFLOW_REPEAT_THRESHOLD` | `0.85` | 相邻句相似度超过就判复读丢掉 |
| `WHISPERFLOW_HISTORY_THRESHOLD` | `0.7` | 纠错结果和上下文太像就不入库 |
| `WHISPERFLOW_INTERIM_INTERVAL_S` | `1.5` | 临时预览的间隔 |
| `WHISPERFLOW_MIN_INTERIM_SAMPLES` | `20000` | 预览前的最少缓冲采样数（约 1.25 秒） |

## 项目结构

```
ai/
├── config.py     环境变量 → 配置对象
├── dedup.py      复读拦截与上下文窗口（纯 difflib，可单测）
├── pipeline.py   Stages 注入 + 会话级管道
├── vad/          Silero VAD 与段缓冲
├── denoise/      CRNN 掩膜降噪、训练与推理脚本
├── whisper/      faster-whisper 封装
└── llm/          GLM 流式纠错
backend/
├── main.py       FastAPI 入口、CORS、静态站点
└── ws_api.py     /api/ws/recognize，二进制音频与控制消息
frontend/         Vue 3 界面
tests/            纯逻辑单测（19 个）
```

## 已知问题

- 单连接串行推理：`_pipeline_executor` 只有一个线程，多人同时说话会排队。要多路并发得给每连接独立 VAD 实例并扩线程池。
- 降噪权重随仓库走（`denoise_crnn.pt`），仓库因此偏大；换机器跑要先确认路径。
- `ai/test_pipeline.py`、`ai/vad/test_vad.py` 需要 torch 和测试音频，跟 `tests/` 下的纯逻辑单测分开跑。
- 前端构建产物要手动 `npm run build`，后端只负责托管 `dist`。
- LLM 纠错依赖外部接口，没网或没 key 时只能拿到原始转录。

## 更多

架构取舍与重构说明见 [docs/DESIGN.md](docs/DESIGN.md)。

## 致谢

[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[Silero VAD](https://github.com/snakers4/silero-vad)、[FastAPI](https://fastapi.tiangolo.com/)、[OpenCC](https://github.com/BYVoid/OpenCC)。

## License

MIT，见 [LICENSE](LICENSE)。
