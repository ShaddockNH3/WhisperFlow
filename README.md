# WhisperFlow 高性能流式语音识别系统

WhisperFlow 是一个基于 FastAPI 和 Vue 3 的实时、流式语音识别（ASR）系统。该项目内置了一套完整的 AI 处理回路，包含硬件级 VAD 智能断句、CRNN 音频降噪、faster-whisper 识别引擎以及基于大语言模型（LLM）的流式纠错机制，旨在提供低延迟、高准确率的语音交互体验。

## 🧠 系统架构与设计思路

WhisperFlow 是一个前后端分离的伪流式 ASR 平台，目标在嘈杂环境下实现低延迟、高鲁棒性的实时语音转写与上下文纠错。

*   前端（客户端）：基于 Vue 3 + Vite；使用 Web Audio API 或 MediaRecorder 采集麦克风音频，按固定时长（例如 500ms）截取为 Int16Array，并通过 WebSocket 以二进制流持续发送；支持展示实时波形与逐字、逐句回显。
*   后端（服务端）：基于 FastAPI + Uvicorn（asyncio）；提供 WebSocket 端点接收音频流，使用相关库进行内存解码与重采样，确保音频满足 16kHz、单声道、16-bit PCM 的标准格式，通过异步队列缓冲并调度后续处理，具备出色的高并发连接支持能力。
*   数据流与调度：前端切片 -> 后端接收并重采样 -> 降噪 -> VAD 判定段边界 -> Whisper 推理 -> LLM 纠错（可选） -> 结果通过 WebSocket 回传前端。

### 流式语音识别管道 (AI Pipeline)

传统端到端识别模型通常需要等待接收到完整的音频段落后方可进行处理，存在延迟较高且易丢失上下文的问题。WhisperFlow 采用了轻巧的伪流式架构，三级级联模块设计具体如下：

1.  Stage 1 — 轻量级降噪前端：基于 PyTorch 实现的轻量 DFN/CRNN，基于 STFT 预测频域掩膜（IRM），对噪声进行抑制并通过 ISTFT 重建增强波形，提高噪声鲁棒性。
2.  Stage 2 — VAD + 伪流式分段：集成 Silero VAD（或类似模块）监测语音活动；基于 VAD 动态缓冲（例如静音持续超 400ms 触发切片）进行分段，避免固定窗口切片带来的延迟和上下文截断。
3.  Stage 3 — Whisper 转录与 LLM 后处理：将增强后的音频段送入经过优化的 faster-whisper（默认 turbo 模型）进行推理；随后将初稿送入外部大语言模型（如 GLM-4-flash）执行上下文纠错、同音字消歧与标点恢复。
4.  模型“幻觉”抑制引擎：本项目引入了基于序列相似度算法 (`difflib`) 的历史记录比对模组。当系统的输出文本与近期识别结果的相似度过高（表明模型发生重复输出错觉）时，系统判定其为无效的重复结果并主动拦截。这种机制极大降低了 Whisper 模型在特定场景下的误识率。

## 🎯 快速开始

1. 环境配置
   ```bash
   git clone https://github.com/ShaddockNH3/WhisperFlow.git
   cd WhisperFlow
   
   uv venv
   
   # Windows 执行: .venv\Scripts\activate
   # Linux/Mac 执行: source .venv/bin/activate
   
   uv pip install -r backend/requirements.txt
   ```

2. 填写配置文件
   在 `backend` 目录下创建 `.env` 环境变量配置文件。若需要启用 LLM 语义流式纠正引擎（建议开启），请填入大参数接口调用鉴权：
   ```env
   # ZhipuAI GLM 接口示例配置
   LLM_API_KEY=您的_API_KEY_填写于此
   ```

3. 静态资源编译
   ```bash
   cd frontend
   npm install
   npm run build
   ```

4. 测试验证运行
   ```bash
   # 在项目根目录下启动入口进程
   python backend/main.py
   ```
   应用服务顺利启动后，系统将自动调用默认浏览器跳转至 `http://127.0.0.1:8000`。
