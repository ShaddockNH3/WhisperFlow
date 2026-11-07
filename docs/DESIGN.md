# 设计说明

技术栈没动：FastAPI + Vue 3 + torch / faster-whisper。动的是"模型从哪儿来"和"魔法值放在哪"。

## 原来什么样

| 位置 | 问题 |
| --- | --- |
| `AIPipeline.__init__` | 内部直接 new VAD、降噪模型、Whisper、LLM，四个重依赖串在一起，想测管道逻辑必须先装几个 G 的库 |
| 默认参数 | 模型 id、权重路径写死在构造函数默认值里，改模型要改代码 |
| `ws_api.py` 模块常量 | `INTERIM_INTERVAL_S`、`MIN_INTERIM_SAMPLES`、线程池大小写死，跟 `ai/` 里的配置两套风格 |
| `pipeline` 的两个去重逻辑 | 相邻句相似度 0.85、上下文相似度 0.7、历史长度 10，全藏在方法体里，而且和 difflib 调用缠在一起 |
| `use_llm` | WS 里默认 `True`，服务端就算把 `use_llm=False` 传进管道，前端一连上又变回开 |

## 现在怎么分

```
ai/config.py     环境变量 → PipelineSettings / StreamSettings（不可变 dataclass）
ai/dedup.py      RepeatFilter（相邻句）+ TextHistory（上下文窗口），纯 difflib
ai/pipeline.py   Stages 注入；build_default_stages 才 import torch
backend/ws_api.py 只从 ai.config 取节奏参数
```

`AIPipeline(stages=..., settings=...)` 两个都可以不传：不传就按环境变量构造真实阶段；传了就用传进来的。生产路径和原来完全一致。

## 去重为什么单独拿出来

两段逻辑原本是：

```python
similarity = difflib.SequenceMatcher(None, raw, last).ratio()
if similarity > 0.85: return None
```

和

```python
for prev in history[-3:]:
    if difflib.SequenceMatcher(None, corrected, prev).ratio() > 0.7: return
```

看着像、语义不同：前者比"上一句"（逐句单调），后者比"最近三条"（窗口）。抽成 `RepeatFilter` 和 `TextHistory` 之后，阈值、窗口大小都能从配置来，各自的边界（空文本、重置、超长裁剪）也都有测试盯着。

## 配置为什么做成 frozen dataclass

环境变量一读就该定下来。用不可变对象传递，避免运行中被某处改掉一半（原来 `use_llm` 就是被前端控制消息直接改的局部变量）。前端仍然可以按连接覆盖，但服务端关掉纠错时 `pipeline.use_llm` 是 False，连接初始化就继承它。

## 前端

`App.vue` 原来 578 行：模板 96 行、脚本 312 行，WebSocket 消息处理、音频采集、断线重连、界面全在一个 `<script setup>` 里。

最难验证的是那段 `ws.onmessage`：一百多行 if/else 直接改 reactive 数组，状态有四种（partial / whisper_done / sentence_open / final），还夹着"过期的 LLM 响应要丢掉"这种时序判断。改坏一个分支，界面上只是少一行字，肉眼很难发现。

拆完：

| 文件 | 职责 |
| --- | --- |
| `services/transcript.js` | 状态机：`createTranscriptState` + `applyMessage`，纯函数 |
| `services/audio-capture.js` | getUserMedia/getDisplayMedia、16k 重采样、Int16 分帧 |
| `composables/useRecognition.js` | 建连、重连、录音开关、把消息喂给状态机 |
| `components/ControlPanel.vue` | 音源、AI 纠错开关、开始/停止、录音提示 |
| `components/TranscriptPanel.vue` | 字幕列表与自动滚动 |
| `App.vue` | 71 行的版面 |

状态机抽出来之后就能测了：13 个用例覆盖"LLM 模式下新片段并进同一句、不新起一行"、"过期 llm_done 丢弃"、"没句号时标记 sentence_open"、"关录音时收尾"等时序分支。

验证方式：`node --test tests/*.test.js`；另外用自研的假后端（静态服务 + 手写 WS 握手，8000 端口按脚本推消息）在浏览器里跑了一遍完整链路，两条字幕的状态和边框颜色都对。

顺带补了 `vite.config.js` 的 `@` 别名（另外两个前端都有，这个没有）。

## 代价

- 多了一层 `Stages` 抽象，读代码时要在 `pipeline.py` 和 `build_default_stages` 之间来回看。
- 默认阶段仍然是"一连接一实例"，模型加载在第一次连接时发生，冷启动慢的问题没解决。
- 没装 torch 的机器只能跑 `tests/` 下的 19 个纯逻辑用例，管道本身的端到端验证还得在有依赖的环境做。
