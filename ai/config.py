"""配置：模型、阈值、开关都从环境变量读，代码里不留魔法值。

原来 `AIPipeline()` 的默认参数写死了模型 id 和权重路径，WS 接口再把
`INTERIM_INTERVAL_S`、`MIN_INTERIM_SAMPLES` 写成模块常量，改一处要翻三个文件。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DENOISE_WEIGHTS = "ai/denoise/weights/denoise_crnn.pt"
DEFAULT_WHISPER_MODEL = "openai/whisper-turbo"
DEFAULT_LLM_MODEL = "glm-4-flash-250414"

SAMPLE_RATE = 16_000


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class PipelineSettings:
    """管道的一份配置快照。不可变，方便在测试里造。"""

    denoise_weights: Path
    whisper_model_id: str = DEFAULT_WHISPER_MODEL
    llm_model_id: str = DEFAULT_LLM_MODEL
    use_llm: bool = True
    vad_threshold: float = 0.15
    sample_rate: int = SAMPLE_RATE
    repeat_threshold: float = 0.85
    history_threshold: float = 0.7
    history_size: int = 10

    def resolved_weights(self) -> Path:
        """相对路径按项目根解析，这样从 backend/ 启动也能找到权重。"""
        if self.denoise_weights.is_absolute():
            return self.denoise_weights
        return PROJECT_ROOT / self.denoise_weights

    def weights_available(self) -> bool:
        return self.resolved_weights().exists()


@dataclass(frozen=True)
class StreamSettings:
    """WebSocket 侧的节奏控制。"""

    interim_interval_s: float = 1.5
    min_interim_samples: int = 20_000
    inference_workers: int = 2
    pipeline_workers: int = 1
    extra: dict = field(default_factory=dict)


def load_pipeline_settings() -> PipelineSettings:
    """从环境变量读一份配置，缺省值和原来代码里的默认值一致。"""
    weights = os.environ.get("WHISPERFLOW_DENOISE_WEIGHTS", DEFAULT_DENOISE_WEIGHTS)
    return PipelineSettings(
        denoise_weights=Path(weights),
        whisper_model_id=os.environ.get("WHISPERFLOW_WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        llm_model_id=os.environ.get("WHISPERFLOW_LLM_MODEL", DEFAULT_LLM_MODEL),
        use_llm=_env_bool("WHISPERFLOW_USE_LLM", True),
        vad_threshold=_env_float("WHISPERFLOW_VAD_THRESHOLD", 0.15),
        repeat_threshold=_env_float("WHISPERFLOW_REPEAT_THRESHOLD", 0.85),
        history_threshold=_env_float("WHISPERFLOW_HISTORY_THRESHOLD", 0.7),
    )


def load_stream_settings() -> StreamSettings:
    return StreamSettings(
        interim_interval_s=_env_float("WHISPERFLOW_INTERIM_INTERVAL_S", 1.5),
        min_interim_samples=int(_env_float("WHISPERFLOW_MIN_INTERIM_SAMPLES", 20_000)),
    )
