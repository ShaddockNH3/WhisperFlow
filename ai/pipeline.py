"""伪流式 ASR 管道：VAD 断句 → 降噪 → Whisper 转录 → （可选）LLM 纠错。

各段模型做成可注入的 `Stages`：默认实现仍然在 `build_default_stages` 里按需 import
torch / faster-whisper，测试可以塞假实现，不必装几个 G 的依赖。
"""

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

import numpy as np

from ai.config import PipelineSettings, load_pipeline_settings
from ai.dedup import RepeatFilter, TextHistory


class VadStage(Protocol):
    """VAD 只要这三件事。"""

    buffer: list

    def process_chunk(self, audio: np.ndarray) -> tuple[np.ndarray, str] | None: ...

    def force_emit(self) -> tuple[np.ndarray, str] | None: ...


Transcriber = Callable[[np.ndarray], str]
Denoiser = Callable[[np.ndarray], np.ndarray]
Corrector = Callable[[str, list[str]], Iterator[str]]


@dataclass
class Stages:
    """管道四个阶段。denoise / transcribe 是函数，换实现不用改管道。"""

    vad: VadStage
    denoise: Denoiser
    transcribe: Transcriber
    correct_stream: Corrector | None = None


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """峰值归一化，避免削顶那一段把 Whisper 带跑。"""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        return audio / peak
    return audio


def build_default_stages(settings: PipelineSettings) -> Stages:
    """真实实现。只有走到这里才 import 模型相关依赖。"""
    import torch  # pylint: disable=import-outside-toplevel

    from ai.denoise.inference import denoise_array  # pylint: disable=import-outside-toplevel
    from ai.denoise.model import DenoiseCRNN  # pylint: disable=import-outside-toplevel
    from ai.llm.corrector import LLMCorrector  # pylint: disable=import-outside-toplevel
    from ai.vad.detector import VADDetector  # pylint: disable=import-outside-toplevel
    from ai.whisper.transcriber import WhisperTranscriber  # pylint: disable=import-outside-toplevel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenoiseCRNN().to(device)
    weights = settings.resolved_weights()
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()
        print(f"降噪权重已加载：{weights}")
    else:
        print(f"警告：没找到降噪权重 {weights}，用未训练权重跑（效果会差）")

    transcriber = WhisperTranscriber(model_id=settings.whisper_model_id)
    corrector = LLMCorrector(model_id=settings.llm_model_id) if settings.use_llm else None

    return Stages(
        vad=VADDetector(threshold=settings.vad_threshold, sample_rate=settings.sample_rate),
        denoise=lambda audio: denoise_array(model, audio, sr=settings.sample_rate),
        transcribe=lambda audio: transcriber.transcribe(audio, sample_rate=settings.sample_rate),
        correct_stream=(corrector.correct_stream if corrector else None),
    )


class AIPipeline:
    """一条连接一个实例：VAD 缓冲和去重状态都是会话级的。"""

    def __init__(self, stages: Stages | None = None, settings: PipelineSettings | None = None) -> None:
        self.settings = settings or load_pipeline_settings()
        self.stages = stages or build_default_stages(self.settings)

        self.vad = self.stages.vad
        self.use_llm = self.settings.use_llm and self.stages.correct_stream is not None

        self._repeat_filter = RepeatFilter(self.settings.repeat_threshold)
        self._history = TextHistory(
            size=self.settings.history_size,
            threshold=self.settings.history_threshold,
        )

    # --- 对外接口 -----------------------------------------------------

    def process_streaming_chunk(self, audio_chunk: np.ndarray):
        """喂一段音频，成句时返回结果字典，否则 None。"""
        vad_result = self.vad.process_chunk(audio_chunk)
        if vad_result is None:
            return None
        phrase, segment_id = vad_result
        return self._process_phrase(phrase, segment_id)

    def get_buffer_snapshot(self) -> tuple[np.ndarray, str] | tuple[None, None]:
        """看一眼当前缓冲，不消费。"""
        if not self.vad.buffer:
            return None, None
        return np.concatenate(self.vad.buffer), getattr(self.vad, "current_segment_id", "")

    def quick_transcribe(self, audio: np.ndarray) -> str:
        """临时预览：降噪 + 转录，不过 LLM。"""
        return self.stages.transcribe(normalize_audio(self.stages.denoise(audio)))

    def force_emit(self):
        """把缓冲里的残余音频强行成句。"""
        vad_result = self.vad.force_emit()
        if vad_result is None:
            return None
        phrase, segment_id = vad_result
        return self._process_phrase(phrase, segment_id)

    def correct_stream(self, raw_text: str):
        """返回纠错的流式生成器；不用 LLM 时返回 None。"""
        if not self.use_llm or not raw_text.strip() or self.stages.correct_stream is None:
            return None

        source = raw_text
        history = self._history
        corrector = self.stages.correct_stream

        def _generate() -> Iterator[str]:
            collected: list[str] = []
            for token in corrector(source, history.entries):
                collected.append(token)
                yield token

            corrected = "".join(collected).strip()
            if not corrected:
                return
            if not history.commit(corrected):
                print(f"[LLM dedup] 纠错结果和上下文太像，丢弃：{corrected[:20]}…")

        return _generate()

    # --- 内部 ---------------------------------------------------------

    def _process_phrase(self, phrase: np.ndarray, segment_id: str) -> dict | None:
        denoised = normalize_audio(self.stages.denoise(phrase))
        raw_text = self.stages.transcribe(denoised)

        if not self._repeat_filter.accept(raw_text):
            return None

        return {
            "segment_id": segment_id,
            "raw_text": raw_text,
            "corrected_text": raw_text,
        }
