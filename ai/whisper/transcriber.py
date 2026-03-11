import threading
import torch
from faster_whisper import WhisperModel
import opencc

class WhisperTranscriber:
    """
    Wrapper for faster-whisper (CTranslate2 backend) for ultra-fast, low-latency inference.
    Also includes OpenCC to convert Traditional Chinese to Simplified automatically.
    """
    def __init__(self, model_id: str = "base"):
        # If user passed an openai tag like "openai/whisper-turbo", extract the last part
        if "/" in model_id:
            model_id = model_id.split("/")[-1]
        # faster-whisper expects "large-v3-turbo" instead of "whisper-turbo", handle some common mappings
        if model_id == "whisper-turbo":
             model_id = "large-v3-turbo"
        elif model_id.startswith("whisper-"):
             model_id = model_id.replace("whisper-", "")
             
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        print(f"Loading faster-whisper model '{model_id}' on {self.device} with {self.compute_type}...")
        
        self.model = WhisperModel(model_id, device=self.device, compute_type=self.compute_type)
        self.cc = opencc.OpenCC('t2s')  # Traditional to Simplified conversion
        # Serialise all inference calls — prevents race condition between
        # partial-preview and committed-phrase transcriptions running concurrently.
        self._lock = threading.Lock()
        
        print("Faster-Whisper model loaded successfully.")

    def transcribe(self, audio_array, sample_rate: int = 16000) -> str:
        """
        Transcribes a 1D audio numpy array (16kHz).
        Returns the recognized transcript string in Simplified Chinese.
        """
        if len(audio_array) == 0:
            return ""

        with self._lock:
            # faster-whisper model.transcribe accepts numpy arrays, forces language="zh"
            segments, info = self.model.transcribe(
                audio_array,
                language="zh",
                beam_size=5,
                condition_on_previous_text=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                temperature=[0.0, 0.2, 0.4],
                # Reject segments that are almost certainly hallucinations
                compression_ratio_threshold=2.0,
                log_prob_threshold=-0.8,
                no_speech_threshold=0.6,
            )

            # Eagerly materialise segments inside the lock so the model stays
            # consistent until we have finished reading all results.
            collected = [
                seg.text for seg in segments
                if seg.compression_ratio <= 2.0
                and seg.avg_logprob >= -0.8
                and seg.no_speech_prob < 0.6
            ]

        raw_text = "".join(collected)

        # Convert Traditional to Simplified Chinese
        simplified_text = self.cc.convert(raw_text)

        return simplified_text
