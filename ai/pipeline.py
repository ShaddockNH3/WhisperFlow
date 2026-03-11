import os
import sys
import difflib
import torch
import numpy as np

# Add project root to path (if running independently)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.vad.detector import VADDetector
from ai.denoise.model import DenoiseCRNN
from ai.denoise.inference import denoise_array
from ai.whisper.transcriber import WhisperTranscriber
from ai.llm.corrector import LLMCorrector

class AIPipeline:
    def __init__(self, 
                 denoise_model_path: str = 'ai/denoise/weights/denoise_crnn.pt',
                 whisper_model_id: str = "openai/whisper-turbo",
                 llm_model_id: str = "glm-4-flash-250414",
                 use_llm: bool = True):
        print("Initializing AI Pipeline...")
        
        # Determine the directory where the executable or script is located
        if getattr(sys, 'frozen', False):
            # The directory where the .exe / binary resides
            app_root = os.path.dirname(sys.executable)
            base_path = sys._MEIPASS # Temp dir for bundled assets
        else:
            app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_path = app_root
            
        # Denoise model weight path (usually bundled, so use base_path)
        if not os.path.isabs(denoise_model_path):
            denoise_model_path = os.path.join(base_path, denoise_model_path)
            
        # Ensure model cache is in a folder NEXT TO the executable for portability
        model_cache_dir = os.path.join(app_root, "data", "models")
        os.makedirs(model_cache_dir, exist_ok=True)
        
        # Set environment variables for cache redirection to ensure NO traces in home dir
        os.environ["HF_HOME"] = os.path.join(model_cache_dir, "huggingface")
        os.environ["XDG_CACHE_HOME"] = os.path.join(model_cache_dir, "xdg")
        os.environ["TORCH_HOME"] = os.path.join(model_cache_dir, "torch")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize VAD — silero-vad will download to ~/.cache/torch/hub or HF_HOME
        self.vad = VADDetector(threshold=0.15, sample_rate=16000)
        
        # 2. Initialize Denoise Model
        self.denoise_model = DenoiseCRNN().to(self.device)
        if os.path.exists(denoise_model_path):
            self.denoise_model.load_state_dict(torch.load(denoise_model_path, map_location=self.device))
            self.denoise_model.eval()
            print(f"Denoise model loaded from {denoise_model_path}")
        else:
            # For lightweight build, if weight is not in app dir, check cache dir
            cached_denoise = os.path.join(model_cache_dir, "denoise_crnn.pt")
            if os.path.exists(cached_denoise):
                self.denoise_model.load_state_dict(torch.load(cached_denoise, map_location=self.device))
                self.denoise_model.eval()
                print(f"Denoised model loaded from cache: {cached_denoise}")
            else:
                print(f"Warning: Denoise model weights not found. Running without denoising weights.")
            
        # 3. Initialize Whisper (faster-whisper handles downloads via 'download_root')
        self.transcriber = WhisperTranscriber(
            model_id=whisper_model_id, 
            download_root=os.path.join(model_cache_dir, "whisper")
        )
        
        # 4. Initialize LLM Corrector
        self.use_llm = use_llm
        if self.use_llm:
            self.corrector = LLMCorrector(model_id=llm_model_id)
            self.historical_context = []

        # Track the last committed text for deduplication
        self._last_committed_text: str = ""
            
        print("AI Pipeline initialized successfully.")

    def process_streaming_chunk(self, audio_chunk: np.ndarray):
        """
        Receives a chunk of audio (e.g. 512 samples, 16kHz, float32).
        Returns a dictionary with results if a phrase was completed, otherwise None.
        """
        # 1. VAD to check for phrase boundaries
        vad_result = self.vad.process_chunk(audio_chunk)
        
        if vad_result is not None:
            phrase, segment_id = vad_result
            return self._process_phrase(phrase, segment_id)
            
        return None
        
    def get_buffer_snapshot(self) -> tuple[np.ndarray, str] | tuple[None, None]:
        """Returns a copy of the current VAD audio buffer and its segment_id without consuming it."""
        if not self.vad.buffer:
            return None, None
        return np.concatenate(self.vad.buffer), self.vad.current_segment_id

    def quick_transcribe(self, audio: np.ndarray) -> str:
        """Denoise + transcribe only (no LLM correction), used for interim partial results."""
        denoised = denoise_array(self.denoise_model, audio, sr=16000)
        max_val = np.max(np.abs(denoised))
        if max_val > 0.0:
            denoised = denoised / max_val
        return self.transcriber.transcribe(denoised, sample_rate=16000)

    def force_emit(self):
        """Forces the VAD to emit whatever is in its buffer."""
        vad_result = self.vad.force_emit()
        if vad_result is not None:
            phrase, segment_id = vad_result
            return self._process_phrase(phrase, segment_id)
        return None

    def _process_phrase(self, phrase: np.ndarray, segment_id: str):
        """Internal method to process a complete continuous speech phrase."""
        result = {'segment_id': segment_id}
        
        # 2. Denoise
        denoised_phrase = denoise_array(self.denoise_model, phrase, sr=16000)
        
        # Normalize the denoised audio to avoid clipping issues with Whisper
        max_val = np.max(np.abs(denoised_phrase))
        if max_val > 0.0:
             denoised_phrase = denoised_phrase / max_val
             
        # 3. Transcribe
        raw_text = self.transcriber.transcribe(denoised_phrase, sample_rate=16000)

        # Deduplication: discard if too similar to the last committed phrase.
        # Similarity > 0.85 almost always means Whisper hallucinated the same sentence again.
        if raw_text.strip():
            similarity = difflib.SequenceMatcher(
                None, raw_text.strip(), self._last_committed_text
            ).ratio()
            if similarity > 0.85:
                return None  # silently discard the hallucinated repetition
            self._last_committed_text = raw_text.strip()

        result['raw_text'] = raw_text
        result['corrected_text'] = raw_text  # default, overwritten if LLM is used
            
        return result

    def correct_stream(self, raw_text: str):
        """
        Returns a streaming generator for LLM correction.
        Also updates historical_context with the final result after iteration.
        """
        if not self.use_llm or not raw_text.strip():
            return

        collected = []

        def _gen():
            for token in self.corrector.correct_stream(raw_text, self.historical_context):
                collected.append(token)
                yield token
            # Update history after stream completes, with deduplication guard.
            corrected = "".join(collected).strip()
            if not corrected:
                return
            # Discard if the corrected output is highly similar to any recent history entry
            # (catches LLM hallucinations that reproduce previous context verbatim).
            for prev in self.historical_context[-3:]:
                sim = difflib.SequenceMatcher(None, corrected, prev).ratio()
                if sim > 0.7:
                    print(f"[LLM dedup] Discarding corrected output (similarity {sim:.2f} to history).")
                    return
            self.historical_context.append(corrected)
            if len(self.historical_context) > 10:
                self.historical_context.pop(0)

        return _gen()
