import os
import sys
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
                 llm_model_id: str = "qwen/Qwen-1_8B-Chat",
                 use_llm: bool = True):
        print("Initializing AI Pipeline...")
        
        # Resolve relative model path based on project root to allow starting backend from /backend dir
        if not os.path.isabs(denoise_model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            denoise_model_path = os.path.join(project_root, denoise_model_path)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize VAD (using softer threshold to avoid word clipping)
        self.vad = VADDetector(threshold=0.15, min_silence_duration_ms=600, sample_rate=16000)
        
        # 2. Initialize Denoise Model
        self.denoise_model = DenoiseCRNN().to(self.device)
        if os.path.exists(denoise_model_path):
            self.denoise_model.load_state_dict(torch.load(denoise_model_path, map_location=self.device))
            self.denoise_model.eval()
            print("Denoise model loaded.")
        else:
            print(f"Warning: Denoise model weights not found at {denoise_model_path}. Denoising will be skipped or use untrained weights.")
            
        # 3. Initialize Whisper
        self.transcriber = WhisperTranscriber(model_id=whisper_model_id)
        
        # 4. Initialize LLM Corrector
        self.use_llm = use_llm
        if self.use_llm:
            self.corrector = LLMCorrector(model_id=llm_model_id)
            self.historical_context = []
            
        print("AI Pipeline initialized successfully.")

    def process_streaming_chunk(self, audio_chunk: np.ndarray):
        """
        Receives a chunk of audio (e.g. 512 samples, 16kHz, float32).
        Returns a dictionary with results if a phrase was completed, otherwise None.
        """
        # 1. VAD to check for phrase boundaries
        phrase = self.vad.process_chunk(audio_chunk)
        
        if phrase is not None:
            return self._process_phrase(phrase)
            
        return None
        
    def force_emit(self):
        """Forces the VAD to emit whatever is in its buffer."""
        phrase = self.vad.force_emit()
        if phrase is not None:
            return self._process_phrase(phrase)
        return None

    def _process_phrase(self, phrase: np.ndarray):
        """Internal method to process a complete continuous speech phrase."""
        result = {}
        
        # 2. Denoise
        # Call the standalone denoise function we refactored
        denoised_phrase = denoise_array(self.denoise_model, phrase, sr=16000)
        
        # Normalize the denoised audio to avoid clipping issues with Whisper
        # (Optional but usually good practice after ISTFT)
        max_val = np.max(np.abs(denoised_phrase))
        if max_val > 0.0:
             denoised_phrase = denoised_phrase / max_val
             
        # 3. Transcribe
        raw_text = self.transcriber.transcribe(denoised_phrase, sample_rate=16000)
        result['raw_text'] = raw_text
        
        # 4. LLM Correction
        if self.use_llm and raw_text.strip():
            corrected_text = self.corrector.correct(raw_text, self.historical_context)
            result['corrected_text'] = corrected_text
            
            # Update history
            self.historical_context.append(corrected_text)
            if len(self.historical_context) > 10:  # Keep bounded
                self.historical_context.pop(0)
        else:
            result['corrected_text'] = raw_text
            
        return result
