import torch
from transformers import pipeline

class WhisperTranscriber:
    """
    Wrapper for OpenAI's Whisper model via Hugging Face Transformers.
    Utilizes FP16 half-precision on GPU if available for ultra-low latency.
    """
    def __init__(self, model_id: str = "openai/whisper-small"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading Whisper model '{model_id}' on {self.device} with dtype {self.torch_dtype}...")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_kwargs={"use_flash_attention_2": False} # Set to True in prod if Flash Attention is installed
        )
        
        # Pre-warm the model with a blank audio tensor specifically to force compilation/caching if needed
        print("Whisper model loaded successfully.")

    def transcribe(self, audio_array: list, sample_rate: int = 16000) -> str:
        """
        Transcribes a 1D audio numpy array (16kHz).
        Returns the recognized transcript string.
        """
        if len(audio_array) == 0:
            return ""
            
        # The transformers pipeline accepts a dictionary with 'array' and 'sampling_rate'
        result = self.pipe({"array": audio_array, "sampling_rate": sample_rate}, generate_kwargs={"language": "chinese"})
        return result['text']
