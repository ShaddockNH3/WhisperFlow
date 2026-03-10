import sys
import os
import asyncio
import numpy as np
import io
import soundfile as sf
import librosa

# Add the 'ai' directory to sys.path so we can import from the root 'ai' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
ai_dir = os.path.join(current_dir, '../../ai')
sys.path.append(ai_dir)

from vad.detector import VADDetector
from whisper.transcriber import WhisperTranscriber
from llm.corrector import LLMCorrector

# Lazily load models globally to prevent reloading on every request
class AIPipeline:
    def __init__(self):
        self.vad = None
        self.transcriber = None
        self.corrector = None
        self._initialized = False
        
    def initialize(self):
        if not self._initialized:
            print("Initializing AI Pipeline (VAD, Whisper, LLM)... Please wait.")
            self.vad = VADDetector(threshold=0.3, min_silence_duration_ms=400, sample_rate=16000)
            self.transcriber = WhisperTranscriber(model_id="openai/whisper-small")
            self.corrector = LLMCorrector()
            self._initialized = True

pipeline = AIPipeline()

async def process_audio(audio_data: bytes) -> str:
    """
    Main entry point for API to process audio using the 3-Stage Pipeline.
    """
    if not pipeline._initialized:
        # In production this might be loaded at app startup or via BackgroundTasks
        pipeline.initialize()
        
    # 1. Convert bytes to 16kHz numpy array
    try:
        audio_stream = io.BytesIO(audio_data)
        waveform, sr = sf.read(audio_stream)
        
        # Convert to float32
        waveform = waveform.astype(np.float32)
        
        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
            
        # Resample if needed
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return f"[Error: Unable to process audio format - {str(e)}]"

    # 2. VAD Chunking (Simulate streaming through the detector)
    chunk_size = 512
    phrases = []
    
    for i in range(0, len(waveform), chunk_size):
        chunk = waveform[i:i + chunk_size]
        result = pipeline.vad.process_chunk(chunk)
        if result is not None:
            phrases.append(result)
            
    # Force emit any remaining audio in the buffer
    final_chunk = pipeline.vad.force_emit()
    if final_chunk is not None:
        phrases.append(final_chunk)
        
    if not phrases:
        # Fallback if VAD didn't trigger, treat whole file as one phrase
        phrases = [waveform]

    # 3. Transcribe & Correct
    full_transcription = []
    history = []
    
    print(f"Processing {len(phrases)} phonetic chunks detected by VAD...")
    for phrase in phrases:
        # Stage 3a: Whisper (requires numpy array)
        raw_text = pipeline.transcriber.transcribe(phrase, sample_rate=16000)
        
        # Stage 3b: LLM Corrector
        corrected_text = pipeline.corrector.correct(raw_text, historical_context=history)
        
        if corrected_text.strip():
            full_transcription.append(corrected_text)
            history.append(corrected_text)
            
    return " ".join(full_transcription)
