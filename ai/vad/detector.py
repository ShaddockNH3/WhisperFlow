import torch
import torchaudio
import numpy as np
import os

class VADDetector:
    """
    Wrapper for Silero VAD (Voice Activity Detection).
    Handles dynamic audio streaming, silence detection, and chunking.
    """
    def __init__(self, threshold: float = 0.15, min_silence_duration_ms: int = 600, sample_rate: int = 16000,
                 max_speech_duration_s: float = 5.0):
        self.sample_rate = sample_rate
        self.threshold = threshold
        # Convert ms silence duration to number of samples (since VAD output is granular)
        self.min_silence_samples = int((min_silence_duration_ms / 1000.0) * sample_rate)
        # Max speech buffer before force-emitting (prevents indefinite buffering of continuous audio)
        self.max_speech_samples = int(max_speech_duration_s * sample_rate)
        
        # Load the PyTorch Hub version of Silero VAD (which resolves to JS/ONNX under the hood if needed)
        # Using torch.hub guarantees we get the optimized JIT version
        print("Loading Silero VAD model from Torch Hub...")
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False)
                                           
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
         
        self.model.eval()
        
        # Tracking states for streaming
        self.buffer = []
        self.vad_probs = []  # VAD probability per chunk, used for smart cut-point detection
        self.silence_accumulator = 0  # Number of silent samples currently tracked
        self.is_speaking = False
        
    def _array_to_tensor(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """Convert incoming numpy float32 chunk (-1.0 to 1.0) to torch tensor."""
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        return torch.from_numpy(audio_chunk)
        
    def process_chunk(self, audio_chunk: np.ndarray):
        """
        Receives a continuous stream of audio chunks (e.g., 512 samples at a time).
        Returns a complete phrase as a numpy array if a trailing silence is detected.
        Otherwise, returns None, buffering the audio internally.
        """
        # 1. Append to internal buffer cache
        self.buffer.append(audio_chunk)
        
        # 2. Run VAD on this specific chunk
        tensor_chunk = self._array_to_tensor(audio_chunk)        
        # Silero VAD processes blocks of 512 samples. 
        # If the chunk is exactly 512, this works natively, otherwise we pad it.
        # But for inference, `model(tensor_chunk)` directly returns a float probability
        # In a real streaming scenario, we might use VADIterator, but for simplicity here
        # we evaluate probabilities block by block.
        
        chunk_len = tensor_chunk.shape[0]
        
        # Make sure chunk is divisible by 512 for optimal Silero performance, 
        # otherwise just pad temporarily (only for VAD check, buffer keeps original data).
        if chunk_len < 512:
            pad = torch.zeros(512 - chunk_len, dtype=torch.float32)
            vad_tensor = torch.cat([tensor_chunk, pad])
        else:
            vad_tensor = tensor_chunk[:512] # Just check the first 512 for this instant
            
        with torch.no_grad():
            speech_prob = self.model(vad_tensor, self.sample_rate).item()

        self.vad_probs.append(speech_prob)
            
        # 3. State Machine Logic
        if speech_prob > self.threshold:
            self.is_speaking = True
            self.silence_accumulator = 0

            # Force emit if speech buffer exceeded max duration.
            # Instead of cutting at the current position (mid-word), find the chunk
            # with the lowest VAD probability in the second half of the buffer —
            # that is the most natural pause point to cut at.
            buffered_samples = sum(c.shape[0] for c in self.buffer)
            if buffered_samples >= self.max_speech_samples:
                # Search for best cut point in the latter half to avoid cutting too early
                search_start = len(self.buffer) // 2
                cut_idx = search_start + int(np.argmin(self.vad_probs[search_start:]))

                phrase = np.concatenate(self.buffer[:cut_idx])
                # Keep the audio after the cut point for the next phrase
                self.buffer = self.buffer[cut_idx:]
                self.vad_probs = self.vad_probs[cut_idx:]
                self.silence_accumulator = 0
                self.model.reset_states()
                return phrase
        else:
            if self.is_speaking:
                self.silence_accumulator += chunk_len
                
                # Check if silence limit is reached
                if self.silence_accumulator >= self.min_silence_samples:
                    # Target Reached! Cut the buffer and emit the phrase.
                    phrase = np.concatenate(self.buffer)
                    # Reset internal state
                    self.buffer = []
                    self.vad_probs = []
                    self.silence_accumulator = 0
                    self.is_speaking = False
                    
                    # Reset VAD internal RNN states
                    self.model.reset_states() 
                    
                    return phrase
                    
        return None

    def force_emit(self):
        """Forces the current buffer to emit regardless of silence (e.g. at end of connection)."""
        if len(self.buffer) > 0:
            phrase = np.concatenate(self.buffer)
            self.buffer = []
            self.vad_probs = []
            self.silence_accumulator = 0
            self.is_speaking = False
            self.model.reset_states()
            return phrase
        return None
