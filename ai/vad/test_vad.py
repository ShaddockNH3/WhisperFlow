import numpy as np
import time
from detector import VADDetector

def test_vad_streaming():
    """
    Simulates a streaming environment by feeding random audio chunks 
    to the VAD Detector to test the state machine and chunking logic.
    """
    samplerate = 16000
    
    # 1. Initialize detector (400ms silence threshold)
    detector = VADDetector(threshold=0.3, min_silence_duration_ms=400, sample_rate=samplerate)
    print("VAD initialized successfully.")
    
    print("\n--- Simulating Streaming Audio ---")
    
    # Simulate receiving 512-sample chunks (about 32ms per chunk)
    chunk_size = 512
    
    # Generate 2 seconds of pure silence (noise)
    print("Sending 2s of silence...")
    for _ in range(int(2.0 * samplerate / chunk_size)):
        silent_chunk = np.random.normal(0, 0.001, chunk_size).astype(np.float32)
        res = detector.process_chunk(silent_chunk)
        if res is not None:
             print("ERROR: VAD emitted a chunk during pure silence!")
             
    # Generate 1.5 seconds of "loud noise" simulating speech
    print("Sending 1.5s of 'speech' (loud signal)...")
    for _ in range(int(1.5 * samplerate / chunk_size)):
        # Generate loud sine wave mixed with noise to ensure VAD triggers
        t = np.linspace(0, chunk_size / samplerate, chunk_size)
        speech_chunk = (0.5 * np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.05, chunk_size)).astype(np.float32)
        res = detector.process_chunk(speech_chunk)
        if res is not None:
             print("ERROR: VAD emitted prematurely during active speech!")
             
    # Generate 0.5 seconds of silence (this should trigger the emission)
    print("Sending 0.5s of trailing silence to trigger emission...")
    emission_triggered = False
    for i in range(int(0.5 * samplerate / chunk_size)):
        silent_chunk = np.random.normal(0, 0.001, chunk_size).astype(np.float32)
        res = detector.process_chunk(silent_chunk)
        
        if res is not None:
            print(f"✅ SUCCESS! VAD successfully detected end of speech and chunked the audio.")
            print(f"Emitted Audio Length: {len(res) / samplerate:.2f} seconds.")
            emission_triggered = True
            break
            
    if not emission_triggered:
        print("❌ FAILED: VAD did not trigger an emission after 400ms of silence.")
        
    print("\nTest completed.")

if __name__ == "__main__":
    test_vad_streaming()
