import sys
import os
import soundfile as sf
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.pipeline import AIPipeline

def test_pipeline():
    # Load noisy test audio
    test_file = 'ai/data_thchs30/test/D12_989.wav'
    print(f"Loading test file: {test_file}")
    
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return
        
    data, sr = sf.read(test_file, dtype='float32')
    
    # Needs to be 16kHz
    if sr != 16000:
        import torchaudio
        import torch
        print("Resampling to 16kHz...")
        waveform = torch.from_numpy(data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        data = resampler(waveform).squeeze(0).numpy()
        
    # Convert to mono if it's not
    if data.ndim > 1:
        data = data.mean(axis=1)
        
    print(f"Total audio length: {len(data) / 16000:.2f} seconds")
    
    # Initialize pipeline
    pipeline = AIPipeline()
    
    # Simulate streaming
    chunk_size = 512 # simulated chunk size
    num_chunks = len(data) // chunk_size
    
    print("\n--- Starting simulated stream ---")
    start_time = time.time()
    
    for i in range(num_chunks + 1):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        
        chunk = data[start_idx:end_idx]
        if len(chunk) == 0:
            break
            
        result = pipeline.process_streaming_chunk(chunk)
        
        if result:
            print(f"\n[Phrase Detected @ {(end_idx / 16000):.2f}s]")
            print(f"Raw Whisper Text: {result.get('raw_text')}")
            print(f"LLM Corrected Text: {result.get('corrected_text')}")
            
    # Force emit any remaining audio at the end
    result = pipeline.force_emit()
    if result:
        print(f"\n[Final Flush]")
        print(f"Raw Whisper Text: {result.get('raw_text')}")
        print(f"LLM Corrected Text: {result.get('corrected_text')}")
        
    elapsed = time.time() - start_time
    print(f"\n--- Stream finished in {elapsed:.2f} seconds ---")

if __name__ == "__main__":
    test_pipeline()
