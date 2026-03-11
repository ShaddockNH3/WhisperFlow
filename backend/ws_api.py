from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import numpy as np
import io
import soundfile as sf
import os
import sys

# Add project root to path so 'ai.pipeline' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.pipeline import AIPipeline
import json

router = APIRouter()

# Global pipeline instance (loaded lazily or on startup)
pipeline_instance = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        print("Initializing Global AI Pipeline Instance for WebSockets...")
        pipeline_instance = AIPipeline()
    return pipeline_instance

@router.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    await websocket.accept()
    pipeline = get_pipeline()
    
    # Store buffer for incomplete chunks or resampling history if needed
    # But since VAD handles chunking, we mostly just decode each WebM/Blob
    # sent by MediaRecorder and pass to pipeline
    
    print("New WebSocket client connected for pseudo-streaming ASR.")
    
    try:
        while True:
            # Receive binary audio chunk from browser MediaRecorder (usually webm/ogg)
            data = await websocket.receive_bytes()
            
            # The browser sends Blob data. We need to decode it.
            # MediaRecorder chunks are often self-contained enough to decode with soundfile
            # or we might need ffmpeg depending on the codec (webm/opus).
            # If the frontend converts to PCM Float32 first, it's easier.
            
            # Let's assume the frontend sends raw Int16 PCM array to simplify backend decoding latency
            # We will convert it to Float32 immediately.
            
            try:
                # Assuming raw 16-bit PCM (Int16) at 16000 Hz sent from frontend
                # For a production app, WebM/Opus -> FFmpeg -> PCM is needed,
                # but for this pseudo-streaming prototype, raw PCM is fastest.
                pcm_data = np.frombuffer(data, dtype=np.int16)
                
                # Convert to Float32 (-1.0 to 1.0)
                audio_float32 = pcm_data.astype(np.float32) / 32768.0
                
                # Pass to pipeline
                result = pipeline.process_streaming_chunk(audio_float32)
                
                if result is not None:
                    # VAD detected a phrase boundary, returned transcription
                    await websocket.send_json({
                        "type": "transcription",
                        "raw_text": result.get("raw_text", ""),
                        "corrected_text": result.get("corrected_text", "")
                    })
                    
            except Exception as e:
                print(f"Error processing chunk: {e}")
                await websocket.send_json({"type": "error", "message": "Failed to process audio chunk"})
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
        # Force emit any remaining audio in the VAD buffer
        if pipeline:
            final_result = pipeline.force_emit()
            if final_result:
                 # Can't send to closed socket, but would normally log or save it
                 print(f"Final phrase (disconnected): {final_result.get('corrected_text')}")
