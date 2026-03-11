from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import numpy as np
import io
import soundfile as sf
import os
import sys
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add project root to path so 'ai.pipeline' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.pipeline import AIPipeline
import json

router = APIRouter()

# Sentinel returned by _safe_next when the generator is exhausted.
# Using run_in_executor(next, gen) would propagate StopIteration into a Future, which
# Python 3.7+ (PEP 479) forbids — TypeError: StopIteration interacts badly with generators.
_STOP = object()

def _safe_next(gen):
    """Call next(gen) in a thread-pool-safe way; return _STOP instead of raising StopIteration."""
    try:
        return next(gen)
    except StopIteration:
        return _STOP

# Thread pool for running blocking Whisper inference without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=2)
# Single-threaded executor that serialises all pipeline calls (VAD + Whisper share state)
_pipeline_executor = ThreadPoolExecutor(max_workers=1)

# How often (seconds) to emit an interim partial transcription while speech is buffering
INTERIM_INTERVAL_S = 1.5
# Minimum buffered audio (samples) before attempting a partial transcription (avoid noise-only results)
MIN_INTERIM_SAMPLES = 20000  # ~1.25 s @ 16 kHz

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
    
    print("New WebSocket client connected for pseudo-streaming ASR.")

    last_interim_time = 0.0
    partial_task: asyncio.Task | None = None
    llm_task: asyncio.Task | None = None
    use_llm = True  # default; toggled by frontend control messages

    try:
        while True:
            # WebSocket can receive either binary audio or JSON control messages
            message = await websocket.receive()

            if message["type"] == "websocket.receive":
                # JSON text control message (e.g. {"type":"set_use_llm","value":false})
                if "text" in message and message["text"]:
                    try:
                        ctrl = json.loads(message["text"])
                        if ctrl.get("type") == "set_use_llm":
                            use_llm = bool(ctrl.get("value", True))
                            print(f"use_llm set to {use_llm}")
                    except Exception:
                        pass
                    continue

                # Binary audio data
                data = message.get("bytes")
                if not data:
                    continue

            elif message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(code=message.get("code", 1000))
            else:
                continue

            try:
                pcm_data = np.frombuffer(data, dtype=np.int16)
                audio_float32 = pcm_data.astype(np.float32) / 32768.0

                loop = asyncio.get_event_loop()
                # Run pipeline (VAD + Whisper) in the dedicated single-thread executor so the
                # event loop is never blocked, and calls are serialised (no concurrent Whisper).
                result = await loop.run_in_executor(
                    _pipeline_executor, pipeline.process_streaming_chunk, audio_float32
                )
                
                if result is not None:
                    # Cancel any pending partial / LLM tasks
                    for t in (partial_task, llm_task):
                        if t and not t.done():
                            t.cancel()
                    last_interim_time = 0.0

                    raw_text = result.get("raw_text", "")
                    segment_id = result.get("segment_id", "")

                    # Send Whisper result — frontend shows this as gray (awaiting LLM)
                    await websocket.send_json({
                        "type": "transcription_raw",
                        "segment_id": segment_id,
                        "raw_text": raw_text,
                    })

                    # Optionally start streaming LLM correction
                    if use_llm and raw_text.strip():
                        llm_task = asyncio.create_task(
                            _stream_llm(websocket, pipeline, raw_text, segment_id)
                        )
                else:
                    # Interim partial preview
                    now = time.monotonic()
                    if now - last_interim_time >= INTERIM_INTERVAL_S:
                        snapshot, seg_id = pipeline.get_buffer_snapshot()
                        if snapshot is not None and len(snapshot) >= MIN_INTERIM_SAMPLES:
                            last_interim_time = now
                            if partial_task is None or partial_task.done():
                                partial_task = asyncio.create_task(
                                    _send_partial(websocket, pipeline, snapshot, seg_id)
                                )

            except Exception as e:
                print(f"Error processing chunk: {e}")
                await websocket.send_json({"type": "error", "message": "Failed to process audio chunk"})
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
        for t in (partial_task, llm_task):
            if t and not t.done():
                t.cancel()
        if pipeline:
            final_result = pipeline.force_emit()
            if final_result:
                print(f"Final phrase (disconnected): {final_result.get('raw_text')}")


async def _stream_llm(websocket: WebSocket, pipeline, raw_text: str, segment_id: str):
    """Run LLM correction stream and push each token to the frontend."""
    loop = asyncio.get_event_loop()
    collected_tokens = []
    try:
        gen = await loop.run_in_executor(_executor, pipeline.correct_stream, raw_text)
        if gen is None:
            # LLM disabled — mark segment as final with the raw text
            await websocket.send_json({
                "type": "llm_done",
                "segment_id": segment_id,
                "corrected_text": raw_text,
            })
            return
        # Iterate the synchronous generator in the thread pool token by token.
        # Use _safe_next to avoid StopIteration escaping into asyncio Future machinery.
        while True:
            token = await loop.run_in_executor(_executor, _safe_next, gen)
            if token is _STOP:
                break
            collected_tokens.append(token)
            await websocket.send_json({"type": "llm_token", "segment_id": segment_id, "token": token})
        corrected_text = "".join(collected_tokens).strip() or raw_text
        await websocket.send_json({
            "type": "llm_done",
            "segment_id": segment_id,
            "corrected_text": corrected_text,
        })
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"LLM stream error: {e}")


async def _send_partial(websocket: WebSocket, pipeline, audio: np.ndarray, segment_id: str):
    """Run a quick (no-LLM) transcription in the thread pool and send as a partial preview."""
    loop = asyncio.get_event_loop()
    try:
        # Use the same single-thread pipeline executor to serialise Whisper access
        text = await loop.run_in_executor(_pipeline_executor, pipeline.quick_transcribe, audio)
        if text and text.strip():
            await websocket.send_json({
                "type": "partial",
                "segment_id": segment_id,
                "text": text,
            })
    except asyncio.CancelledError:
        pass  # Superseded by a committed phrase — silently discard
    except Exception as e:
        print(f"Partial transcription error: {e}")
