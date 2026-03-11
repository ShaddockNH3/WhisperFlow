import torch
import numpy as np
import uuid


class VADDetector:
    """
    Wrapper for Silero VAD with two ultra-long utterance defence strategies:

    Strategy 1 — Dynamic silence threshold
        The longer the user has been speaking without a cut, the more aggressively
        we look for a silence boundary:
            0 – 5 s   buffered → 800 ms silence triggers cut  (full sentence context)
            5 – 10 s  buffered → 300 ms silence triggers cut  (catch micro-pauses)
            10 s+     buffered → 100 ms silence triggers cut  (grab any valley)

    Strategy 2 — Soft max-duration cut (default 15 s)
        When the buffer hits the hard cap we never cut blindly at the current frame.
        Instead, we look back over the most recent `lookback_s` seconds and find
        the frame whose VAD probability is the local minimum (most natural pause).
        If all frames in the lookback window are above the speech threshold we fall
        back to the minimum-RMS frame to avoid splitting a phoneme.
    """

    # (max_buffered_seconds, silence_threshold_ms)  — checked in order
    SILENCE_TIERS: list[tuple[float, int]] = [
        (5.0,          200),
        (10.0,         120),
        (float("inf"), 60),
    ]

    def __init__(
        self,
        threshold: float = 0.15,
        sample_rate: int = 16000,
        max_speech_duration_s: float = 10.0,
        soft_cut_interval_s: float = 3.0,
        lookback_s: float = 1.5,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.max_speech_samples = int(max_speech_duration_s * sample_rate)
        # Proactive soft-cut: if no silence-triggered cut has happened within this
        # many samples of continuous speech, find the local minimum and cut anyway.
        self.soft_cut_interval_samples = int(soft_cut_interval_s * sample_rate)
        # Number of samples to scan backwards when doing a soft-cut
        self.lookback_samples = int(lookback_s * sample_rate)
        # Samples accumulated since the last cut (either silence or soft-cut)
        self._samples_since_cut: int = 0

        print("Loading Silero VAD model from Torch Hub...")
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils
        self.model.eval()

        # Streaming state
        self.buffer: list[np.ndarray] = []
        self.vad_probs: list[float] = []
        self.silence_accumulator: int = 0   # consecutive silent samples
        self.is_speaking: bool = False
        self.current_segment_id: str = ""
        self._samples_since_cut: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, chunk: np.ndarray) -> torch.Tensor:
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        return torch.from_numpy(chunk)

    def _dynamic_silence_samples(self) -> int:
        """Return the current silence-trigger threshold (samples) based on how much
        speech is already buffered (Strategy 1)."""
        buffered_s = sum(c.shape[0] for c in self.buffer) / self.sample_rate
        for max_s, silence_ms in self.SILENCE_TIERS:
            if buffered_s < max_s:
                return int(silence_ms / 1000.0 * self.sample_rate)
        return int(0.1 * self.sample_rate)  # unreachable, safety fallback

    def _soft_cut_index(self) -> int:
        """Find the best cut index within the last `lookback_samples` of the buffer
        (Strategy 2).

        Priority:
          1. Frame with the lowest VAD probability below the speech threshold.
          2. Frame with the lowest RMS energy (if all probs are above threshold).

        Returns the index into self.buffer / self.vad_probs to cut *before*.
        """
        # Determine how many tail-chunks span `lookback_samples`
        lookback_chunks = 0
        acc = 0
        for sz in (c.shape[0] for c in reversed(self.buffer)):
            acc += sz
            lookback_chunks += 1
            if acc >= self.lookback_samples:
                break

        search_start = max(0, len(self.buffer) - lookback_chunks)
        search_probs = self.vad_probs[search_start:]

        min_prob = min(search_probs)
        if min_prob < self.threshold:
            # A genuine low-activity frame exists → cut there
            local_idx = int(np.argmin(search_probs))
        else:
            # All frames look like speech → fall back to minimum RMS
            rms = [
                float(np.sqrt(np.mean(self.buffer[search_start + i] ** 2)))
                for i in range(len(search_probs))
            ]
            local_idx = int(np.argmin(rms))

        return search_start + local_idx

    def _emit(self, cut_idx: int | None = None) -> tuple[np.ndarray, str]:
        """Consume the buffer up to `cut_idx` (exclusive) and return
        (phrase_array, segment_id).  Pass None to consume the whole buffer.
        Any audio after the cut point is kept for the next segment."""
        if cut_idx is None:
            phrase = np.concatenate(self.buffer)
            seg_id = self.current_segment_id
            self.buffer = []
            self.vad_probs = []
            self.silence_accumulator = 0
            self.is_speaking = False
            self.current_segment_id = ""
        else:
            phrase = np.concatenate(self.buffer[:cut_idx])
            seg_id = self.current_segment_id
            self.buffer = self.buffer[cut_idx:]
            self.vad_probs = self.vad_probs[cut_idx:]
            self.silence_accumulator = 0
            # Continuous speech — open a new segment immediately
            self.current_segment_id = uuid.uuid4().hex[:8]

        self._samples_since_cut = 0
        self.model.reset_states()
        return phrase, seg_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, audio_chunk: np.ndarray):
        """Receive one audio chunk (typically 512 samples, 16 kHz float32).

        Returns (phrase_array, segment_id) when a phrase boundary is detected,
        otherwise returns None.
        """
        self.buffer.append(audio_chunk)

        # VAD inference — Silero expects exactly 512 samples; pad if shorter
        tensor = self._to_tensor(audio_chunk)
        chunk_len = tensor.shape[0]
        if chunk_len < 512:
            vad_tensor = torch.cat([tensor, torch.zeros(512 - chunk_len)])
        else:
            vad_tensor = tensor[:512]

        with torch.no_grad():
            speech_prob = self.model(vad_tensor, self.sample_rate).item()

        self.vad_probs.append(speech_prob)
        self._samples_since_cut += chunk_len

        if speech_prob > self.threshold:
            # ── Active speech ──────────────────────────────────────────
            if not self.is_speaking:
                self.current_segment_id = uuid.uuid4().hex[:8]
            self.is_speaking = True
            self.silence_accumulator = 0

            buffered_samples = sum(c.shape[0] for c in self.buffer)

            # Strategy 2a: absolute hard cap → soft-cut at local minimum
            if buffered_samples >= self.max_speech_samples:
                cut_idx = self._soft_cut_index()
                return self._emit(cut_idx)

            # Strategy 2b: proactive timed soft-cut — no silence detected for a while,
            # find local minimum and cut anyway to keep Whisper inputs short.
            if self._samples_since_cut >= self.soft_cut_interval_samples:
                cut_idx = self._soft_cut_index()
                return self._emit(cut_idx)

        else:
            # ── Silent frame ───────────────────────────────────────────
            if self.is_speaking:
                self.silence_accumulator += chunk_len

                # Strategy 1: dynamic silence threshold
                if self.silence_accumulator >= self._dynamic_silence_samples():
                    return self._emit(None)

        return None

    def force_emit(self):
        """Force-emit whatever is buffered (e.g. on WebSocket close)."""
        if self.buffer:
            return self._emit(None)
        return None
