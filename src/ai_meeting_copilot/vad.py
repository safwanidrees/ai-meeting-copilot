"""Voice activity detection: cut a continuous audio stream into utterances.

`SileroVAD` scores each 32 ms frame with a speech probability. `SpeechSegmenter`
is a small state machine over those scores that decides when someone started
talking and when they stopped. It takes the VAD as a plain callable, so it can be
unit-tested with fake probabilities.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .audio import FRAME_SIZE, SAMPLE_RATE, to_dbfs


class SileroVAD:
    """Silero VAD (ONNX runtime). Returns P(speech) for one 512-sample frame."""

    def __init__(self):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad(onnx=True)

    def __call__(self, frame: np.ndarray) -> float:
        return float(self._model(self._torch.from_numpy(frame), SAMPLE_RATE).item())

    def reset(self) -> None:
        self._model.reset_states()


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_silence_ms: int = 900
    min_speech_ms: int = 400
    speech_pad_ms: int = 300
    max_segment_s: float = 45.0
    noise_gate_dbfs: float = -55.0


class SpeechSegmenter:
    """Turns frames into utterances.

    States:
      idle      - waiting for speech; keeps a short pre-roll so the first syllable is not clipped.
      speaking  - collecting frames; counts trailing silence.

    An utterance is emitted when trailing silence reaches `min_silence_ms`, or when it
    grows past `max_segment_s`. Utterances with less than `min_speech_ms` of actual
    speech are dropped as noise.

    Hysteresis: a frame starts/continues speech at >= threshold but only counts as
    silence below (threshold - 0.15), so a speaker's soft syllables don't end the turn.
    """

    def __init__(
        self,
        vad: Callable[[np.ndarray], float],
        config: VADConfig | None = None,
        frame_size: int = FRAME_SIZE,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.vad = vad
        self.config = config or VADConfig()
        self.frame_ms = 1000.0 * frame_size / sample_rate
        self._neg_threshold = max(self.config.threshold - 0.15, 0.01)
        self._silence_limit = max(1, round(self.config.min_silence_ms / self.frame_ms))
        self._min_speech_frames = max(1, round(self.config.min_speech_ms / self.frame_ms))
        self._max_frames = max(1, round(self.config.max_segment_s * 1000 / self.frame_ms))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(0, round(self.config.speech_pad_ms / self.frame_ms)))
        self._reset()

    def _reset(self) -> None:
        self._in_speech = False
        self._buffer: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_frames = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def speech_probability(self, frame: np.ndarray) -> float:
        # Noise gate: near-silent frames are not worth a model call.
        if to_dbfs(frame) < self.config.noise_gate_dbfs:
            return 0.0
        return self.vad(frame)

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame. Returns a finished utterance, or None."""
        prob = self.speech_probability(frame)

        if not self._in_speech:
            if prob >= self.config.threshold:
                self._in_speech = True
                self._buffer = [*self._pre_roll, frame]
                self._speech_frames = 1
                self._silence_frames = 0
                self._pre_roll.clear()
            else:
                self._pre_roll.append(frame)
            return None

        self._buffer.append(frame)
        if prob >= self.config.threshold:
            self._speech_frames += 1
            self._silence_frames = 0
        elif prob < self._neg_threshold:
            self._silence_frames += 1

        if self._silence_frames >= self._silence_limit or len(self._buffer) >= self._max_frames:
            return self._finish()
        return None

    def flush(self) -> np.ndarray | None:
        """End the current utterance now (manual trigger / end of input)."""
        if not self._in_speech:
            return None
        return self._finish()

    def _finish(self) -> np.ndarray | None:
        audio = np.concatenate(self._buffer) if self._buffer else None
        long_enough = self._speech_frames >= self._min_speech_frames
        self._reset()
        return audio if long_enough else None
