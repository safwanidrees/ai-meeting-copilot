"""Speech-to-text with faster-whisper, plus filters for Whisper's known hallucinations."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

import numpy as np

# Phrases Whisper famously invents from silence or music (it was trained on subtitled
# YouTube videos). Dropped only when they make up an entire segment.
HALLUCINATIONS = {
    "you",
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "thank you so much for watching",
    "please subscribe",
    "like and subscribe",
    "bye",
    "subtitles by the amaraorg community",
    "transcription by castingwords",
    "music",
    "applause",
}


class WhisperSegment(Protocol):
    text: str
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def clean_segments(
    segments: Iterable[WhisperSegment],
    no_speech_threshold: float = 0.6,
    logprob_threshold: float = -1.0,
    compression_ratio_threshold: float = 2.4,
) -> str:
    """Join Whisper segments into one string, dropping the ones that are likely invented.

    - no_speech_prob high AND avg_logprob low: Whisper itself thinks it heard nothing
      (the same rule Whisper uses internally).
    - compression_ratio high: repetitive text loops ("the the the the ...").
    - whole segment is a stock hallucination phrase.
    """
    kept = []
    for seg in segments:
        if seg.no_speech_prob > no_speech_threshold and seg.avg_logprob < logprob_threshold:
            continue
        if seg.compression_ratio > compression_ratio_threshold:
            continue
        text = seg.text.strip()
        if not text or _normalise(text) in HALLUCINATIONS:
            continue
        kept.append(text)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


class Transcriber:
    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = "en",
        beam_size: int = 5,
        hotwords: str | None = None,
    ):
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.hotwords = hotwords
        # First run downloads the model from Hugging Face (~150 MB-1.5 GB depending on size).
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def warmup(self) -> None:
        """Run one tiny inference so the first real question isn't slowed by lazy init."""
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # Silero already cut the audio to speech
            condition_on_previous_text=False,  # stops one bad segment poisoning the next
            hotwords=self.hotwords,
        )
        return clean_segments(segments)
