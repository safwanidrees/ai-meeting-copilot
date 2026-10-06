"""Audio input: find a loopback device, capture it, and turn it into 16 kHz frames.

Everything downstream (Silero VAD, Whisper) wants mono float32 audio at 16 kHz in
fixed 512-sample frames (32 ms). Devices deliver whatever they like (BlackHole is
usually 48 kHz stereo), so `FramePipeline` normalises it.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soxr
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 16_000
FRAME_SIZE = 512  # 32 ms at 16 kHz: the window Silero VAD is trained on

# Virtual devices that carry *system output* rather than a microphone.
LOOPBACK_HINTS = (
    "blackhole",  # macOS
    "loopback",  # macOS (Rogue Amoeba)
    "soundflower",  # macOS (legacy)
    "cable output",  # Windows VB-Audio Virtual Cable
    "stereo mix",  # Windows
    "monitor",  # Linux PulseAudio / PipeWire "Monitor of ..."
)


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    samplerate: int

    @property
    def is_loopback(self) -> bool:
        lowered = self.name.lower()
        return any(hint in lowered for hint in LOOPBACK_HINTS)


def list_input_devices() -> list[InputDevice]:
    import sounddevice as sd

    return [
        InputDevice(i, d["name"], int(d["max_input_channels"]), int(d["default_samplerate"]))
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def default_input_device() -> InputDevice | None:
    import sounddevice as sd

    try:
        d = sd.query_devices(kind="input")
    except Exception:
        return None
    index = sd.default.device[0]
    return InputDevice(int(index), d["name"], int(d["max_input_channels"]), int(d["default_samplerate"]))


def choose_device(devices: list[InputDevice], preferred: str | None) -> InputDevice | None:
    """Pick the capture device.

    `preferred` may be an index ("1") or a case-insensitive name substring
    ("blackhole"). Without a preference, the first loopback device wins.
    Returns None when nothing suitable exists; the caller then falls back to the
    default microphone.
    """
    if preferred:
        if preferred.isdigit():
            return next((d for d in devices if d.index == int(preferred)), None)
        lowered = preferred.lower()
        return next((d for d in devices if lowered in d.name.lower()), None)
    return next((d for d in devices if d.is_loopback), None)


def to_dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
    return 20.0 * np.log10(rms + 1e-10)


class FramePipeline:
    """Raw device blocks -> mono -> 16 kHz -> high-pass filter -> 512-sample frames.

    Both the resampler and the filter are *stateful streams*, so block boundaries
    do not introduce clicks the way resampling each block independently would.
    """

    def __init__(self, input_rate: int, highpass_hz: float | None = 100.0):
        self._resampler = (
            soxr.ResampleStream(input_rate, SAMPLE_RATE, 1, dtype="float32") if input_rate != SAMPLE_RATE else None
        )
        if highpass_hz and highpass_hz > 0:
            self._sos = butter(4, highpass_hz, btype="highpass", fs=SAMPLE_RATE, output="sos")
            self._zi = np.zeros((self._sos.shape[0], 2))
        else:
            self._sos = None
            self._zi = None
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        mono = block.mean(axis=1) if block.ndim == 2 else block
        mono = np.ascontiguousarray(mono, dtype=np.float32)
        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono)
        if self._sos is not None and mono.size:
            mono, self._zi = sosfilt(self._sos, mono, zi=self._zi)
            mono = mono.astype(np.float32)
        self._pending = np.concatenate([self._pending, mono])

        n_frames = self._pending.size // FRAME_SIZE
        frames = [self._pending[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] for i in range(n_frames)]
        self._pending = self._pending[n_frames * FRAME_SIZE :]
        return frames


class AudioSource(Protocol):
    description: str

    def frames(self, stop: threading.Event) -> Iterator[np.ndarray]:
        """Yield 16 kHz mono float32 frames of FRAME_SIZE samples until `stop` is set or input ends."""
        ...


class DeviceAudioSource:
    """Live capture from a sound device (BlackHole, a microphone, ...)."""

    def __init__(self, device: InputDevice, highpass_hz: float | None = 100.0):
        self.device = device
        self.highpass_hz = highpass_hz
        self.description = f"{device.name} (index {device.index}, {device.samplerate} Hz)"
        self.dropped_blocks = 0

    def frames(self, stop: threading.Event) -> Iterator[np.ndarray]:
        import sounddevice as sd

        blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)  # ~16 s of slack

        def callback(indata, _frames, _time, _status):
            # Runs on PortAudio's real-time thread: copy and get out, never block.
            try:
                blocks.put_nowait(indata.copy())
            except queue.Full:
                self.dropped_blocks += 1

        pipeline = FramePipeline(self.device.samplerate, self.highpass_hz)
        with sd.InputStream(
            device=self.device.index,
            channels=min(2, self.device.channels),
            samplerate=self.device.samplerate,
            dtype="float32",
            blocksize=int(self.device.samplerate * 0.032),
            callback=callback,
        ):
            while not stop.is_set():
                try:
                    block = blocks.get(timeout=0.1)
                except queue.Empty:
                    continue
                yield from pipeline.push(block)


def load_audio_file(path: Path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 (ffmpeg if present, else WAV only)."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()

    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Without ffmpeg only 16-bit PCM WAV files are supported")
        rate, channels = wf.getframerate(), wf.getnchannels()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    audio = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32) / 32768.0
    return soxr.resample(audio, rate, SAMPLE_RATE).astype(np.float32) if rate != SAMPLE_RATE else audio


class FileAudioSource:
    """Replays a recording as if it were live: handy for demos, debugging and tests."""

    def __init__(self, path: Path, highpass_hz: float | None = 100.0, realtime: bool = True, tail_silence_s: float = 2.0):
        self.path = path
        self.description = f"file {path.name}"
        self._audio = load_audio_file(path)
        self._pipeline = FramePipeline(SAMPLE_RATE, highpass_hz)
        self._realtime = realtime
        self._tail = np.zeros(int(tail_silence_s * SAMPLE_RATE), dtype=np.float32)

    @property
    def duration_s(self) -> float:
        return self._audio.size / SAMPLE_RATE

    def frames(self, stop: threading.Event) -> Iterator[np.ndarray]:
        # Trailing silence lets the VAD notice the final utterance has ended.
        audio = np.concatenate([self._audio, self._tail])
        frame_s = FRAME_SIZE / SAMPLE_RATE
        start = time.perf_counter()
        for i, frame in enumerate(self._pipeline.push(audio)):
            if stop.is_set():
                return
            if self._realtime:
                delay = start + i * frame_s - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield frame
