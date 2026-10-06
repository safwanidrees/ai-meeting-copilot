"""Create examples/demo-interview.wav with macOS text-to-speech.

The demo audio isn't shipped in the repository: macOS system voices are licensed for
personal use, so each user generates their own copy locally.

    uv run python scripts/make_demo_audio.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

RATE = 16_000
OUT = Path(__file__).resolve().parent.parent / "examples" / "demo-interview.wav"

# (voice, text, seconds of silence before it)
LINES = [
    ("Samantha", "Good morning. Thanks for joining us today. Can you explain the difference between "
                 "generative and discriminative models in machine learning?", 1.5),
    ("Daniel", "Okay.", 2.0),
    ("Samantha", "And how would you use retrieval augmented generation with a vector database like Qdrant?", 2.0),
]


def speak(voice: str, text: str, path: Path) -> np.ndarray:
    subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text], check=True)
    with wave.open(str(path), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def main() -> int:
    if sys.platform != "darwin" or not shutil.which("say"):
        print("This script uses macOS text-to-speech (`say`). On other systems, record any short\n"
              f"interview question yourself and save it as {OUT}.")
        return 1

    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (voice, text, pause) in enumerate(LINES):
            parts.append(np.zeros(int(pause * RATE), dtype=np.float32))
            parts.append(speak(voice, text, Path(tmp) / f"{i}.wav"))
    audio = np.concatenate(parts)
    audio += np.random.default_rng(0).normal(0, 0.004, audio.size).astype(np.float32)  # light room noise

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"Wrote {OUT} ({audio.size / RATE:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
