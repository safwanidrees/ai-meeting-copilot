"""The orchestrator: wires audio -> VAD -> Whisper -> RAG -> LLM -> terminal.

Three threads, connected by a queue:

    audio thread   reads frames, runs the VAD segmenter, emits utterances
    input thread   reads your keyboard commands (m / Enter / c / q / typed questions)
    worker thread  transcribes, retrieves context and streams the answer, one job at a time

The worker is single-threaded on purpose: answers must print one after another,
and if the speaker keeps talking while an answer streams, their queued utterances
are merged into a single follow-up job instead of producing a burst of answers.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, TextIO

import numpy as np

from .audio import FRAME_SIZE, SAMPLE_RATE, AudioSource, to_dbfs
from .config import Settings
from .display import Display
from .llm import SYSTEM_PROMPT, LLMClient, build_messages, describe_error, parse_response
from .rag import HybridRetriever
from .session import Exchange, Session
from .vad import SpeechSegmenter


class TranscriberLike(Protocol):  # see transcriber.Transcriber
    def transcribe(self, audio: np.ndarray) -> str: ...


@dataclass
class Job:
    kind: str  # "audio" | "text" | "end"
    audio: np.ndarray | None = None
    text: str = ""
    forced: bool = False  # user explicitly asked (Enter / typed): bypass the min-words filter


SILENCE_HINT_S = 10.0  # warn if no sound at all reaches the device for this long


class Copilot:
    def __init__(
        self,
        *,
        settings: Settings,
        source: AudioSource,
        segmenter: SpeechSegmenter,
        transcriber: TranscriberLike,
        llm: LLMClient | None,
        retriever: HybridRetriever | None,
        display: Display,
        session: Session,
        input_stream: TextIO | None = None,
    ):
        self.settings = settings
        self.source = source
        self.segmenter = segmenter
        self.transcriber = transcriber
        self.llm = llm
        self.retriever = retriever
        self.display = display
        self.session = session
        self.input_stream = input_stream if input_stream is not None else sys.stdin

        self.manual = settings.mode == "manual"
        self._jobs: queue.Queue[Job] = queue.Queue()
        self._deferred: deque[Job] = deque()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()  # guards self.manual and self._manual_audio
        self._segmenter_lock = threading.Lock()  # segmenter is touched by audio + input threads
        self._manual_audio: list[np.ndarray] = []
        self._retry_text: str | None = None  # last question whose answer failed (r + Enter)

    # ------------------------------------------------------------------ lifecycle

    def run(self) -> Session:
        self.display.mode(self.manual)
        self.display.help()
        self.display.print()
        self.display.print("Listening... Press Ctrl+C to stop")

        audio = threading.Thread(target=self._audio_loop, name="audio", daemon=True)
        worker = threading.Thread(target=self._worker_loop, name="worker", daemon=True)
        keyboard = threading.Thread(target=self._input_loop, name="input", daemon=True)
        for thread in (audio, worker, keyboard):
            thread.start()
        try:
            while not self._stop.wait(0.2):
                pass
        except KeyboardInterrupt:
            self.display.print()
            self.display.dim("Stopping...")
        finally:
            self._stop.set()
            audio.join(timeout=2)
            worker.join(timeout=5)
        return self.session

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ audio thread

    def _audio_loop(self) -> None:
        quiet_frames = 0
        silence_limit = round(SILENCE_HINT_S * SAMPLE_RATE / FRAME_SIZE)
        try:
            for frame in self.source.frames(self._stop):
                # Tell the user if nothing audible is arriving (wrong device, no routing, muted).
                if to_dbfs(frame) < self.settings.noise_gate_dbfs:
                    quiet_frames += 1
                    if quiet_frames == silence_limit:
                        self.display.try_print(
                            f"[ no sound reaching {self.source.description} for {SILENCE_HINT_S:.0f}s: "
                            "is audio playing and loud enough? See README for BlackHole setup ]",
                            "yellow",
                        )
                else:
                    quiet_frames = 0
                with self._segmenter_lock:
                    utterance = self.segmenter.process(frame)
                if utterance is not None:
                    self._on_utterance(utterance)
        except Exception as exc:
            self.display.error(f"Audio capture failed: {exc}")
            self._stop.set()
            return

        if self._stop.is_set():
            return
        # Input ended (file replay): finish what was heard, then end the session.
        with self._segmenter_lock:
            tail = self.segmenter.flush()
        if tail is not None:
            self._on_utterance(tail)
        if self.manual:
            self.submit_manual()
        self._jobs.put(Job("end"))

    def _on_utterance(self, audio: np.ndarray) -> None:
        # Never print from here: the display lock is held while an answer streams, and the
        # audio thread must not wait on it or capture would back up and drop audio.
        with self._state_lock:
            if self.manual:
                self._manual_audio.append(audio)
                held = sum(a.size for a in self._manual_audio) / SAMPLE_RATE
            else:
                held = None
        if held is not None:
            self.display.try_print(f"[ heard {held:.1f}s of speech — press Enter to answer ]")
            return
        self._jobs.put(Job("audio", audio=audio))

    # ------------------------------------------------------------------ commands

    def submit_manual(self) -> None:
        """Manual trigger: everything heard since the last trigger becomes one question."""
        with self._segmenter_lock:
            tail = self.segmenter.flush()
        with self._state_lock:
            pieces = self._manual_audio + ([tail] if tail is not None else [])
            self._manual_audio = []
        if not pieces:
            self.display.dim("[ nothing captured yet ]")
            return
        self._jobs.put(Job("audio", audio=np.concatenate(pieces), forced=True))

    def toggle_mode(self) -> None:
        with self._state_lock:
            self.manual = not self.manual
            leftover, self._manual_audio = self._manual_audio, []
        self.display.mode(self.manual)
        if leftover:  # don't silently drop speech captured in manual mode
            self._jobs.put(Job("audio", audio=np.concatenate(leftover)))

    def handle_command(self, line: str) -> None:
        command = line.strip()
        lowered = command.lower()
        if lowered == "q":
            self._stop.set()
        elif lowered == "m":
            self.toggle_mode()
        elif lowered == "":
            if self.manual:
                self.submit_manual()
            else:
                self.display.dim("(auto mode: answers trigger by themselves — m + Enter switches to manual)")
        elif lowered == "c":
            with self._state_lock:
                self._manual_audio = []
            with self._segmenter_lock:
                self.segmenter.flush()
            self.display.dim("[ buffer cleared ]")
        elif lowered == "r":
            if self._retry_text:
                self._jobs.put(Job("text", text=self._retry_text, forced=True))
            else:
                self.display.dim("(nothing to retry)")
        elif lowered in {"h", "?", "help"}:
            self.display.help()
        else:
            self._jobs.put(Job("text", text=command, forced=True))

    def _input_loop(self) -> None:
        for line in self.input_stream:
            if self._stop.is_set():
                return
            self.handle_command(line)

    # ------------------------------------------------------------------ worker thread

    def _next_job(self, timeout: float = 0.2) -> Job | None:
        if self._deferred:
            return self._deferred.popleft()
        try:
            return self._jobs.get(timeout=timeout)
        except queue.Empty:
            return None

    def _merge_pending_audio(self, job: Job) -> Job:
        """Fold audio jobs that queued up behind this one into a single job."""
        pieces = [job.audio]
        forced = job.forced
        while True:
            try:
                nxt = self._jobs.get_nowait()
            except queue.Empty:
                break
            if nxt.kind != "audio":
                self._deferred.append(nxt)
                break
            pieces.append(nxt.audio)
            forced = forced or nxt.forced
        if len(pieces) == 1:
            return job
        return Job("audio", audio=np.concatenate(pieces), forced=forced)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            job = self._next_job()
            if job is None:
                continue
            if job.kind == "end":
                self._stop.set()
                return
            if job.kind == "audio":
                job = self._merge_pending_audio(job)
            try:
                self.process(job)
            except Exception as exc:  # one failed turn must not end the meeting
                self.display.error(describe_error(exc))
                if self._retry_text:
                    self.display.dim("  (type r + Enter to retry this question)")

    def process(self, job: Job) -> Exchange | None:
        timings: dict[str, float] = {}
        if job.kind == "audio":
            self.display.dim("[ processing... ]")
            start = time.perf_counter()
            text = self.transcriber.transcribe(job.audio)
            timings["transcribe"] = time.perf_counter() - start
            if not text:
                self.display.dim("(no speech recognised)")
                return None
        else:
            text = job.text

        if self.llm is None:  # transcribe-only mode
            self.display.they_said(text)
            exchange = Exchange(heard=text, timings=timings)
            self.session.add(exchange)
            self.display.timings(timings, [])
            return exchange

        if not job.forced and len(text.split()) < self.settings.min_words:
            self.display.dim(f'(ignored short utterance: "{text}")')
            return None

        self.display.they_said(text)

        context = []
        if self.retriever is not None:
            start = time.perf_counter()
            context = self.retriever.retrieve(text)
            timings["retrieve"] = time.perf_counter() - start

        messages = build_messages(self.session.history(self.settings.history_turns), text, context)
        start = time.perf_counter()
        first_token: list[float] = []

        def timed(stream: Iterator[str]) -> Iterator[str]:
            try:
                for piece in stream:
                    if not first_token:
                        first_token.append(time.perf_counter() - start)
                    yield piece
                    if self._stop.is_set():  # q pressed mid-answer
                        return
            finally:
                getattr(stream, "close", lambda: None)()  # release the HTTP stream now, not at GC

        try:
            raw = self.display.stream_answer(timed(self.llm.stream(SYSTEM_PROMPT, messages)))
        except Exception:
            self._retry_text = text
            raise
        self._retry_text = None
        if first_token:
            timings["first token"] = first_token[0]
        timings["answer"] = time.perf_counter() - start

        answer, follow_up = parse_response(raw)
        sources = list(dict.fromkeys(r.chunk.source for r in context))
        exchange = Exchange(heard=text, answer=answer, follow_up=follow_up, sources=sources, timings=timings)
        self.session.add(exchange)
        self.display.timings(timings, sources, self._fallback_used())
        return exchange

    def _fallback_used(self) -> str | None:
        """The model that answered, if it wasn't the configured one (OpenRouter fallback)."""
        answered = getattr(self.llm, "last_model", None)
        configured = getattr(self.llm, "model", "")
        # Providers may echo "vendor/name-20260101" for "vendor/name:free", so compare base names.
        if not answered or answered.startswith(configured.split(":")[0]):
            return None
        return answered
