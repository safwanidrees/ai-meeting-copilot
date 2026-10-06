"""Terminal output. All printing goes through one lock so threads never interleave lines."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from rich.console import Console
from rich.text import Text

from .session import Session

LABEL_STYLES = {
    "ANSWER:": "bold green",
    "FOLLOW-UP:": "bold yellow",
}


class LabelStreamer:
    """Finds section labels in a token stream so they can be coloured as they arrive.

    Tokens split labels arbitrarily ("FOLL" + "OW-UP:"), so any tail of the buffer
    that could be the start of a label is held back until the next token decides it.
    """

    def __init__(self, labels: Iterable[str] = LABEL_STYLES):
        self.labels = tuple(labels)
        self._buffer = ""

    def _held_back(self) -> int:
        for size in range(min(len(self._buffer), max(map(len, self.labels))), 0, -1):
            tail = self._buffer[-size:]
            if any(label.startswith(tail) for label in self.labels):
                return size
        return 0

    def feed(self, text: str) -> list[tuple[str, str | None]]:
        """Returns (piece, label) pairs; label is None for ordinary text."""
        self._buffer += text
        out: list[tuple[str, str | None]] = []
        while True:
            hits = [(self._buffer.find(label), label) for label in self.labels if label in self._buffer]
            if not hits:
                break
            pos, label = min(hits)
            if pos:
                out.append((self._buffer[:pos], None))
            out.append((label, label))
            self._buffer = self._buffer[pos + len(label) :]
        keep = self._held_back()
        ready = self._buffer[: len(self._buffer) - keep]
        if ready:
            out.append((ready, None))
        self._buffer = self._buffer[len(self._buffer) - keep :]
        return out

    def finish(self) -> list[tuple[str, str | None]]:
        rest, self._buffer = self._buffer, ""
        return [(rest, None)] if rest else []


class Display:
    def __init__(self, console: Console | None = None):
        self.console = console or Console(highlight=False)
        self._lock = threading.RLock()

    # markup=False everywhere: "[ MODE: AUTO ]" must print literally, not as Rich markup.
    def print(self, message: str = "", style: str | None = None) -> None:
        with self._lock:
            self.console.print(message, style=style, markup=False, soft_wrap=True)

    def try_print(self, message: str, style: str | None = "dim") -> bool:
        """Print only if nobody else is printing right now; never waits.

        For the audio thread, which must not block while an answer is streaming.
        """
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self.console.print(message, style=style, markup=False, soft_wrap=True)
        finally:
            self._lock.release()
        return True

    def dim(self, message: str) -> None:
        self.print(message, "dim")

    def warn(self, message: str) -> None:
        self.print(f"! {message}", "yellow")

    def error(self, message: str) -> None:
        self.print(f"x {message}", "bold red")

    def rule(self) -> None:
        self.print("─" * min(60, self.console.width), "dim")

    def mode(self, manual: bool) -> None:
        if manual:
            self.print("[ MODE: MANUAL — press Enter to transcribe ]", "bold magenta")
        else:
            self.print("[ MODE: AUTO — VAD will detect end of speech ]", "bold cyan")

    def help(self) -> None:
        self.print("[ Type m + Enter to toggle mode | Press Enter in manual mode to transcribe ]", "dim")
        self.print("[ Type c + Enter to clear the manual buffer | Type r + Enter to retry a failed answer ]", "dim")
        self.print("[ Type any question + Enter to ask it directly ]", "dim")
        self.print("[ Type q + Enter to end session and view chat history ]", "dim")

    def they_said(self, text: str) -> None:
        with self._lock:
            self.print()
            self.rule()
            self.console.print(Text("THEY SAID: ", style="bold cyan") + Text(text), soft_wrap=True)
            self.print()

    def stream_answer(self, chunks: Iterable[str]) -> str:
        """Print a streamed reply as it arrives; returns the full text."""
        streamer = LabelStreamer()
        received: list[str] = []
        with self._lock:
            for chunk in chunks:
                received.append(chunk)
                self._write(streamer.feed(chunk))
            self._write(streamer.finish())
            self.console.print()
            self.rule()
        return "".join(received)

    def _write(self, pieces: list[tuple[str, str | None]]) -> None:
        for piece, label in pieces:
            self.console.print(Text(piece, style=LABEL_STYLES.get(label, "")), end="", soft_wrap=True)
        self.console.file.flush()

    def timings(self, timings: dict[str, float], sources: list[str], model: str | None = None) -> None:
        parts = [f"{name} {seconds:.2f}s" for name, seconds in timings.items()]
        if model:
            parts.append(f"answered by {model}")
        if sources:
            parts.append("sources: " + ", ".join(sources))
        if parts:
            self.dim("  " + " · ".join(parts))

    def history(self, session: Session) -> None:
        with self._lock:
            self.print()
            self.print(f"══ Session {session.id} — {len(session.exchanges)} exchange(s) ══", "bold")
            for n, e in enumerate(session.exchanges, start=1):
                self.print()
                self.console.print(Text(f"{n}. [{e.timestamp[11:]}] ", style="dim") + Text("THEY SAID: ", style="bold cyan") + Text(e.heard), soft_wrap=True)
                if e.answer:
                    self.console.print(Text("   ANSWER: ", style="bold green") + Text(e.answer), soft_wrap=True)
                if e.follow_up:
                    self.console.print(Text("   FOLLOW-UP: ", style="bold yellow") + Text(e.follow_up), soft_wrap=True)
