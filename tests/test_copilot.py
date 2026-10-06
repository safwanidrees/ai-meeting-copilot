import io
import threading
import time

import numpy as np
from rich.console import Console

from ai_meeting_copilot.config import Settings
from ai_meeting_copilot.copilot import Copilot, Job
from ai_meeting_copilot.display import Display
from ai_meeting_copilot.session import Session
from ai_meeting_copilot.vad import SpeechSegmenter, VADConfig

FRAME = 512


def speech_frames(n, word_id):
    """Loud frames; sample[0] = speech prob, sample[1] = id the fake transcriber reads back."""
    frames = []
    for _ in range(n):
        f = np.full(FRAME, 0.1, dtype=np.float32)
        f[0], f[1] = 0.9, word_id
        frames.append(f)
    return frames


def silence(n):
    return [np.zeros(FRAME, dtype=np.float32) for _ in range(n)]


class ListSource:
    description = "test"

    def __init__(self, frames):
        self._frames = frames

    def frames(self, stop):
        for f in self._frames:
            if stop.is_set():
                return
            yield f


class FakeTranscriber:
    TEXT = {1: "Can you explain how hybrid search works?", 2: "Okay.", 3: "And why rerank at all?"}

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        ids = sorted({int(round(v)) for v in audio[1::FRAME] if v > 0.5})
        return " ".join(self.TEXT[i] for i in ids)


class FakeLLM:
    name = "fake"

    def __init__(self):
        self.messages = []

    def stream(self, system, messages):
        self.messages.append(messages)
        yield "ANSWER: Combine BM25 and vectors."
        yield "\n\nFOLLOW-UP: How do you fuse the scores?"


def make(frames=(), mode="auto", llm=True, **settings):
    out = io.StringIO()
    s = Settings(mode=mode, min_words=3, **settings)
    seg = SpeechSegmenter(lambda f: float(f[0]), VADConfig(min_silence_ms=320, min_speech_ms=96, speech_pad_ms=0, noise_gate_dbfs=-60))
    cop = Copilot(
        settings=s,
        source=ListSource(list(frames)),
        segmenter=seg,
        transcriber=FakeTranscriber(),
        llm=FakeLLM() if llm else None,
        retriever=None,
        display=Display(Console(file=out, width=120, force_terminal=False)),
        session=Session(),
        input_stream=io.StringIO(""),
    )
    return cop, out


def run_with_timeout(cop):
    t = threading.Thread(target=cop.run)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "copilot did not finish"


def test_auto_mode_end_to_end():
    frames = speech_frames(10, 1) + silence(12) + speech_frames(5, 2) + silence(12)
    cop, out = make(frames)
    run_with_timeout(cop)
    # The fake source is instant, so "Okay." may be merged into the question if it queued up
    # while the worker was busy (by design). Either way: exactly one answer, to the question.
    assert len(cop.session.exchanges) == 1
    e = cop.session.exchanges[0]
    assert e.heard.startswith("Can you explain how hybrid search works?")
    assert e.answer == "Combine BM25 and vectors." and e.follow_up == "How do you fuse the scores?"
    assert "[ MODE: AUTO" in out.getvalue()  # printed literally, not eaten as Rich markup


def test_auto_mode_ignores_short_utterances():
    cop, out = make()
    assert cop.process(Job("audio", audio=np.concatenate(speech_frames(5, 2)))) is None
    assert cop.session.exchanges == []
    assert 'ignored short utterance: "Okay."' in out.getvalue()


def test_conversation_memory_is_sent_to_the_llm():
    frames = speech_frames(10, 1) + silence(12) + speech_frames(10, 3) + silence(12)
    cop, _ = make(frames)
    # Process sequentially (no merging) by running jobs one at a time.
    seg_out = [o for f in frames if (o := cop.segmenter.process(f)) is not None]
    for audio in seg_out:
        cop.process(Job("audio", audio=audio))
    second_call = cop.llm.messages[1]
    assert [m["role"] for m in second_call] == ["user", "assistant", "user"]
    assert "hybrid search" in second_call[0]["content"]


def test_manual_mode_buffers_until_enter():
    cop, _ = make(mode="manual")
    for f in speech_frames(10, 1) + silence(12) + speech_frames(6, 3):  # second one still in progress
        if (u := cop.segmenter.process(f)) is not None:
            cop._on_utterance(u)
    assert cop._jobs.empty()  # nothing sent yet
    cop.handle_command("")  # Enter
    job = cop._jobs.get_nowait()
    assert job.forced
    cop.process(job)
    assert cop.session.exchanges[0].heard == "Can you explain how hybrid search works? And why rerank at all?"


def test_queued_utterances_are_merged_into_one_answer():
    cop, _ = make()
    a = np.concatenate(speech_frames(4, 1))
    b = np.concatenate(speech_frames(4, 3))
    cop._jobs.put(Job("audio", audio=a))
    cop._jobs.put(Job("audio", audio=b))
    cop._jobs.put(Job("text", text="typed question here"))
    merged = cop._merge_pending_audio(cop._jobs.get_nowait())
    assert merged.audio.size == a.size + b.size
    assert cop._next_job().text == "typed question here"  # non-audio job preserved in order


def test_typed_question_bypasses_min_words():
    cop, _ = make()
    cop.handle_command("RAG?")
    cop.process(cop._jobs.get_nowait())
    assert cop.session.exchanges[0].heard == "RAG?"


def test_toggle_mode_flushes_manual_buffer_into_auto():
    cop, _ = make(mode="manual")
    cop._manual_audio.append(np.concatenate(speech_frames(4, 1)))
    cop.handle_command("m")
    assert not cop.manual
    assert cop._jobs.get_nowait().kind == "audio"


def test_transcribe_only_records_captions():
    cop, out = make(speech_frames(10, 2) + silence(12), llm=False)
    run_with_timeout(cop)
    assert [e.heard for e in cop.session.exchanges] == ["Okay."]  # no min-words filter for captions
    assert cop.session.exchanges[0].answer == ""


def test_quit_command_stops_run():
    cop, _ = make(silence(100000))
    cop.input_stream = io.StringIO("q\n")
    run_with_timeout(cop)


def test_audio_thread_never_blocks_on_display():
    cop, _ = make()
    holder = threading.Thread(target=lambda: (cop.display._lock.acquire(), time.sleep(1), cop.display._lock.release()))
    holder.start()
    time.sleep(0.05)  # display lock now held, as during a streaming answer
    start = time.perf_counter()
    cop._on_utterance(np.concatenate(speech_frames(4, 1)))
    assert time.perf_counter() - start < 0.1
    holder.join()


class FlakyLLM(FakeLLM):
    """Fails the first call (like a 429), succeeds afterwards."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def stream(self, system, messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("rate limited")
        yield from super().stream(system, messages)


def test_failed_answer_can_be_retried_with_r():
    cop, out = make()
    cop.llm = FlakyLLM()
    cop._jobs.put(Job("text", text="What is hybrid search?", forced=True))
    cop._jobs.put(Job("end"))
    cop._worker_loop()  # processes the failing job, then stops at "end"
    assert "type r + Enter to retry" in out.getvalue()
    assert cop.session.exchanges == []
    cop.handle_command("r")
    cop.process(cop._jobs.get_nowait())
    assert cop.session.exchanges[0].heard == "What is hybrid search?"
    assert cop._retry_text is None


def test_fallback_model_is_reported_only_when_different():
    cop, _ = make()
    cop.llm.model = "qwen/qwen3.8-27b:free"
    cop.llm.last_model = "qwen/qwen3.8-27b-20260801"
    assert cop._fallback_used() is None
    cop.llm.last_model = "google/gemma-4-31b-it:free"
    assert cop._fallback_used() == "google/gemma-4-31b-it:free"


def test_manual_mode_tells_you_speech_was_heard():
    cop, out = make(mode="manual")
    cop._on_utterance(np.concatenate(speech_frames(125, 1)))  # 4.0 s
    assert "[ heard 4.0s of speech — press Enter to answer ]" in out.getvalue()
    assert cop._jobs.empty()


def test_warns_when_no_sound_reaches_the_device():
    cop, out = make(silence(round(10 * 16000 / FRAME) + 5))
    run_with_timeout(cop)
    assert "no sound reaching test for 10s" in out.getvalue()


def test_no_silence_warning_when_audio_is_present():
    cop, out = make(speech_frames(10, 2) + silence(12) + speech_frames(10, 2) + silence(12))
    run_with_timeout(cop)
    assert "no sound reaching" not in out.getvalue()
