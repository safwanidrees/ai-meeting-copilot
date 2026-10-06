from dataclasses import dataclass

from ai_meeting_copilot.transcriber import clean_segments


@dataclass
class Seg:
    text: str
    no_speech_prob: float = 0.01
    avg_logprob: float = -0.2
    compression_ratio: float = 1.3


def test_keeps_normal_speech():
    assert clean_segments([Seg(" Can you explain"), Seg(" gradient descent?")]) == "Can you explain gradient descent?"


def test_drops_stock_hallucinations():
    assert clean_segments([Seg(" Thank you for watching!"), Seg(" You"), Seg(" What is RAG?")]) == "What is RAG?"


def test_drops_segments_whisper_thinks_are_silence():
    assert clean_segments([Seg(" something", no_speech_prob=0.9, avg_logprob=-1.5)]) == ""
    # high no-speech prob alone is not enough if the model was confident
    assert clean_segments([Seg(" Hello there", no_speech_prob=0.9, avg_logprob=-0.3)]) == "Hello there"


def test_drops_repetition_loops():
    assert clean_segments([Seg(" the the the the the the", compression_ratio=3.1)]) == ""
