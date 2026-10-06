import numpy as np

from ai_meeting_copilot.vad import SpeechSegmenter, VADConfig

FRAME = 512  # 32 ms


def frame(prob: float) -> np.ndarray:
    """A loud frame whose first sample carries the fake speech probability."""
    f = np.full(FRAME, 0.1, dtype=np.float32)
    f[0] = prob
    return f


def fake_vad(f: np.ndarray) -> float:
    return float(f[0])


def segmenter(**overrides) -> SpeechSegmenter:
    cfg = VADConfig(min_silence_ms=320, min_speech_ms=160, speech_pad_ms=64, max_segment_s=10, noise_gate_dbfs=-60)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return SpeechSegmenter(fake_vad, cfg)


def feed(seg, probs):
    return [out for p in probs if (out := seg.process(frame(p))) is not None]


def test_emits_utterance_after_silence_with_pre_roll():
    seg = segmenter()
    # 3 idle frames, 10 speech frames, 10 silence frames (silence limit = 320/32 = 10)
    out = feed(seg, [0.0] * 3 + [0.9] * 10 + [0.0] * 10)
    assert len(out) == 1
    # pre-roll (2 frames) + 10 speech + 10 trailing silence
    assert out[0].size == (2 + 10 + 10) * FRAME
    assert not seg.in_speech


def test_short_blip_is_discarded():
    seg = segmenter()
    assert feed(seg, [0.9] * 3 + [0.0] * 10) == []  # 96 ms < 160 ms min speech


def test_hysteresis_keeps_soft_syllables_in_utterance():
    seg = segmenter()
    # probs between neg threshold (0.35) and threshold (0.5) don't count as silence
    out = feed(seg, [0.9] * 6 + [0.4] * 30 + [0.9] * 6 + [0.0] * 10)
    assert len(out) == 1


def test_noise_gate_blocks_quiet_frames_even_if_vad_says_speech():
    seg = segmenter(noise_gate_dbfs=-10)  # our 0.1-amplitude frames are ~ -20 dBFS
    assert feed(seg, [0.9] * 20 + [0.0] * 10) == []


def test_max_segment_forces_a_cut():
    seg = segmenter(max_segment_s=0.64)  # 20 frames
    out = feed(seg, [0.9] * 45)
    assert len(out) == 2


def test_flush_returns_in_progress_speech():
    seg = segmenter()
    feed(seg, [0.9] * 8)
    assert seg.in_speech
    audio = seg.flush()
    assert audio is not None and audio.size == 8 * FRAME
    assert seg.flush() is None
