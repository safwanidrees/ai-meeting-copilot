import numpy as np

from ai_meeting_copilot.audio import FRAME_SIZE, SAMPLE_RATE, FramePipeline, InputDevice, choose_device, to_dbfs

DEVICES = [
    InputDevice(0, "iPhone Microphone", 1, 48000),
    InputDevice(1, "BlackHole 2ch", 2, 48000),
    InputDevice(2, "MacBook Pro Microphone", 1, 48000),
]


def test_auto_selects_loopback_device():
    assert choose_device(DEVICES, None).name == "BlackHole 2ch"


def test_preferred_device_by_index_and_name():
    assert choose_device(DEVICES, "2").index == 2
    assert choose_device(DEVICES, "macbook").index == 2
    assert choose_device(DEVICES, "nope") is None


def test_no_loopback_returns_none():
    assert choose_device([DEVICES[0], DEVICES[2]], None) is None


def test_pipeline_resamples_stereo_48k_into_16k_frames():
    pipeline = FramePipeline(48000, highpass_hz=None)
    t = np.arange(48000) / 48000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    frames = []
    for start in range(0, 48000, 1536):  # odd-sized device blocks
        frames += pipeline.push(stereo[start : start + 1536])
    assert all(f.shape == (FRAME_SIZE,) and f.dtype == np.float32 for f in frames)
    # ~1 s in -> ~1 s out (the resampler holds back a few ms of latency)
    assert abs(len(frames) * FRAME_SIZE - SAMPLE_RATE) < 2 * FRAME_SIZE
    # The 440 Hz tone survives resampling: dominant FFT bin of a middle frame.
    mid = frames[len(frames) // 2]
    freq = np.argmax(np.abs(np.fft.rfft(mid))) * SAMPLE_RATE / FRAME_SIZE
    assert abs(freq - 440) < SAMPLE_RATE / FRAME_SIZE


def test_highpass_removes_hum_but_keeps_speech_band():
    def energy_after(freq):
        pipeline = FramePipeline(SAMPLE_RATE, highpass_hz=100)
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        frames = pipeline.push(np.sin(2 * np.pi * freq * t).astype(np.float32))
        return to_dbfs(np.concatenate(frames[10:]))  # skip filter warm-up

    assert energy_after(50) < -20  # mains hum heavily attenuated
    assert energy_after(1000) > -4  # speech-band tone essentially untouched


def test_dbfs():
    assert to_dbfs(np.zeros(512, dtype=np.float32)) < -150
    assert abs(to_dbfs(np.ones(512, dtype=np.float32))) < 1e-6
